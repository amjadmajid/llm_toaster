"""data CLI (prepare/inspect/validate/migrate-legacy) + legacy downloader wrapper. Offline."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from llm_toaster.toaster.data import cli
from llm_toaster.toaster.data.errors import ShardError
from llm_toaster.toaster.data.manifest import load_manifest


def _write_config(td, store_dir, corpus_path):
    cfg = Path(td) / "cfg.yaml"
    cfg.write_text(
        "training:\n"
        "  batch_size: 1\n  n_batches: 1\n  seq_len: 8\n  max_iter: 5\n"
        "data:\n"
        f"  manifest_path: {store_dir}/manifest.json\n"
        "  source:\n    type: local\n"
        f"    dataset_name: {corpus_path}\n    text_field: text\n"
        "  transform:\n    shard_tokens: 64\n    dtype: uint16\n"
        f"  materialization:\n    mode: prepared\n    store_dir: {store_dir}\n"
        "  validation:\n    tokens: 64\n",
        encoding="utf-8",
    )
    return str(cfg)


def _corpus(td):
    path = Path(td) / "corpus.txt"
    path.write_text("\n".join(f"document number {i} has words" for i in range(120)), encoding="utf-8")
    return str(path)


class PrepareCliTests(unittest.TestCase):
    def test_dry_run_emits_plan_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            store = str(Path(td) / "store")
            cfg = _write_config(td, store, _corpus(td))
            self.assertEqual(cli.main(["prepare", "--config", cfg, "--dry-run"]), 0)
            self.assertFalse(Path(store, "manifest.json").exists())

    def test_prepare_materializes_manifest_then_inspect_and_validate(self):
        with tempfile.TemporaryDirectory() as td:
            store = str(Path(td) / "store")
            cfg = _write_config(td, store, _corpus(td))
            self.assertEqual(cli.main(["prepare", "--config", cfg]), 0)
            manifest_path = str(Path(store) / "manifest.json")
            manifest = load_manifest(manifest_path)
            self.assertTrue(manifest.split("validation").complete)
            self.assertGreaterEqual(manifest.next_index("train"), 1)

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["inspect", "--manifest", manifest_path]), 0)
            self.assertIn("dataset_id", out.getvalue())

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["validate", "--manifest", manifest_path]), 0)
            self.assertIn("validation OK", out.getvalue())

    def test_validate_detects_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            store = str(Path(td) / "store")
            cfg = _write_config(td, store, _corpus(td))
            cli.main(["prepare", "--config", cfg])
            manifest_path = str(Path(store) / "manifest.json")
            # Corrupt the first train shard's bytes.
            shard = next(Path(store, "shards").glob("train-000000.npy"))
            shard.write_bytes(b"\x00" * shard.stat().st_size)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = cli.main(["validate", "--manifest", manifest_path])
            self.assertEqual(rc, 1)
            self.assertIn("FAIL", out.getvalue())

    def test_prepare_refuses_to_clobber_unmanifested_legacy_shards(self):
        from llm_toaster.toaster.config import ConfigHandler

        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "store"
            store.mkdir()
            np.save(store / "shard_000000_train.npy", np.arange(64, dtype=np.uint16))  # legacy, unmanifested
            config = ConfigHandler()
            config.data.source.type = "local"
            config.data.source.dataset_name = _corpus(td)
            config.data.materialization.store_dir = str(store)
            config.data.manifest_path = str(store / "manifest.json")
            config.data.transform.shard_tokens = 64
            with self.assertRaises(ShardError):
                cli.prepare_from_config(config, train_shards=1, val_shards=0)


class MigrateCliTests(unittest.TestCase):
    def test_migrate_legacy_directory(self):
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "old"
            data_dir.mkdir()
            np.save(data_dir / "shard_000000_train.npy", np.arange(64, dtype=np.uint16))
            np.save(data_dir / "shard_000000_val.npy", np.arange(64, dtype=np.uint16))
            manifest_path = str(data_dir / "manifest.json")
            rc = cli.main(["migrate-legacy", "--data-dir", str(data_dir), "--manifest", manifest_path])
            self.assertEqual(rc, 0)
            manifest = load_manifest(manifest_path)
            self.assertTrue(manifest.split("train").complete)
            self.assertTrue(manifest.split("validation").complete)


class LegacyDownloaderWrapperTests(unittest.TestCase):
    def test_args_map_to_new_pipeline(self):
        from dataspace.src import download_tokenize_hf as dl

        captured = {}

        def fake_prepare(config, *, train_shards=None, val_shards=None, dry_run=False):
            captured["dataset_name"] = config.data.source.dataset_name
            captured["config_name"] = config.data.source.config_name
            captured["shard_tokens"] = config.data.transform.shard_tokens
            captured["train_shards"] = train_shards
            captured["val_shards"] = val_shards

        with mock.patch.object(dl, "prepare_from_config", side_effect=fake_prepare):
            ok = dl.download_and_tokenize(
                dataset_name="foo/bar",
                remote_name="sample-10BT",
                split_ratio=0.75,
                output_dir="/tmp/store",
                shard_size=1234,
                stream=True,
                max_shards=4,
            )
        self.assertTrue(ok)
        self.assertEqual(captured["dataset_name"], "foo/bar")
        self.assertEqual(captured["config_name"], "sample-10BT")
        self.assertEqual(captured["shard_tokens"], 1234)
        self.assertEqual(captured["val_shards"], 1)  # round(4 * 0.25)
        self.assertEqual(captured["train_shards"], 3)

    def test_dataspace_import_still_works(self):
        from dataspace import DataLoaderLite, download_and_tokenize

        self.assertIsNotNone(DataLoaderLite)
        self.assertTrue(callable(download_and_tokenize))


if __name__ == "__main__":
    unittest.main()
