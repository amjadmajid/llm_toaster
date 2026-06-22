"""Atomic shard publication: sealed/checksummed, partial-invisible, no overwrite, append-safe."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from llm_toaster.toaster.data.errors import ChecksumMismatchError, ConcurrentProducerError, ShardError
from llm_toaster.toaster.data.manifest import Manifest, SourceSpec, TokenizerSpec, TransformSpec, load_manifest
from llm_toaster.toaster.data.shard_store import ShardStore, read_shard


def _manifest():
    return Manifest(
        dataset_id="ds-v1",
        source=SourceSpec(dataset_name="fake", resolved_revision="rev0"),
        tokenizer=TokenizerSpec(vocab_size=256, fingerprint="sha256:x"),
        transform=TransformSpec(shard_tokens=8, dtype="uint16"),
    )


class ShardStoreTests(unittest.TestCase):
    def test_sealed_shard_is_readable_and_checksummed(self):
        with tempfile.TemporaryDirectory() as td:
            store = ShardStore(td)
            m = _manifest()
            tokens = np.arange(8, dtype=np.uint16)
            entry = store.publish_shard(m, tokens, "train")
            self.assertEqual(entry.tokens, 8)
            shard_path = store.shards_dir / "train-000000.npy"
            self.assertTrue(shard_path.exists())
            loaded = read_shard(shard_path, expected_sha256=entry.sha256)
            self.assertEqual(loaded.tolist(), tokens.tolist())

    def test_checksum_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            store = ShardStore(td)
            m = _manifest()
            store.publish_shard(m, np.arange(8, dtype=np.uint16), "train")
            shard_path = store.shards_dir / "train-000000.npy"
            with self.assertRaises(ChecksumMismatchError):
                read_shard(shard_path, expected_sha256="f" * 64)

    def test_partial_not_visible_and_failure_leaves_no_entry(self):
        with tempfile.TemporaryDirectory() as td:
            store = ShardStore(td)
            m = _manifest()
            with mock.patch("llm_toaster.toaster.data.shard_store.os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    store.publish_shard(m, np.arange(8, dtype=np.uint16), "train")
            # No committed entry, no published manifest, and no leftover .partial.
            self.assertEqual(len(m.split("train").shards), 0)
            self.assertEqual(m.generation, 0)
            self.assertFalse((store.shards_dir / "train-000000.npy").exists())
            partials = list(store.shards_dir.glob("*.partial")) if store.shards_dir.exists() else []
            self.assertEqual(partials, [])

    def test_rerun_does_not_overwrite_committed_shard(self):
        with tempfile.TemporaryDirectory() as td:
            store = ShardStore(td)
            m = _manifest()
            store.publish_shard(m, np.arange(8, dtype=np.uint16), "train")
            # Simulate a second producer that lost the manifest and tries to rewrite index 0.
            stale = _manifest()
            with self.assertRaises(ShardError):
                store.publish_shard(stale, np.arange(8, dtype=np.uint16), "train")

    def test_next_index_is_append_safe(self):
        with tempfile.TemporaryDirectory() as td:
            store = ShardStore(td)
            m = _manifest()
            store.publish_shard(m, np.arange(8, dtype=np.uint16), "train")
            store.publish_shard(m, np.arange(8, 16, dtype=np.uint16), "train")
            self.assertEqual([s.id for s in m.split("train").shards], ["train-000000", "train-000001"])
            reloaded = load_manifest(store.manifest_path)
            self.assertEqual(reloaded.next_index("train"), 2)

    def test_orphan_unmanifested_file_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as td:
            store = ShardStore(td)
            store.shards_dir.mkdir(parents=True, exist_ok=True)
            # An orphan sealed file with no manifest entry (a failed prior publish) is reclaimable.
            (store.shards_dir / "train-000000.npy").write_bytes(b"orphan")
            m = _manifest()
            entry = store.publish_shard(m, np.arange(8, dtype=np.uint16), "train")
            self.assertEqual(entry.id, "train-000000")

    def test_writer_lock_rejects_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as td:
            store = ShardStore(td)
            with store.lock():
                with self.assertRaises(ConcurrentProducerError):
                    ShardStore(td).lock().acquire()

    def test_cleanup_partials_removes_leftovers(self):
        with tempfile.TemporaryDirectory() as td:
            store = ShardStore(td)
            store.shards_dir.mkdir(parents=True, exist_ok=True)
            (store.shards_dir / "x.partial").write_bytes(b"junk")
            self.assertEqual(store.cleanup_partials(), 1)

    def test_text_fixture_read(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train_000.txt"
            p.write_text("1 2 3 4 5", encoding="utf-8")
            arr = read_shard(p)
            self.assertEqual(arr.tolist(), [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
