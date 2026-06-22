"""Batch sources: prepared shard source, growing-manifest wait, legacy adapter, direct stream.

All offline (no network). A fake deterministic document iterable stands in for HF streaming.
"""

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from llm_toaster.toaster.data.errors import (
    DataExhausted,
    DataWaitTimeout,
    ProducerFailedError,
    ResumeIncompatibleError,
)
from llm_toaster.toaster.data.hf_source import HFDirectTokenSource
from llm_toaster.toaster.data.legacy import LegacyShardDirSource, migrate_legacy_directory
from llm_toaster.toaster.data.manifest import Manifest, SourceSpec, TokenizerSpec, TransformSpec, load_manifest
from llm_toaster.toaster.data.packing import TokenPacker
from llm_toaster.toaster.data.shard_source import ManifestShardSource
from llm_toaster.toaster.data.shard_store import ShardStore


class FakeTokenizer:
    eos_token_id = 0
    bos_token_id = None
    vocab_size = 256

    def __init__(self, name="fake"):
        self.name = name

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 250) + 1 for c in text]


def _build_store(td, *, shard_tokens=8, dtype="uint16", resolved_revision="rev0", dataset_id="ds-v1"):
    store = ShardStore(td)
    manifest = Manifest(
        dataset_id=dataset_id,
        source=SourceSpec(dataset_name="fake", resolved_revision=resolved_revision),
        tokenizer=TokenizerSpec(vocab_size=256, fingerprint="sha256:tok"),
        transform=TransformSpec(shard_tokens=shard_tokens, dtype=dtype),
    )
    return store, manifest


class ManifestShardSourceTests(unittest.TestCase):
    def test_normal_batches_single_shift_and_multi_shard(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td)
            store.publish_shard(m, np.arange(0, 10, dtype=np.uint16), "train")
            store.publish_shard(m, np.arange(100, 110, dtype=np.uint16), "train")
            src = ManifestShardSource(str(store.manifest_path), "train", batch_size=1, seq_len=3)
            x, y, info = src.next_batch()
            self.assertEqual(x.tolist(), [[0, 1, 2]])
            self.assertEqual(y.tolist(), [[1, 2, 3]])
            self.assertEqual(info.shard_id, "train-000000")
            # Exhaust the first shard, transition to the second.
            seen_second = False
            for _ in range(10):
                _, _, info = src.next_batch()
                if info.shard_id == "train-000001":
                    seen_second = True
                    break
            self.assertTrue(seen_second)

    def test_short_shard_raises(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td)
            store.publish_shard(m, np.arange(4, dtype=np.uint16), "train")
            src = ManifestShardSource(str(store.manifest_path), "train", batch_size=2, seq_len=8)
            with self.assertRaises(Exception):
                src.next_batch()

    def test_stop_does_not_wrap(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td)
            store.publish_shard(m, np.arange(10, dtype=np.uint16), "train")
            src = ManifestShardSource(str(store.manifest_path), "train", batch_size=1, seq_len=3, exhaustion="stop")
            count = 0
            with self.assertRaises(DataExhausted):
                for _ in range(100):
                    src.next_batch()
                    count += 1
            self.assertGreater(count, 0)
            self.assertEqual(src.pass_index, 0)

    def test_repeat_wraps_and_counts_passes(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td)
            store.publish_shard(m, np.arange(10, dtype=np.uint16), "train")
            src = ManifestShardSource(str(store.manifest_path), "train", batch_size=1, seq_len=3, exhaustion="repeat")
            infos = [src.next_batch()[2] for _ in range(10)]
            self.assertTrue(any(i.repeated for i in infos))
            self.assertGreaterEqual(infos[-1].pass_index, 1)

    def test_deterministic_shard_shuffle(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td)
            for i in range(4):
                store.publish_shard(m, np.arange(i * 100, i * 100 + 10, dtype=np.uint16), "train")
            a = ManifestShardSource(str(store.manifest_path), "train", 1, 3, shuffle="shard", seed=42)
            b = ManifestShardSource(str(store.manifest_path), "train", 1, 3, shuffle="shard", seed=42)
            self.assertEqual(a._order, b._order)

    def test_state_dict_roundtrip_resumes_exact_next_batch(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td, shard_tokens=50)
            store.publish_shard(m, np.arange(0, 50, dtype=np.uint16), "train")
            store.publish_shard(m, np.arange(100, 150, dtype=np.uint16), "train")
            src = ManifestShardSource(str(store.manifest_path), "train", batch_size=1, seq_len=3)
            for _ in range(4):
                src.next_batch()
            saved = src.state_dict()
            expected_x, _, _ = src.next_batch()

            resumed = ManifestShardSource(str(store.manifest_path), "train", batch_size=1, seq_len=3)
            resumed.load_state_dict(saved)
            got_x, _, _ = resumed.next_batch()
            self.assertEqual(got_x.tolist(), expected_x.tolist())

    def test_resume_rejects_changed_revision(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td, resolved_revision="rev0")
            store.publish_shard(m, np.arange(10, dtype=np.uint16), "train")
            src = ManifestShardSource(str(store.manifest_path), "train", 1, 3)
            src.next_batch()
            saved = src.state_dict()
            saved["resolved_revision"] = "DIFFERENT"
            with self.assertRaises(ResumeIncompatibleError):
                ManifestShardSource(str(store.manifest_path), "train", 1, 3).load_state_dict(saved)

    def test_resume_rejects_changed_shard_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td)
            store.publish_shard(m, np.arange(10, dtype=np.uint16), "train")
            src = ManifestShardSource(str(store.manifest_path), "train", 1, 3)
            src.next_batch()
            saved = src.state_dict()
            saved["shard_sha256"] = "deadbeef"
            with self.assertRaises(ResumeIncompatibleError):
                ManifestShardSource(str(store.manifest_path), "train", 1, 3).load_state_dict(saved)


class GrowingManifestWaitTests(unittest.TestCase):
    def test_wait_discovers_appended_shard_via_manifest_refresh(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td)
            store.publish_shard(m, np.arange(10, dtype=np.uint16), "train")
            src = ManifestShardSource(
                str(store.manifest_path), "train", 1, 3, exhaustion="wait", wait_timeout_s=5, poll_interval_s=0.01
            )
            # Drain the first shard (10 tokens, T=3 -> 3 full batches before exhaustion).
            for _ in range(3):
                src.next_batch()
            # Append a second shard; the source must find it through a manifest refresh.
            store.publish_shard(m, np.arange(100, 110, dtype=np.uint16), "train")
            _, _, info = src.next_batch()
            self.assertEqual(info.shard_id, "train-000001")

    def test_wait_times_out_clearly(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td)
            store.publish_shard(m, np.arange(10, dtype=np.uint16), "train")
            src = ManifestShardSource(
                str(store.manifest_path), "train", 1, 3, exhaustion="wait", wait_timeout_s=0.05, poll_interval_s=0.01
            )
            with self.assertRaises(DataWaitTimeout):
                for _ in range(100):
                    src.next_batch()

    def test_wait_propagates_producer_failure(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td)
            store.publish_shard(m, np.arange(10, dtype=np.uint16), "train")
            status = {"status": "failed", "error": "producer blew up"}
            src = ManifestShardSource(
                str(store.manifest_path),
                "train",
                1,
                3,
                exhaustion="wait",
                wait_timeout_s=5,
                poll_interval_s=0.01,
                status_provider=lambda: status,
            )
            with self.assertRaises(ProducerFailedError):
                for _ in range(100):
                    src.next_batch()

    def test_wait_clean_exhaustion_on_producer_complete(self):
        with tempfile.TemporaryDirectory() as td:
            store, m = _build_store(td)
            store.publish_shard(m, np.arange(10, dtype=np.uint16), "train")
            status = {"status": "complete"}
            src = ManifestShardSource(
                str(store.manifest_path),
                "train",
                1,
                3,
                exhaustion="wait",
                wait_timeout_s=5,
                poll_interval_s=0.01,
                status_provider=lambda: status,
            )
            with self.assertRaises(DataExhausted):
                for _ in range(100):
                    src.next_batch()


class LegacyAndMigrationTests(unittest.TestCase):
    def _write_legacy(self, directory):
        np.save(Path(directory) / "shard_000000_train.npy", np.arange(0, 64, dtype=np.uint16))
        np.save(Path(directory) / "shard_000000_val.npy", np.arange(0, 64, dtype=np.uint16))

    def test_legacy_source_wraps(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_legacy(td)
            src = LegacyShardDirSource(td, "train", batch_size=1, seq_len=8)
            infos = [src.next_batch()[2] for _ in range(20)]
            self.assertTrue(any(i.repeated for i in infos))  # legacy path wraps (deprecated behavior)

    def test_migrate_then_read_via_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_legacy(td)
            manifest_path = Path(td) / "manifest.json"
            migrate_legacy_directory(td, manifest_path, tokenizer=FakeTokenizer())
            m = load_manifest(manifest_path)
            self.assertTrue(m.split("train").complete)
            self.assertEqual(len(m.split("train").shards), 1)
            self.assertEqual(len(m.split("validation").shards), 1)
            src = ManifestShardSource(str(manifest_path), "train", batch_size=1, seq_len=8)
            x, y, _ = src.next_batch()
            self.assertEqual(x.tolist(), [list(range(8))])


class FakeStatefulIterable:
    """Stands in for a resumable HF IterableDataset."""

    def __init__(self, docs):
        self.docs = docs
        self._pos = 0

    def __iter__(self):
        while self._pos < len(self.docs):
            doc = self.docs[self._pos]
            self._pos += 1
            yield {"text": doc}

    def state_dict(self):
        return {"pos": self._pos}

    def load_state_dict(self, state):
        self._pos = int(state.get("pos", 0))


class DirectSourceTests(unittest.TestCase):
    def _packer(self):
        return TokenPacker(FakeTokenizer(), add_eot=True, dtype_name="uint16")

    def test_emits_eot_and_packs_contiguously(self):
        docs = ["abcdef", "ghijkl", "mnopqr", "stuvwx"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            src = HFDirectTokenSource(
                FakeStatefulIterable(docs), self._packer(), "text", 1, 3, resolved_revision="rev0"
            )
            x, y, info = src.next_batch()
        self.assertEqual(x.shape, (1, 3))
        self.assertEqual(int(x[0][0]), 0)  # leading EOT from the first document
        self.assertEqual(info.source_mode, "direct")

    def test_experimental_warning_emitted(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            HFDirectTokenSource(FakeStatefulIterable(["x" * 50]), self._packer(), "text", 1, 3, resolved_revision="r")
        self.assertTrue(any("EXPERIMENTAL" in str(w.message) for w in caught))

    def test_requires_resolved_revision(self):
        with self.assertRaises(ValueError):
            HFDirectTokenSource(FakeStatefulIterable(["x"]), self._packer(), "text", 1, 3, resolved_revision=None)

    def test_pending_buffer_persists_and_resumes(self):
        docs = ["abcdefghij" for _ in range(6)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            src = HFDirectTokenSource(
                FakeStatefulIterable(docs), self._packer(), "text", 1, 4, resolved_revision="rev0"
            )
            src.next_batch()
            saved = src.state_dict()
            expected_x, _, _ = src.next_batch()

            resumed = HFDirectTokenSource(
                FakeStatefulIterable(docs), self._packer(), "text", 1, 4, resolved_revision="rev0"
            )
            resumed.load_state_dict(saved)
            got_x, _, _ = resumed.next_batch()
        self.assertEqual(got_x.tolist(), expected_x.tolist())
        self.assertIn("pending_buffer", saved)

    def test_resume_rejects_changed_revision(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            src = HFDirectTokenSource(
                FakeStatefulIterable(["x" * 50]), self._packer(), "text", 1, 3, resolved_revision="rev0"
            )
            saved = src.state_dict()
            saved["resolved_revision"] = "other"
            target = HFDirectTokenSource(
                FakeStatefulIterable(["x" * 50]), self._packer(), "text", 1, 3, resolved_revision="rev0"
            )
            with self.assertRaises(ResumeIncompatibleError):
                target.load_state_dict(saved)

    def test_missing_state_api_raises_actionable_error(self):
        class Stateless:
            def __iter__(self):
                yield {"text": "hello world"}

        from llm_toaster.toaster.data.errors import HFDependencyError

        with self.assertRaises(HFDependencyError):
            HFDirectTokenSource(
                Stateless(), self._packer(), "text", 1, 3, resolved_revision="r", require_state_api=True
            )


if __name__ == "__main__":
    unittest.main()
