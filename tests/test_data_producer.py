"""Prefetch producer + coordinator. Offline: a deterministic fake document source, no network."""

import threading
import time
import unittest

import numpy as np

from llm_toaster.toaster.data.manifest import (
    Manifest,
    SourceSpec,
    TokenizerSpec,
    TransformSpec,
    load_manifest,
)
from llm_toaster.toaster.data.packing import TokenPacker
from llm_toaster.toaster.data.producer import ProducerBudgets, ShardProducer
from llm_toaster.toaster.data.shard_store import ShardStore, read_shard, sha256_file


class FakeTokenizer:
    eos_token_id = 0
    bos_token_id = None
    vocab_size = 256

    name = "fake"

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 250) + 1 for c in text]


def _doc(i: int, doc_len: int) -> str:
    return "".join(chr(33 + ((i * 7 + j) % 90)) for j in range(doc_len))


def _source_factory(n_docs: int, doc_len: int):
    docs = [{"text": _doc(i, doc_len)} for i in range(n_docs)]
    return lambda: iter(docs)


def _expected_stream(n_docs: int, doc_len: int, tok: FakeTokenizer) -> list[int]:
    stream: list[int] = []
    for i in range(n_docs):
        stream.append(0)  # EOT prepend
        stream.extend(tok.encode(_doc(i, doc_len)))
    return stream


def _new_manifest(shard_tokens: int) -> Manifest:
    return Manifest(
        dataset_id="prefetch-ds-v1",
        source=SourceSpec(dataset_name="fake", resolved_revision="rev0", text_field="text"),
        tokenizer=TokenizerSpec(type="fake", name="fake", vocab_size=256, fingerprint="sha256:tok"),
        transform=TransformSpec(add_eot=True, packing="contiguous", shard_tokens=shard_tokens, dtype="uint16"),
    )


def _make_producer(td, *, shard_tokens=20, n_docs=40, doc_len=7, val_shards=1, train_shards=3, prefetch=100, stop=None):
    store = ShardStore(td)
    manifest = _new_manifest(shard_tokens)
    packer = TokenPacker(FakeTokenizer(), add_eot=True, dtype_name="uint16")
    return ShardProducer(
        store,
        manifest,
        packer,
        _source_factory(n_docs, doc_len),
        "text",
        ProducerBudgets(val_shards=val_shards, train_shards=train_shards),
        prefetch_shards=prefetch,
        stop_event=stop,
        poll_interval_s=0.01,
    )


def _wait_until(cond, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


class ProducerTests(unittest.TestCase):
    def test_validation_frozen_and_train_complete(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            prod = _make_producer(td, val_shards=1, train_shards=2)
            prod.run()
            m = load_manifest(prod.store.manifest_path)
            self.assertTrue(m.split("validation").complete)
            self.assertEqual(len(m.split("validation").shards), 1)
            self.assertTrue(m.split("train").complete)
            self.assertEqual(len(m.split("train").shards), 2)

    def test_restart_is_append_safe_and_validation_not_remade(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            _make_producer(td, val_shards=1, train_shards=2).run()
            val_sha_before, _ = sha256_file(ShardStore(td).shards_dir / "validation-000000.npy")
            # Re-run over the same store: completed splits must not be touched or duplicated.
            _make_producer(td, val_shards=1, train_shards=2).run()
            m = load_manifest(ShardStore(td).manifest_path)
            self.assertEqual(len(m.split("train").shards), 2)
            self.assertEqual(len(m.split("validation").shards), 1)
            val_sha_after, _ = sha256_file(ShardStore(td).shards_dir / "validation-000000.npy")
            self.assertEqual(val_sha_before, val_sha_after)

    def test_carry_over_tokens_are_contiguous(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            shard_tokens, n_docs, doc_len = 20, 40, 7
            prod = _make_producer(
                td, shard_tokens=shard_tokens, n_docs=n_docs, doc_len=doc_len, val_shards=1, train_shards=2
            )
            prod.run()
            m = load_manifest(prod.store.manifest_path)
            collected = []
            for split in ("validation", "train"):
                for entry in m.split(split).shards:
                    collected.extend(read_shard(prod.store.shards_dir / f"{entry.id}.npy").tolist())
            expected = _expected_stream(n_docs, doc_len, FakeTokenizer())
            self.assertEqual(collected, expected[: len(collected)])  # no gaps or duplicates across boundaries

    def test_bounded_queue_pauses_and_resumes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            stop = threading.Event()
            prod = _make_producer(td, val_shards=1, train_shards=5, prefetch=2, stop=stop)
            thread = threading.Thread(target=prod.run)
            thread.start()
            try:
                # Producer should stall at prefetch_shards (=2) train shards with no consumer.
                self.assertTrue(_wait_until(lambda: prod.manifest.next_index("train") >= 2))
                time.sleep(0.1)
                self.assertEqual(prod.manifest.next_index("train"), 2)  # paused, not running away
                # Advance the consumer cursor; the producer must resume.
                from llm_toaster.toaster.data.statefiles import atomic_write_json

                atomic_write_json(prod.consumer_path, {"consumed_train_shards": 5})
                self.assertTrue(_wait_until(lambda: prod.manifest.next_index("train") >= 3, timeout=5))
            finally:
                stop.set()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

    def test_identical_sealed_bytes_uninterrupted_vs_resumed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td_ref, tempfile.TemporaryDirectory() as td_run:
            # Uninterrupted reference run.
            _make_producer(td_ref, val_shards=1, train_shards=3).run()

            # Interrupted run: stop after the first train shard, then resume in a fresh producer.
            stop = threading.Event()
            prod_a = _make_producer(td_run, val_shards=1, train_shards=3, prefetch=100, stop=stop)
            thread = threading.Thread(target=prod_a.run)
            thread.start()
            self.assertTrue(_wait_until(lambda: prod_a.manifest.next_index("train") >= 1))
            stop.set()
            thread.join(timeout=5)
            _make_producer(td_run, val_shards=1, train_shards=3).run()  # resume to completion

            ref_store, run_store = ShardStore(td_ref), ShardStore(td_run)
            for idx in range(3):
                ref_sha, _ = sha256_file(ref_store.shards_dir / f"train-{idx:06d}.npy")
                run_sha, _ = sha256_file(run_store.shards_dir / f"train-{idx:06d}.npy")
                self.assertEqual(ref_sha, run_sha, f"train shard {idx} bytes differ after resume")

    def test_failure_sets_failed_status_and_reraises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            store = ShardStore(td)
            manifest = _new_manifest(20)
            packer = TokenPacker(FakeTokenizer(), add_eot=True, dtype_name="uint16")

            def bad_factory():
                def gen():
                    yield {"text": "abc"}
                    raise RuntimeError("source blew up")

                return gen()

            prod = ShardProducer(
                store, manifest, packer, bad_factory, "text", ProducerBudgets(0, 3), poll_interval_s=0.01
            )
            with self.assertRaises(RuntimeError):
                prod.run()
            from llm_toaster.toaster.data.statefiles import read_json

            status = read_json(prod.status_path)
            self.assertEqual(status["status"], "failed")
            self.assertIn("traceback", status)


def _fake_producer_target(spec, stop_event):
    """A stand-in producer for coordinator tests: materialize a tiny frozen dataset, then idle."""
    store = ShardStore(spec["store_dir"], spec.get("manifest_path"))
    manifest = _new_manifest(8)
    store.publish_shard(manifest, np.arange(8, dtype=np.uint16), "validation")
    manifest.mark_complete("validation")
    manifest.save_atomic(store.manifest_path)
    store.publish_shard(manifest, np.arange(8, dtype=np.uint16), "train")
    from llm_toaster.toaster.data.statefiles import atomic_write_json

    atomic_write_json(store.store_dir / ".producer_status.json", {"status": "running", "last_train_shard": 0})
    stop_event.wait(10)


class CoordinatorTests(unittest.TestCase):
    def test_coordinator_starts_waits_and_cleans_up(self):
        import tempfile

        from llm_toaster.toaster.data.coordinator import PrefetchCoordinator

        with tempfile.TemporaryDirectory() as td:
            spec = {"store_dir": td, "val_shards": 1, "train_shards": 3}
            # fork avoids re-importing the test module (spawn) for this lifecycle test.
            coord = PrefetchCoordinator(spec, target=_fake_producer_target, start_method="fork")
            coord.start()
            try:
                coord.wait_until_ready(min_ready_shards=1, timeout_s=10, poll_interval_s=0.05)
                manifest = load_manifest(coord.manifest_path)
                self.assertTrue(manifest.split("validation").complete)
                self.assertGreaterEqual(manifest.next_index("train"), 1)
                coord.update_consumer(1)
            finally:
                coord.stop(timeout_s=5)
            self.assertIsNotNone(coord._process)
            self.assertFalse(coord._process.is_alive())


if __name__ == "__main__":
    unittest.main()
