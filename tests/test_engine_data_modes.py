"""Engine integration over the manifest-backed data layer: prepared/prefetch, max_tokens,
explicit exhaustion, fixed validation, and resume compatibility. Offline + CPU-only."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.data.errors import ResumeIncompatibleError
from llm_toaster.toaster.data.manifest import (
    Manifest,
    SourceSpec,
    TokenizerSpec,
    TransformSpec,
    load_manifest,
)
from llm_toaster.toaster.data.shard_store import ShardStore
from llm_toaster.toaster.training.engine import TrainingEngine


def _make_manifest(
    store_dir, *, train_shards, val_shards, shard_tokens, vocab=256, revision="rev0", dataset_id="ds-v1"
):
    store = ShardStore(store_dir)
    manifest = Manifest(
        dataset_id=dataset_id,
        source=SourceSpec(dataset_name="fake", resolved_revision=revision),
        tokenizer=TokenizerSpec(type="tiktoken", name="gpt2", vocab_size=vocab, fingerprint="sha256:tok"),
        transform=TransformSpec(shard_tokens=shard_tokens, dtype="uint16"),
    )
    rng = np.random.default_rng(0)
    for _ in range(val_shards):
        store.publish_shard(manifest, rng.integers(0, vocab, size=shard_tokens, dtype=np.uint16), "validation")
    if val_shards:
        manifest.mark_complete("validation")
        manifest.save_atomic(store.manifest_path)
    for _ in range(train_shards):
        store.publish_shard(manifest, rng.integers(0, vocab, size=shard_tokens, dtype=np.uint16), "train")
    return store, manifest


def _cfg(
    td, manifest_path, *, B=1, T=3, NB=1, max_iter=10, max_tokens=None, vocab=256, exhaustion="stop", eval_every=0
):
    cfg = ConfigHandler()
    cfg.training.device = "cpu"
    cfg.distributed.mixed_precision = "no"
    cfg.training.batch_size = B
    cfg.training.n_batches = NB
    cfg.training.seq_len = T
    cfg.training.max_iter = max_iter
    cfg.training.max_tokens = max_tokens
    cfg.model.vocab_size = vocab
    cfg.model.n_embd = 16
    cfg.model.n_head = 4
    cfg.model.n_blocks = 1
    cfg.model.seq_len = T
    cfg.data.manifest_path = str(manifest_path)
    cfg.data.materialization.mode = "prepared"
    cfg.data.sampling.exhaustion = exhaustion
    cfg.data.validation.reset_each_eval = True
    cfg.evaluation.eval_every_steps = eval_every
    cfg.evaluation.eval_steps = 50
    cfg.logging.log_file = os.path.join(td, "log.txt")
    cfg.logging.metrics_file = os.path.join(td, "metrics.jsonl")
    cfg.checkpointing.output_dir = td
    cfg.training.ckpt = os.path.join(td, "ckpt")
    cfg.training.ckpt_config = os.path.join(td, "ckpt_config.yaml")
    return cfg


class PreparedModeEngineTests(unittest.TestCase):
    def test_prepared_trains_and_stops_at_exhaustion_with_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            store, _ = _make_manifest(td, train_shards=1, val_shards=0, shard_tokens=10)
            cfg = _cfg(td, store.manifest_path, B=1, T=3, NB=1, max_iter=100, exhaustion="stop")
            engine = TrainingEngine(cfg).train()
            self.assertTrue(engine._data_exhausted)
            self.assertLess(engine.global_step, 100)  # stopped at exhaustion, did not wrap
            self.assertEqual(engine.data_pass, 0)  # stop never starts a new pass
            self.assertTrue(os.path.exists(cfg.training.ckpt))  # clean final checkpoint

    def test_repeat_mode_wraps_and_counts_passes(self):
        with tempfile.TemporaryDirectory() as td:
            store, _ = _make_manifest(td, train_shards=1, val_shards=0, shard_tokens=10)
            cfg = _cfg(td, store.manifest_path, B=1, T=3, NB=1, max_iter=12, exhaustion="repeat")
            engine = TrainingEngine(cfg).train()
            self.assertEqual(engine.global_step, 12)  # repeat keeps going to max_iter
            self.assertGreaterEqual(engine.data_pass, 1)
            self.assertGreater(engine.repeated_tokens_seen, 0)

    def test_discarded_partial_step_does_not_advance_global_step(self):
        # 10 tokens, B=1 T=3 -> 3 batches available. With n_batches=2: step1 ok (2 batches),
        # step2 takes 1 batch then exhausts on the 2nd -> discarded, global_step stays 1.
        with tempfile.TemporaryDirectory() as td:
            store, _ = _make_manifest(td, train_shards=1, val_shards=0, shard_tokens=10)
            cfg = _cfg(td, store.manifest_path, B=1, T=3, NB=2, max_iter=50, exhaustion="stop")
            engine = TrainingEngine(cfg).train()
            self.assertEqual(engine.global_step, 1)
            self.assertEqual(engine.tokens_seen, 1 * 3 * 2)  # only the one completed step counted

    def test_max_tokens_stops_within_budget(self):
        with tempfile.TemporaryDirectory() as td:
            store, _ = _make_manifest(td, train_shards=3, val_shards=0, shard_tokens=100)
            cfg = _cfg(td, store.manifest_path, B=1, T=3, NB=1, max_iter=100, max_tokens=9, exhaustion="repeat")
            engine = TrainingEngine(cfg).train()
            self.assertEqual(engine.global_step, 3)  # 9 tokens / 3 per step
            self.assertLessEqual(engine.tokens_seen, 9)

    def test_first_limit_wins_between_max_iter_and_max_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            store, _ = _make_manifest(td, train_shards=3, val_shards=0, shard_tokens=100)
            cfg = _cfg(td, store.manifest_path, B=1, T=3, NB=1, max_iter=2, max_tokens=900, exhaustion="repeat")
            engine = TrainingEngine(cfg).train()
            self.assertEqual(engine.global_step, 2)  # max_iter is the tighter bound


class FixedValidationTests(unittest.TestCase):
    def test_validation_resets_each_eval_so_evals_are_identical(self):
        with tempfile.TemporaryDirectory() as td:
            store, _ = _make_manifest(td, train_shards=2, val_shards=1, shard_tokens=100)
            cfg = _cfg(td, store.manifest_path, B=1, T=3, NB=1, max_iter=2, eval_every=1)
            engine = TrainingEngine(cfg)
            engine.setup_tokenizer()
            engine.setup_model()
            engine.setup_dataloaders()
            engine.setup_optimizer()
            engine.setup_scheduler()
            engine.setup_scaler()
            first = engine.eval_step()
            second = engine.eval_step()  # reset_each_eval -> same examples -> identical loss
            self.assertIsNotNone(first)
            self.assertAlmostEqual(first, second, places=6)

    def test_oversized_eval_steps_does_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            store, _ = _make_manifest(td, train_shards=2, val_shards=1, shard_tokens=20)
            cfg = _cfg(td, store.manifest_path, B=1, T=3, NB=1, max_iter=2, eval_every=1)
            engine = TrainingEngine(cfg).train()  # eval_steps=50 over a tiny val split must not crash
            self.assertIsNotNone(engine.last_val_loss)


class ResumeCompatibilityTests(unittest.TestCase):
    def _train_once(self, td, **kw):
        store, manifest = _make_manifest(td, train_shards=2, val_shards=0, shard_tokens=100)
        cfg = _cfg(td, store.manifest_path, B=1, T=3, NB=1, max_iter=2, exhaustion="repeat")
        TrainingEngine(cfg).train()
        return store, cfg

    def test_resume_over_same_manifest_continues(self):
        with tempfile.TemporaryDirectory() as td:
            store, cfg = self._train_once(td)
            cfg.training.max_iter = 4
            cfg.checkpointing.resume_from_checkpoint = cfg.training.ckpt
            resumed = TrainingEngine(cfg).train()
            self.assertTrue(resumed.resumed)
            self.assertEqual(resumed.global_step, 4)

    def test_resume_rejects_changed_revision(self):
        with tempfile.TemporaryDirectory() as td:
            store, cfg = self._train_once(td)
            # Rewrite the manifest with a changed resolved revision.
            manifest = load_manifest(store.manifest_path)
            manifest.source.resolved_revision = "DIFFERENT"
            manifest.dataset_fingerprint = manifest.compute_dataset_fingerprint()
            manifest.save_atomic(store.manifest_path)
            cfg.training.max_iter = 4
            cfg.checkpointing.resume_from_checkpoint = cfg.training.ckpt
            with self.assertRaises(ResumeIncompatibleError):
                TrainingEngine(cfg).train()

    def test_appended_shard_stays_resume_compatible(self):
        with tempfile.TemporaryDirectory() as td:
            store, cfg = self._train_once(td)
            # Append a new train shard after the checkpoint; the committed prefix is unchanged.
            manifest = load_manifest(store.manifest_path)
            store.publish_shard(manifest, np.arange(100, dtype=np.uint16), "train")
            cfg.training.max_iter = 4
            cfg.checkpointing.resume_from_checkpoint = cfg.training.ckpt
            resumed = TrainingEngine(cfg).train()  # must not raise
            self.assertEqual(resumed.global_step, 4)


class PrefetchEngineSmokeTest(unittest.TestCase):
    def test_prefetch_end_to_end_with_local_source(self):
        previous = os.environ.get("LLM_TOASTER_MP_START")
        os.environ["LLM_TOASTER_MP_START"] = "fork"  # reliable + fast for the offline test
        try:
            with tempfile.TemporaryDirectory() as td:
                corpus = Path(td) / "corpus.txt"
                corpus.write_text("\n".join(f"line number {i} with some words here" for i in range(200)), "utf-8")
                store_dir = os.path.join(td, "store")

                cfg = ConfigHandler()
                cfg.training.device = "cpu"
                cfg.distributed.mixed_precision = "no"
                cfg.training.batch_size = 1
                cfg.training.n_batches = 1
                cfg.training.seq_len = 8
                cfg.training.max_iter = 3
                cfg.model.n_embd = 16
                cfg.model.n_head = 4
                cfg.model.n_blocks = 1
                cfg.model.seq_len = 8
                cfg.data.source.type = "local"
                cfg.data.source.dataset_name = str(corpus)
                cfg.data.transform.shard_tokens = 64
                cfg.data.materialization.mode = "prefetch"
                cfg.data.materialization.store_dir = store_dir
                cfg.data.materialization.min_ready_shards = 1
                cfg.data.materialization.wait_timeout_s = 60
                cfg.data.sampling.exhaustion = "wait"
                cfg.data.validation.tokens = 64
                cfg.evaluation.eval_every_steps = 0
                cfg.logging.log_file = os.path.join(td, "log.txt")
                cfg.logging.metrics_file = os.path.join(td, "metrics.jsonl")
                cfg.checkpointing.output_dir = td
                cfg.training.ckpt = os.path.join(td, "ckpt")
                cfg.training.ckpt_config = os.path.join(td, "ckpt_config.yaml")

                engine = TrainingEngine(cfg).train()
                self.assertGreaterEqual(engine.global_step, 1)
                manifest = load_manifest(os.path.join(store_dir, "manifest.json"))
                self.assertTrue(manifest.split("validation").complete)  # validation frozen first
                self.assertGreaterEqual(manifest.next_index("train"), 1)
        finally:
            if previous is None:
                os.environ.pop("LLM_TOASTER_MP_START", None)
            else:
                os.environ["LLM_TOASTER_MP_START"] = previous


if __name__ == "__main__":
    unittest.main()
