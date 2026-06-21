"""End-to-end engine smoke runs on tiny fixtures (pretrain + finetune) and a
checkpoint resume that restores trained weights exactly.

The tokenizer falls back to a byte-level encoder when tiktoken assets cannot be
downloaded, so these tests run offline.
"""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.training.engine import TrainingEngine


def _smoke_cfg(td, mode="pretrain"):
    config = ConfigHandler.from_yaml("config/smoke_test_config.yaml")
    config.training.mode = mode
    config.finetune.enabled = mode in {"finetune", "sft"}
    config.training.device = "cpu"
    config.distributed.mixed_precision = "no"
    config.logging.log_file = os.path.join(td, "log.txt")
    config.logging.metrics_file = os.path.join(td, "metrics.jsonl")
    config.checkpointing.output_dir = td
    config.training.ckpt = os.path.join(td, "base_ckpt")
    config.training.ckpt_config = os.path.join(td, "base_config.yaml")
    config.finetune.output_ckpt = os.path.join(td, "instruct_ckpt")
    config.finetune.output_config = os.path.join(td, "instruct_config.yaml")
    return config


class TrainingSmokeTests(unittest.TestCase):
    def test_pretrain_runs_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            config = _smoke_cfg(td, "pretrain")
            engine = TrainingEngine(config).train()
            self.assertEqual(engine.global_step, config.training.max_iter)
            self.assertTrue(os.path.exists(config.training.ckpt))

    def test_metrics_file_has_architecture_and_step_rows(self):
        with tempfile.TemporaryDirectory() as td:
            config = _smoke_cfg(td, "pretrain")
            TrainingEngine(config).train()
            rows = [json.loads(line) for line in open(config.logging.metrics_file, encoding="utf-8")]
            types = [r["type"] for r in rows]
            self.assertEqual(types[0], "architecture")
            self.assertIn("step", types)
            step_rows = [r for r in rows if r["type"] == "step"]
            self.assertTrue(all({"loss", "lr", "tokens_per_sec"} <= r.keys() for r in step_rows))

    def test_structured_logging_csv_jsonl_and_validation(self):
        # Issue #7/#8: per-step metrics in both JSONL and CSV; architecture row carries git/config;
        # validation loss is surfaced in the records; oversized eval_steps must not crash (shards wrap).
        with tempfile.TemporaryDirectory() as td:
            config = _smoke_cfg(td, "pretrain")
            config.training.max_iter = 2
            config.logging.log_every_steps = 1
            config.evaluation.eval_every_steps = 1
            config.evaluation.eval_steps = 50  # more than available val batches -> must wrap, not crash
            TrainingEngine(config).train()

            rows = [json.loads(line) for line in open(config.logging.metrics_file, encoding="utf-8")]
            arch = next(r for r in rows if r["type"] == "architecture")
            self.assertIn("git_commit", arch)
            self.assertIn("config_path", arch)
            step_rows = [r for r in rows if r["type"] == "step"]
            self.assertTrue(step_rows)
            for record in step_rows:
                self.assertTrue({"step", "loss", "tokens_per_sec", "elapsed_s"} <= record.keys())
                self.assertIn("peak_mem_reserved_bytes", record)
            self.assertTrue(any(r.get("val_loss") is not None for r in step_rows))  # validation surfaced

            csv_path = str(Path(config.logging.metrics_file).with_suffix(".csv"))
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertTrue(csv_rows)
            for record in csv_rows:
                for key in ("step", "loss", "tokens_per_sec", "elapsed_s"):
                    self.assertIn(key, record)
                    self.assertNotEqual(record[key], "")

    def test_finetune_runs_on_jsonl_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            config = _smoke_cfg(td, "finetune")
            engine = TrainingEngine(config).train()
            self.assertEqual(engine.global_step, config.training.max_iter)
            self.assertTrue(os.path.exists(config.finetune.output_ckpt))

    def test_resume_restores_trained_weights(self):
        with tempfile.TemporaryDirectory() as td:
            config = _smoke_cfg(td, "pretrain")
            trained = TrainingEngine(config).train()
            reference = trained.model.token_embeddings.weight.detach().clone()

            resumed = TrainingEngine(_smoke_cfg(td, "pretrain"))
            resumed.setup_tokenizer()
            resumed.setup_model()
            resumed.setup_optimizer()
            resumed.setup_scheduler()
            resumed.setup_scaler()
            resumed.load_checkpoint(config.training.ckpt)
            self.assertTrue(torch.allclose(resumed.model.token_embeddings.weight, reference))
            self.assertEqual(resumed.global_step, trained.global_step)

    def test_resume_continues_to_same_step_count_and_tokens(self):
        # Issue #6: train N uninterrupted vs. train K -> checkpoint -> resume to N. The resumed run
        # must CONTINUE (not restart): same final step count and tokens, with elapsed time persisted.
        with tempfile.TemporaryDirectory() as ref_td, tempfile.TemporaryDirectory() as run_td:
            ref_cfg = _smoke_cfg(ref_td, "pretrain")
            ref_cfg.training.max_iter = 2
            reference = TrainingEngine(ref_cfg).train()

            k_cfg = _smoke_cfg(run_td, "pretrain")
            k_cfg.training.max_iter = 1
            interrupted = TrainingEngine(k_cfg).train()
            self.assertEqual(interrupted.global_step, 1)  # stopped early

            resume_cfg = _smoke_cfg(run_td, "pretrain")
            resume_cfg.training.max_iter = 2
            resume_cfg.checkpointing.resume_from_checkpoint = resume_cfg.training.ckpt
            resumed = TrainingEngine(resume_cfg).train()

            self.assertEqual(resumed.global_step, 2)  # continued 1 -> 2, did not restart
            self.assertEqual(resumed.global_step, reference.global_step)
            self.assertEqual(resumed.tokens_seen, reference.tokens_seen)
            self.assertGreater(resumed.wall_clock_s, 0.0)  # elapsed seconds persisted across resume

    def test_resume_restores_data_position(self):
        with tempfile.TemporaryDirectory() as td:
            config = _smoke_cfg(td, "pretrain")
            trained = TrainingEngine(config).train()

            resumed = TrainingEngine(_smoke_cfg(td, "pretrain"))
            resumed.setup_tokenizer()
            resumed.setup_model()
            resumed.setup_dataloaders()  # fresh loader starts at shard 0
            resumed.setup_optimizer()
            resumed.setup_scheduler()
            resumed.setup_scaler()
            resumed.load_checkpoint(config.training.ckpt)  # must re-seek the data cursor
            self.assertEqual(resumed.train_loader.current_shard, trained.train_loader.current_shard)
            self.assertEqual(resumed.train_loader.current_position, trained.train_loader.current_position)


if __name__ == "__main__":
    unittest.main()
