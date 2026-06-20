"""End-to-end engine smoke runs on tiny fixtures (pretrain + finetune) and a
checkpoint resume that restores trained weights exactly.

The tokenizer falls back to a byte-level encoder when tiktoken assets cannot be
downloaded, so these tests run offline.
"""

import os
import tempfile
import unittest

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
        import json

        with tempfile.TemporaryDirectory() as td:
            config = _smoke_cfg(td, "pretrain")
            TrainingEngine(config).train()
            rows = [json.loads(line) for line in open(config.logging.metrics_file, encoding="utf-8")]
            types = [r["type"] for r in rows]
            self.assertEqual(types[0], "architecture")
            self.assertIn("step", types)
            step_rows = [r for r in rows if r["type"] == "step"]
            self.assertTrue(all({"loss", "lr", "tokens_per_sec"} <= r.keys() for r in step_rows))

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


if __name__ == "__main__":
    unittest.main()
