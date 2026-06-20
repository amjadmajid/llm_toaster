"""Phase 4 robustness: determinism, perplexity, checkpoint versioning/loading."""

import math
import os
import tempfile
import unittest

import numpy as np
import torch

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.models.registry import build_model
from llm_toaster.toaster.peft.lora import inject_lora
from llm_toaster.toaster.training.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    load_state_dict_any,
    save_checkpoint,
)
from llm_toaster.toaster.training.engine import model_size_summary, perplexity, seed_everything


def _tiny_config():
    config = ConfigHandler()
    config.model.vocab_size = 64
    config.model.n_embd = 16
    config.model.n_head = 4
    config.model.n_blocks = 1
    config.model.seq_len = 8
    return config


class SeedTests(unittest.TestCase):
    def test_seed_everything_is_reproducible(self):
        seed_everything(123)
        torch_a, numpy_a = torch.rand(4), np.random.rand(3)
        seed_everything(123)
        torch_b, numpy_b = torch.rand(4), np.random.rand(3)
        self.assertTrue(torch.equal(torch_a, torch_b))
        self.assertTrue(np.allclose(numpy_a, numpy_b))


class ModelSizeTests(unittest.TestCase):
    def test_summary_reports_total_and_trainable_drop_with_lora(self):
        model = build_model(_tiny_config())
        total = sum(p.numel() for p in model.parameters())
        summary = model_size_summary(model)
        self.assertIsInstance(summary, str)
        self.assertIn(f"{total:,}", summary)

        inject_lora(model, ConfigHandler().peft)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLess(trainable, total)
        self.assertIn(f"{trainable:,} trainable", model_size_summary(model))


class PerplexityTests(unittest.TestCase):
    def test_perplexity_matches_exp_and_caps(self):
        self.assertAlmostEqual(perplexity(0.0), 1.0)
        self.assertAlmostEqual(perplexity(1.0), math.e)
        self.assertEqual(perplexity(1e9), float("inf"))  # capped, no OverflowError


class CheckpointLoadingTests(unittest.TestCase):
    def test_format_version_written(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "ckpt.pt")
            save_checkpoint(ckpt, build_model(_tiny_config()), config=_tiny_config(), global_step=1)
            raw = torch.load(ckpt, weights_only=False)
            self.assertEqual(raw["format_version"], CHECKPOINT_FORMAT_VERSION)

    def test_load_state_dict_any_handles_checkpoint_and_bare_state_dict(self):
        config = _tiny_config()
        model = build_model(config)
        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "ckpt.pt")
            llm = os.path.join(td, "model.llm")
            save_checkpoint(ckpt, model, config=config, global_step=1)
            torch.save(model.state_dict(), llm)

            from_ckpt = build_model(config)
            from_ckpt.load_state_dict(load_state_dict_any(ckpt))
            from_bare = build_model(config)
            from_bare.load_state_dict(load_state_dict_any(llm))

            self.assertTrue(torch.equal(from_ckpt.token_embeddings.weight, model.token_embeddings.weight))
            self.assertTrue(torch.equal(from_bare.token_embeddings.weight, model.token_embeddings.weight))


if __name__ == "__main__":
    unittest.main()
