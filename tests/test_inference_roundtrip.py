"""Extract -> load -> generate round-trip over the unified engine stack.

Mirrors what extract_inference_model.py and inference.py now do: a full engine
checkpoint is reduced to a plain state_dict, reloaded into a registry-built
model, and used for generation through the shared helper. Uses a fake tokenizer
so the test needs no network.
"""

import os
import tempfile
import unittest

import torch

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.generation import generate
from llm_toaster.toaster.models.registry import build_model
from llm_toaster.toaster.training.checkpointing import load_checkpoint, save_checkpoint


class FakeTokenizer:
    eos_token_id = 2
    vocab_size = 128

    def encode(self, s, add_special_tokens=False):
        return [3 + (ord(c) % 100) for c in s]

    def decode(self, ids):
        return "".join(chr((i - 3) % 100) for i in ids if i > 2)


def _tiny_config():
    config = ConfigHandler()
    config.model.vocab_size = 128
    config.model.n_embd = 16
    config.model.n_head = 4
    config.model.n_blocks = 1
    config.model.seq_len = 8
    return config


class InferenceRoundTripTests(unittest.TestCase):
    def test_extract_then_generate(self):
        config = _tiny_config()
        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "ckpt.pt")
            llm = os.path.join(td, "model.llm")

            trained = build_model(config)
            save_checkpoint(ckpt, trained, config=config, global_step=1, tokens_seen=8)

            # extract_inference_model.py path: load full checkpoint, save weights only.
            extracted = build_model(config)
            load_checkpoint(ckpt, extracted, device="cpu", strict=False)
            torch.save(extracted.state_dict(), llm)

            # inference.py path: registry model + weights_only load + shared generate.
            infer_model = build_model(config)
            infer_model.load_state_dict(torch.load(llm, weights_only=True))

            text = generate(infer_model, FakeTokenizer(), "hi", "cpu", max_new_tokens=4, top_k=5, top_p=0.9)
            self.assertIsInstance(text, str)
            # Weights survive the round-trip unchanged.
            self.assertTrue(torch.equal(trained.token_embeddings.weight, infer_model.token_embeddings.weight))


if __name__ == "__main__":
    unittest.main()
