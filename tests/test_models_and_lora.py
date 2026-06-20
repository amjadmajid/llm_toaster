"""Model factory matrix (norms x ffn x attention) and LoRA adapter behavior."""

import os
import tempfile
import unittest

import torch

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.models.registry import build_model
from llm_toaster.toaster.peft.lora import LoRALinear, inject_lora, lora_state_dict, merge_lora


def _tiny_config(**overrides):
    config = ConfigHandler()
    config.model.vocab_size = 64
    config.model.n_embd = 16
    config.model.n_head = 4
    config.model.n_blocks = 1
    config.model.seq_len = 8
    for key, value in overrides.items():
        setattr(config.model, key, value)
    return config


class ModelMatrixTests(unittest.TestCase):
    def test_forward_shape_across_norm_ffn_backend(self):
        x = torch.ones(2, 8, dtype=torch.long)
        for norm in ("layernorm", "rmsnorm"):
            for ffn in ("gelu", "geglu", "swiglu"):
                for backend in ("eager", "sdpa"):
                    config = _tiny_config(norm=norm, ffn=ffn)
                    config.attention.backend = backend
                    out = build_model(config)(x)
                    self.assertEqual(out.shape, (2, 8, 64), msg=f"{norm}/{ffn}/{backend}")

    def test_grouped_query_attention_shape(self):
        config = _tiny_config(num_key_value_heads=2)
        out = build_model(config)(torch.ones(2, 8, dtype=torch.long))
        self.assertEqual(out.shape, (2, 8, 64))

    def test_unimplemented_features_raise(self):
        with self.assertRaises(NotImplementedError):
            build_model(_tiny_config(ffn="moe"))
        with self.assertRaises(NotImplementedError):
            build_model(_tiny_config(position="rope"))


class LoRATests(unittest.TestCase):
    def test_inject_freezes_base_and_exposes_adapters(self):
        model = build_model(_tiny_config())
        config = ConfigHandler()
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        inject_lora(model, config.peft)
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLess(trainable_after, trainable_before)
        # q/k/v/o per block become LoRALinear -> 2 adapter tensors each.
        adapters = lora_state_dict(model)
        self.assertEqual(len(adapters), 8)
        self.assertTrue(all("lora_" in key for key in adapters))

    def test_inject_raises_when_no_target_matches(self):
        model = build_model(_tiny_config())
        config = ConfigHandler()
        config.peft.target_modules = ["does_not_exist"]
        with self.assertRaises(ValueError):
            inject_lora(model, config.peft)

    def test_adapter_save_and_load_roundtrip(self):
        model = build_model(_tiny_config())
        inject_lora(model, ConfigHandler().peft)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "adapter.pt")
            torch.save(lora_state_dict(model), path)
            fresh = build_model(_tiny_config())
            inject_lora(fresh, ConfigHandler().peft)
            missing, unexpected = fresh.load_state_dict(torch.load(path, weights_only=True), strict=False)
            self.assertEqual(unexpected, [])  # every saved key belongs to the fresh model

    def test_merge_removes_lora_modules_and_preserves_output(self):
        model = build_model(_tiny_config())
        model.eval()  # disable dropout so the comparison is deterministic
        x = torch.ones(1, 8, dtype=torch.long)
        inject_lora(model, ConfigHandler().peft)
        with torch.no_grad():
            before = model(x)
            merge_lora(model)
            self.assertFalse(any(isinstance(m, LoRALinear) for m in model.modules()))
            after = model(x)
        # lora_B initializes to zero, so merging is a no-op on the forward pass.
        self.assertTrue(torch.allclose(before, after, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
