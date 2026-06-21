"""Model factory matrix (norms x ffn x attention) and LoRA adapter behavior."""

import os
import tempfile
import unittest

import torch

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.models.feedforward import GEGLUFFN, GELUFFN, SwiGLUFFN
from llm_toaster.toaster.models.registry import build_model
from llm_toaster.toaster.peft.lora import LoRALinear, inject_lora, lora_state_dict, merge_lora


def _params(model):
    return sum(p.numel() for p in model.parameters())


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

    def test_moe_is_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            build_model(_tiny_config(ffn="moe"))

    def test_gpt_style_init_uses_small_std(self):
        # Default PyTorch init gives embeddings std ~1.0; GPT-style init must be ~0.02.
        config = _tiny_config(n_blocks=2)
        model = build_model(config)
        self.assertLess(float(model.token_embeddings.weight.detach().std()), 0.1)
        # Residual output projections are scaled down further (0.02 / sqrt(2*n_blocks)).
        o_proj = model.TransformerBlocks[0].attention.o_proj
        self.assertLess(float(o_proj.weight.detach().std()), 0.02)


class PositionAndFFNTests(unittest.TestCase):
    def test_rope_and_none_skip_position_table(self):
        x = torch.ones(2, 8, dtype=torch.long)
        for position in ("rope", "none"):
            model = build_model(_tiny_config(position=position))
            self.assertIsNone(model.position_embeddings, msg=position)
            self.assertEqual(model(x).shape, (2, 8, 64), msg=position)
        self.assertIsNotNone(build_model(_tiny_config(position="learned")).position_embeddings)

    def test_rope_requires_even_head_dim(self):
        config = _tiny_config(position="rope")
        config.model.n_embd = 12  # head_dim = 12/4 = 3 (odd) -> RoPE must reject
        with self.assertRaises(ValueError):
            build_model(config)

    def test_geglu_is_real_gated_not_aliased_to_gelu(self):
        blocks = {
            kind: build_model(_tiny_config(ffn=kind)).TransformerBlocks[0].feed_forward
            for kind in ("gelu", "geglu", "swiglu")
        }
        self.assertIsInstance(blocks["gelu"], GELUFFN)
        self.assertIsInstance(blocks["geglu"], GEGLUFFN)  # NOT a GELUFFN alias
        self.assertIsInstance(blocks["swiglu"], SwiGLUFFN)
        # Gated FFNs carry an extra input projection, so they have more params than plain GELU.
        self.assertGreater(
            _params(build_model(_tiny_config(ffn="geglu"))), _params(build_model(_tiny_config(ffn="gelu")))
        )

    def test_ffn_mult_scales_params(self):
        self.assertLess(_params(build_model(_tiny_config(ffn_mult=2))), _params(build_model(_tiny_config(ffn_mult=8))))


class KVCacheTests(unittest.TestCase):
    def test_cached_matches_uncached_greedy(self):
        # Greedy (top_k=1) is deterministic, so a correct KV-cache must reproduce the exact tokens.
        ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
        for position in ("learned", "rope", "none"):
            for kv in (None, 2, 1):
                config = _tiny_config(position=position, num_key_value_heads=kv, n_blocks=2, seq_len=32)
                config.attention.backend = "eager"
                model = build_model(config).eval()
                uncached = model.generate_text(ids, 8, top_k=1)
                cached = model.generate_cached(ids, 8, top_k=1)
                self.assertTrue(torch.equal(uncached, cached), msg=f"{position}/kv={kv}")

    def test_generate_cached_runs_on_sdpa_and_stops_at_seq_len(self):
        config = _tiny_config(position="rope", n_blocks=2, seq_len=8)
        config.attention.backend = "sdpa"
        model = build_model(config).eval()
        out = model.generate_cached(torch.tensor([[1, 2, 3]]), 50, top_k=5, top_p=0.9)
        self.assertLessEqual(out.shape[1], 8)  # bounded by seq_len, no crash past the context window


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
