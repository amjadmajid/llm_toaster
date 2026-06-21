"""Analytic parameter count must match the real model, and the matched-param solver."""

import unittest

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.models.registry import build_model
from llm_toaster.toaster.models.sizing import estimate_params, solve_for_target_params


def _config(**model_overrides):
    config = ConfigHandler()
    config.model.vocab_size = 256
    config.model.n_embd = 32
    config.model.n_head = 4
    config.model.n_blocks = 2
    config.model.seq_len = 16
    for key, value in model_overrides.items():
        setattr(config.model, key, value)
    return config


class EstimateParamsTests(unittest.TestCase):
    def test_matches_real_model_across_variants(self):
        for ffn in ("gelu", "geglu", "swiglu"):
            for norm in ("layernorm", "rmsnorm"):
                for position in ("learned", "rope", "none"):
                    for tie in (True, False):
                        for kv in (None, 1, 2):
                            config = _config(
                                ffn=ffn, norm=norm, position=position, tie_embeddings=tie, num_key_value_heads=kv
                            )
                            actual = sum(p.numel() for p in build_model(config).parameters())
                            self.assertEqual(
                                estimate_params(config),
                                actual,
                                msg=f"{ffn}/{norm}/{position}/tie={tie}/kv={kv}",
                            )

    def test_ffn_mult_affects_estimate(self):
        self.assertLess(estimate_params(_config(ffn_mult=2)), estimate_params(_config(ffn_mult=8)))


class SolveTargetTests(unittest.TestCase):
    def test_solve_n_embd_hits_target_within_tolerance(self):
        target = 2_000_000
        solved = solve_for_target_params(_config(), target, vary="n_embd")
        achieved = estimate_params(solved)
        self.assertEqual(sum(p.numel() for p in build_model(solved).parameters()), achieved)  # still buildable
        self.assertEqual(solved.model.n_embd % solved.model.n_head, 0)
        self.assertLess(abs(achieved - target) / target, 0.1)

    def test_solve_n_embd_keeps_even_head_dim_for_rope(self):
        solved = solve_for_target_params(_config(position="rope"), 1_500_000, vary="n_embd")
        head_dim = solved.model.n_embd // solved.model.n_head
        self.assertEqual(head_dim % 2, 0)
        build_model(solved)  # must not raise the RoPE even-head_dim error

    def test_solve_n_blocks_hits_target(self):
        target = 3_000_000
        solved = solve_for_target_params(_config(), target, vary="n_blocks")
        self.assertGreaterEqual(solved.model.n_blocks, 1)
        self.assertLess(abs(estimate_params(solved) - target) / target, 0.25)


if __name__ == "__main__":
    unittest.main()
