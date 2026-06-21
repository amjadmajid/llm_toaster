"""On-device benchmark core: energy integration and the generation timing loop (CPU)."""

import unittest

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.experiments.bench import NullSampler, benchmark_generation, integrate_energy
from llm_toaster.toaster.models.registry import build_model


class FakeTokenizer:
    eos_token_id = 2
    vocab_size = 128

    def encode(self, s, add_special_tokens=False):
        return [3 + (ord(c) % 100) for c in s] or [3]

    def decode(self, ids):
        return "".join(chr((i - 3) % 100) for i in ids if i > 2)


def _tiny_model():
    config = ConfigHandler()
    config.model.vocab_size = 128
    config.model.n_embd = 16
    config.model.n_head = 4
    config.model.n_blocks = 1
    config.model.seq_len = 16
    return build_model(config)


class EnergyIntegrationTests(unittest.TestCase):
    def test_trapezoid(self):
        self.assertEqual(integrate_energy([(0, 10), (1, 10), (2, 10)]), 20.0)  # 10 W for 2 s
        self.assertEqual(integrate_energy([(0, 0), (1, 20)]), 10.0)  # ramp
        self.assertEqual(integrate_energy([(0, 5)]), 0.0)  # need >= 2 samples


class BenchmarkLoopTests(unittest.TestCase):
    def test_benchmark_generation_reports_metrics(self):
        result = benchmark_generation(
            _tiny_model(), FakeTokenizer(), "cpu", prompt="hi", max_new_tokens=4, warmup_tokens=2, sampler=NullSampler()
        )
        self.assertGreaterEqual(result["generated_tokens"], 1)
        self.assertGreater(result["total_s"], 0)
        self.assertGreater(result["decode_tokens_per_sec"], 0)
        self.assertGreaterEqual(result["ttft_s"], 0)
        self.assertEqual(result["energy_joules"], 0.0)  # NullSampler
        self.assertIsNone(result["energy_per_token"])
        self.assertGreaterEqual(result["peak_mem_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
