"""Issue #2 confirmation: weight init and parameter counting work via the legacy surfaces.

Issue #2 referenced ``from model import TransformerModel`` and ``from utils import ... count_parameters``.
The canonical init lives in ``models/transformer.py`` (``_init_weights``/``_init_parameters``), not a
missing ``utils.init_weights``. This confirms the legacy positional shim still instantiates, runs a
forward pass on random input, reports a parameter count, and initializes deterministically under a seed.
"""

import unittest
import warnings

import torch


def _legacy_model_cls():
    # Importing the deprecated shim emits a DeprecationWarning at module load; ignore it here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from model import TransformerModel

    return TransformerModel


def _build(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return _legacy_model_cls()(n_head=4, vocab_size=64, n_embd=16, seq_len=8, n_blocks=2)


class LegacyModelShimTests(unittest.TestCase):
    def test_positional_constructor_forwards_random_input(self):
        model = _build()
        logits = model(torch.randint(0, 64, (2, 8)))
        self.assertEqual(logits.shape, (2, 8, 64))

    def test_count_parameters_reports_millions_string(self):
        from utils import count_parameters

        result = count_parameters(_build())
        self.assertIsInstance(result, str)
        self.assertTrue(result.endswith("M"))
        self.assertGreater(float(result[:-1]), 0.0)

    def test_initialization_is_deterministic_under_seed(self):
        a = _build(seed=0)
        b = _build(seed=0)
        self.assertTrue(torch.equal(a.token_embeddings.weight, b.token_embeddings.weight))
        self.assertTrue(torch.equal(a.lm_head.weight, b.lm_head.weight))


if __name__ == "__main__":
    unittest.main()
