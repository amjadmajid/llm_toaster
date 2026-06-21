"""Architecture card generator: parameter table accuracy and Mermaid/structure content."""

import unittest

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.models.registry import build_model
from llm_toaster.toaster.report import architecture_report, parameter_table


def _cfg(**model_overrides):
    config = ConfigHandler()
    config.model.vocab_size = 128
    config.model.n_embd = 16
    config.model.n_head = 4
    config.model.n_blocks = 2
    config.model.seq_len = 8
    for key, value in model_overrides.items():
        setattr(config.model, key, value)
    return config


class ParameterTableTests(unittest.TestCase):
    def test_table_sums_to_total_with_tied_head(self):
        model = build_model(_cfg(tie_embeddings=True))
        rows, total = parameter_table(model)
        self.assertEqual(total, sum(p.numel() for p in model.parameters()))
        self.assertEqual(sum(count for _, count in rows), total)  # tied lm_head=0 avoids double count

    def test_untied_head_counts_lm_head(self):
        model = build_model(_cfg(tie_embeddings=False))
        rows, total = parameter_table(model)
        self.assertEqual(sum(count for _, count in rows), total)
        self.assertGreater(dict(rows)["lm_head"], 0)


class ReportTests(unittest.TestCase):
    def test_report_has_diagrams_table_and_real_total(self):
        config = _cfg()
        model = build_model(config)
        report = architecture_report(model, config)
        self.assertEqual(report.count("```mermaid"), 2)  # dataflow + block diagrams
        self.assertIn("Parameters by component", report)
        self.assertIn(f"{sum(p.numel() for p in model.parameters()):,}", report)

    def test_rope_card_skips_position_row_and_notes_rope(self):
        config = _cfg(position="rope")
        model = build_model(config)
        names = [name for name, _ in parameter_table(model)[0]]
        self.assertNotIn("position_embeddings", names)
        self.assertIn("RoPE", architecture_report(model, config))


if __name__ == "__main__":
    unittest.main()
