"""Metrics formatting + JSONL writer (torch-free)."""

import json
import os
import tempfile
import unittest

from llm_toaster.toaster.training.metrics import (
    JsonlMetricsWriter,
    format_duration,
    format_metrics_line,
    human_bytes,
    human_count,
)


class FormattingTests(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual(format_duration(45), "45s")
        self.assertEqual(format_duration(711), "11m51s")
        self.assertEqual(format_duration(7740), "2h09m")

    def test_human_count_and_bytes(self):
        self.assertEqual(human_count(11_403_264), "11.4M")
        self.assertEqual(human_count(500), "500")
        self.assertEqual(human_bytes(536_870_912), "512.0 MB")

    def test_format_metrics_line_has_key_fields(self):
        line = format_metrics_line(
            {
                "step": 139,
                "max_iter": 100_000,
                "loss": 7.5664,
                "lr": 6e-4,
                "grad_norm": 1.23,
                "tokens_per_sec": 21098,
                "tokens_seen": 11_403_264,
                "elapsed_s": 711,
                "eta_s": 7740,
            }
        )
        for token in ("step", "loss 7.5664", "tok/s", "eta"):
            self.assertIn(token, line)


class JsonlWriterTests(unittest.TestCase):
    def test_writes_parseable_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sub", "metrics.jsonl")  # parent dir auto-created
            writer = JsonlMetricsWriter(path)
            writer.write({"type": "architecture", "params_total": 10})
            writer.write({"type": "step", "step": 1, "loss": 2.0})
            writer.close()
            rows = [json.loads(line) for line in open(path, encoding="utf-8")]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["type"], "architecture")
            self.assertEqual(rows[1]["step"], 1)

    def test_empty_path_disables_writing(self):
        writer = JsonlMetricsWriter("")
        writer.write({"step": 1})  # no-op, must not raise
        writer.close()


if __name__ == "__main__":
    unittest.main()
