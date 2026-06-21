"""Sweep runner + aggregator, end-to-end on the tiny smoke config."""

import tempfile
import unittest

from llm_toaster.toaster.experiments.aggregate import aggregate_runs, to_markdown
from llm_toaster.toaster.experiments.sweep import build_runs, run_sweep


def _spec(output_dir):
    return {
        "base_config": "config/smoke_test_config.yaml",
        "output_dir": output_dir,
        "seeds": [0],
        "overrides": {
            "training.max_iter": 2,
            "training.device": "cpu",
            "distributed.mixed_precision": "no",
            "logging.log_every_steps": 1,
        },
        "axes": {"model.ffn": ["gelu", "swiglu"]},
    }


class BuildRunsTests(unittest.TestCase):
    def test_one_variable_at_a_time_cells(self):
        with tempfile.TemporaryDirectory() as td:
            runs = build_runs(_spec(td))
            self.assertEqual([r["run_name"] for r in runs], ["ffn=gelu__seed0", "ffn=swiglu__seed0"])
            self.assertEqual(runs[0]["config"].model.ffn, "gelu")
            self.assertEqual(runs[1]["config"].model.ffn, "swiglu")
            # Per-run metrics file is namespaced under the run dir.
            self.assertTrue(runs[0]["config"].logging.metrics_file.endswith("ffn=gelu__seed0/metrics.jsonl"))


class SweepEndToEndTests(unittest.TestCase):
    def test_run_sweep_then_aggregate(self):
        with tempfile.TemporaryDirectory() as td:
            spec = _spec(td)
            run_dirs = run_sweep(spec)
            self.assertEqual(len(run_dirs), 2)
            for run_dir in run_dirs:
                self.assertTrue((run_dir / "metrics.jsonl").exists())
                self.assertTrue((run_dir / "done").exists())

            rows = aggregate_runs(td)
            self.assertEqual(len(rows), 2)
            ffns = sorted(r["ffn"] for r in rows)
            self.assertEqual(ffns, ["gelu", "swiglu"])
            for row in rows:
                self.assertIsNotNone(row["params_total"])
                self.assertIsNotNone(row["loss"])
                self.assertGreater(row["flops_per_token"], 0)
            self.assertIn("perplexity", to_markdown(rows))

    def test_completed_runs_are_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            spec = _spec(td)
            run_dirs = run_sweep(spec)
            before = {d / "metrics.jsonl": (d / "metrics.jsonl").stat().st_mtime_ns for d in run_dirs}
            run_sweep(spec)  # second pass should skip (done markers exist), not retrain
            for metrics_path, mtime in before.items():
                self.assertEqual(metrics_path.stat().st_mtime_ns, mtime)


if __name__ == "__main__":
    unittest.main()
