"""Sweep runner + aggregator, end-to-end on the tiny smoke config."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm_toaster.toaster.experiments import sweep
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


class SweepFaultIsolationTests(unittest.TestCase):
    def test_failed_run_does_not_abort_sweep(self):
        """A cell that raises is recorded and skipped; later cells still run."""

        class _FlakyEngine:
            def __init__(self, config):
                self.config = config

            def train(self):
                if self.config.model.ffn == "gelu":
                    raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as td:
            spec = _spec(td)  # axes -> ffn: [gelu, swiglu]
            with mock.patch.object(sweep, "TrainingEngine", _FlakyEngine):
                completed = run_sweep(spec)  # cpu -> in-process

            gelu_dir = Path(td) / "ffn=gelu__seed0"
            swiglu_dir = Path(td) / "ffn=swiglu__seed0"
            # The failure was isolated: gelu marked failed (no done), swiglu still ran.
            self.assertTrue((gelu_dir / "failed").exists())
            self.assertFalse((gelu_dir / "done").exists())
            self.assertIn("boom", (gelu_dir / "failed").read_text())
            self.assertTrue((swiglu_dir / "done").exists())
            self.assertEqual(completed, [swiglu_dir])

    def test_isolated_subprocess_path(self):
        """Forcing isolation actually trains the cell in a spawned subprocess."""
        with tempfile.TemporaryDirectory() as td:
            spec = _spec(td)
            spec["axes"] = {"model.ffn": ["gelu"]}  # single cell keeps the spawn cheap
            run_dirs = run_sweep(spec, isolate=True)
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "metrics.jsonl").exists())
            self.assertTrue((run_dirs[0] / "done").exists())


if __name__ == "__main__":
    unittest.main()
