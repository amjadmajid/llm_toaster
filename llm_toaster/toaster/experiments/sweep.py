"""Architecture sweep runner.

A sweep spec (YAML/dict) describes a one-variable-at-a-time study from a base config:

    base_config: config/default_config.yaml
    output_dir: runs/ffn_ablation
    seeds: [0, 1, 2]
    target_params: 60000000      # optional: solve `vary` so every run matches this param count
    vary: n_embd                 # n_embd | n_blocks (which dim the solver adjusts)
    overrides:                   # applied to every run
      training.max_iter: 2000
    axes:                        # for each axis, each value is run with all else at baseline
      model.ffn: [gelu, swiglu, geglu]

Each (axis, value, seed) becomes one run under ``output_dir/<axis>=<value>__seed<seed>/`` with its
own ``metrics.jsonl``. Completed runs drop a ``done`` marker and are skipped on re-runs (resumable).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..config import ConfigHandler
from ..models.sizing import solve_for_target_params
from ..training.engine import TrainingEngine

logger = logging.getLogger(__name__)


def load_spec(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _set_dotted(config, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    obj = config
    for part in parts[:-1]:
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        raise ValueError(f"Unknown config field in axis/override: {dotted_key!r}")
    setattr(obj, parts[-1], value)


def _sanitize(value) -> str:
    return str(value).replace("/", "_").replace(" ", "")


def build_runs(spec: dict) -> list[dict]:
    """Materialise (and validate) one config per (axis, value, seed) cell."""
    output_dir = Path(spec["output_dir"])
    overrides = spec.get("overrides", {})
    seeds = spec.get("seeds", [0])
    target = spec.get("target_params")
    vary = spec.get("vary", "n_embd")

    runs = []
    for axis_key, values in spec.get("axes", {}).items():
        short = axis_key.split(".")[-1]
        for value in values:
            for seed in seeds:
                config = ConfigHandler.from_yaml(spec["base_config"])
                for key, override_value in overrides.items():
                    _set_dotted(config, key, override_value)
                _set_dotted(config, axis_key, value)
                config.training.seed = seed
                if target:
                    config = solve_for_target_params(config, int(target), vary=vary)

                run_name = f"{short}={_sanitize(value)}__seed{seed}"
                run_dir = output_dir / run_name
                config.logging.metrics_file = str(run_dir / "metrics.jsonl")
                config.logging.log_file = str(run_dir / "train.log")
                config.checkpointing.output_dir = str(run_dir)
                config.training.ckpt = str(run_dir / "ckpt")
                config.training.ckpt_config = str(run_dir / "config.yaml")
                try:
                    config.validate()
                except (ValueError, NotImplementedError) as exc:
                    raise type(exc)(f"run {run_name}: {exc}") from exc
                runs.append(
                    {
                        "run_name": run_name,
                        "run_dir": run_dir,
                        "axis": axis_key,
                        "value": value,
                        "seed": seed,
                        "config": config,
                    }
                )
    return runs


def run_sweep(spec: dict, force: bool = False) -> list[Path]:
    """Train every run in the sweep, skipping completed ones unless ``force``."""
    runs = build_runs(spec)
    completed = []
    for index, run in enumerate(runs, start=1):
        run_dir: Path = run["run_dir"]
        done_marker = run_dir / "done"
        if done_marker.exists() and not force:
            logger.info("skip %s (already done)", run["run_name"])
            completed.append(run_dir)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(
            json.dumps({key: run[key] for key in ("run_name", "axis", "value", "seed")}), encoding="utf-8"
        )
        logger.info("[%d/%d] training %s", index, len(runs), run["run_name"])
        TrainingEngine(run["config"]).train()
        done_marker.write_text("ok", encoding="utf-8")
        completed.append(run_dir)
    return completed


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run an LLM Toaster architecture sweep.")
    parser.add_argument("--spec", required=True, help="Path to a sweep spec YAML.")
    parser.add_argument("--force", action="store_true", help="Re-run runs that already completed.")
    args = parser.parse_args()
    run_sweep(load_spec(args.spec), force=args.force)


if __name__ == "__main__":
    main()
