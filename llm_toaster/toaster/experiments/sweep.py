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

Each GPU run trains in its own spawned subprocess (fresh CUDA context), so a hard device fault —
e.g. ``cudaErrorLaunchFailure``, which corrupts the process's CUDA context — fails just that run
instead of aborting the whole sweep. A failed run leaves a ``failed`` marker (and no ``done``
marker, so it retries on the next invocation); the sweep moves on to the next cell.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
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


def _train_one(config) -> None:
    """Run a single sweep cell to completion.

    Top-level (so it is picklable) so it can serve as the target of a spawned
    subprocess as well as be called in-process. Run on its own process it gets a
    fresh CUDA context, which is what makes one cell's hard GPU fault survivable.
    """
    logging.basicConfig(level=logging.INFO)
    TrainingEngine(config).train()


def _run_isolated(config) -> str | None:
    """Execute :func:`_train_one` in a spawned subprocess.

    Returns ``None`` on success or a human-readable reason on failure. A spawned
    (not forked) child starts with a clean CUDA context, so a corrupted context or
    a segfault from one run dies with the child instead of poisoning the parent and
    every cell that follows it.
    """
    process = mp.get_context("spawn").Process(target=_train_one, args=(config,))
    process.start()
    try:
        process.join()
    except KeyboardInterrupt:
        process.terminate()
        process.join()
        raise
    code = process.exitcode
    if code == 0:
        return None
    if code is not None and code < 0:
        return f"worker killed by signal {-code}"
    return f"worker exited with code {code}"


def run_sweep(spec: dict, force: bool = False, isolate: bool | None = None) -> list[Path]:
    """Train every run in the sweep, skipping completed ones unless ``force``.

    A cell's failure is isolated: it is logged, recorded as a ``failed`` marker, and the
    sweep continues to the next cell. ``isolate`` controls subprocess execution — ``None``
    (default, overridable via ``spec["isolate"]``) isolates only non-CPU runs; ``True``/
    ``False`` forces isolation on/off for every run. Returns the successfully completed
    (and already-done) run directories.
    """
    runs = build_runs(spec)
    if isolate is None:
        isolate = spec.get("isolate")
    completed: list[Path] = []
    failed: list[Path] = []
    for index, run in enumerate(runs, start=1):
        run_dir: Path = run["run_dir"]
        done_marker = run_dir / "done"
        if done_marker.exists() and not force:
            logger.info("skip %s (already done)", run["run_name"])
            completed.append(run_dir)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "failed").unlink(missing_ok=True)
        (run_dir / "run.json").write_text(
            json.dumps({key: run[key] for key in ("run_name", "axis", "value", "seed")}), encoding="utf-8"
        )
        logger.info("[%d/%d] training %s", index, len(runs), run["run_name"])

        device = str(run["config"].training.device).lower()
        use_subprocess = isolate if isolate is not None else device != "cpu"
        try:
            if use_subprocess:
                reason = _run_isolated(run["config"])
                if reason is not None:
                    raise RuntimeError(reason)
            else:
                _train_one(run["config"])
        except Exception as exc:  # noqa: BLE001 - one cell's failure must not abort the sweep
            logger.error("run %s failed: %s", run["run_name"], exc)
            (run_dir / "failed").write_text(str(exc) or repr(exc), encoding="utf-8")
            failed.append(run_dir)
            continue

        done_marker.write_text("ok", encoding="utf-8")
        completed.append(run_dir)

    if failed:
        logger.warning(
            "sweep finished: %d ok, %d failed (%s)",
            len(completed),
            len(failed),
            ", ".join(p.name for p in failed),
        )
    return completed


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run an LLM Toaster architecture sweep.")
    parser.add_argument("--spec", required=True, help="Path to a sweep spec YAML.")
    parser.add_argument("--force", action="store_true", help="Re-run runs that already completed.")
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="Run every cell in-process (no subprocess isolation); useful for debugging with CUDA_LAUNCH_BLOCKING=1.",
    )
    args = parser.parse_args()
    run_sweep(load_spec(args.spec), force=args.force, isolate=False if args.no_isolate else None)


if __name__ == "__main__":
    main()
