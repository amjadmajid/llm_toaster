"""Collate sweep runs into a comparison table (Markdown + CSV) and optional Pareto plots.

Reads each run's ``metrics.jsonl`` (the self-describing ``architecture`` row + the final ``step``
row) plus ``run.json``, producing one tidy record per run. Pure stdlib; plotting is optional and
only imported when ``--plot`` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

_ARCH_FIELDS = [
    "params_total",
    "params_non_embedding",
    "flops_per_token",
    "kv_bytes_per_token",
    "attention_kind",
    "positions",
    "norm",
    "ffn",
    "num_key_value_heads",
]
_STEP_FIELDS = ["step", "loss", "tokens_per_sec", "peak_mem_bytes", "mfu"]


def _read_run(run_dir: Path) -> dict | None:
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return None
    arch, last_step = None, None
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "architecture":
            arch = row
        elif row.get("type") == "step":
            last_step = row
    if arch is None or last_step is None:
        return None

    meta = {}
    run_json = run_dir / "run.json"
    if run_json.exists():
        meta = json.loads(run_json.read_text(encoding="utf-8"))

    record = {"run": meta.get("run_name", run_dir.name), "axis": meta.get("axis", ""), "value": meta.get("value", "")}
    record["seed"] = meta.get("seed", "")
    for field in _ARCH_FIELDS:
        record[field] = arch.get(field)
    for field in _STEP_FIELDS:
        record[field] = last_step.get(field)
    loss = last_step.get("loss")
    record["perplexity"] = math.exp(loss) if loss is not None and loss < 100 else float("inf")
    return record


def aggregate_runs(output_dir) -> list[dict]:
    """One record per run found under ``output_dir`` (recursively)."""
    rows = []
    for metrics in sorted(Path(output_dir).rglob("metrics.jsonl")):
        record = _read_run(metrics.parent)
        if record is not None:
            rows.append(record)
    return rows


_TABLE_COLUMNS = [
    "run",
    "ffn",
    "norm",
    "positions",
    "attention_kind",
    "num_key_value_heads",
    "params_total",
    "flops_per_token",
    "kv_bytes_per_token",
    "loss",
    "perplexity",
    "tokens_per_sec",
    "peak_mem_bytes",
    "mfu",
]


def _fmt(column: str, value) -> str:
    if value is None:
        return "-"
    if column in {"params_total", "flops_per_token"}:
        return f"{value / 1e6:.1f}M"
    if column in {"kv_bytes_per_token", "peak_mem_bytes"}:
        return f"{value / 1024:.0f}KB" if value < 1024**2 else f"{value / 1024**2:.0f}MB"
    if column == "mfu":
        return f"{value * 100:.1f}%"
    if column in {"loss", "perplexity", "tokens_per_sec"}:
        return f"{value:.2f}"
    return str(value)


def to_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(_TABLE_COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in _TABLE_COLUMNS) + " |"
    body = ["| " + " | ".join(_fmt(col, row.get(col)) for col in _TABLE_COLUMNS) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def to_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_pareto(rows: list[dict], output_dir: str) -> list[str]:
    """Scatter final loss vs each efficiency axis. Returns saved PNG paths (empty if no matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping plots (pip install matplotlib).")
        return []

    saved = []
    for axis in ("params_total", "flops_per_token", "kv_bytes_per_token"):
        points = [(r[axis], r["loss"], r["run"]) for r in rows if r.get(axis) is not None and r.get("loss") is not None]
        if not points:
            continue
        figure, ax = plt.subplots()
        for x, y, name in points:
            ax.scatter(x, y)
            ax.annotate(name, (x, y), fontsize=7)
        ax.set_xlabel(axis)
        ax.set_ylabel("final loss")
        ax.set_title(f"loss vs {axis}")
        path = str(Path(output_dir) / f"pareto_{axis}.png")
        figure.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(figure)
        saved.append(path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LLM Toaster sweep runs into a table.")
    parser.add_argument("--dir", required=True, help="Sweep output directory.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--plot", action="store_true", help="Save Pareto PNGs (needs matplotlib).")
    args = parser.parse_args()

    rows = aggregate_runs(args.dir)
    if not rows:
        print(f"No completed runs found under {args.dir}")
        return
    print(to_markdown(rows))
    if args.csv:
        to_csv(rows, args.csv)
        print(f"\nWrote CSV: {args.csv}")
    if args.plot:
        for path in plot_pareto(rows, args.dir):
            print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
