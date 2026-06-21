"""Training metrics: formatting helpers, an architecture summary, and a JSONL writer.

Stdlib-only so it runs anywhere (offline/embedded) and keeps runs comparable: the
per-step console line is human-readable while ``logs/metrics.jsonl`` is machine-readable.
"""

from __future__ import annotations

import json
from pathlib import Path


def format_duration(seconds: float) -> str:
    """Compact duration, e.g. '45s', '11m51s', '2h09m'."""
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def human_count(n: float) -> str:
    """Human-readable count, e.g. 11403264 -> '11.4M'."""
    for unit, divisor in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= divisor:
            return f"{n / divisor:.1f}{unit}"
    return str(int(n))


def human_bytes(n: float) -> str:
    """Human-readable byte size, e.g. 536870912 -> '512.0 MB'."""
    for unit, divisor in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if abs(n) >= divisor:
            return f"{n / divisor:.1f} {unit}"
    return f"{int(n)} B"


def format_metrics_line(record: dict) -> str:
    """Aligned per-step console line. MFU and peak memory are appended when available."""
    line = (
        f"step {record['step']:>7,}/{record['max_iter']:,} | "
        f"loss {record['loss']:.4f} | "
        f"lr {record['lr']:.2e} | "
        f"gnorm {record['grad_norm']:.2f} | "
        f"{record['tokens_per_sec']:>8,.0f} tok/s | "
        f"seen {human_count(record['tokens_seen'])} | "
        f"{format_duration(record['elapsed_s'])} | "
        f"eta {format_duration(record['eta_s'])}"
    )
    if record.get("mfu") is not None:
        line += f" | mfu {record['mfu'] * 100:.1f}%"
    if record.get("peak_mem_bytes"):
        line += f" | mem {human_bytes(record['peak_mem_bytes'])}"
    return line


def architecture_summary(model, config) -> dict:
    """Key architecture properties for reporting and cross-run comparison."""
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(p.numel() for name, p in model.named_parameters() if "embedding" in name.lower())
    n_head = config.model.n_head
    kv_heads = config.model.num_key_value_heads or n_head
    head_dim = config.model.n_embd // n_head
    # KV cache (autoregressive inference), fp16: 2 tensors (k, v) per layer.
    kv_bytes_per_token = 2 * config.model.n_blocks * kv_heads * head_dim * 2
    attention_kind = "MHA" if kv_heads == n_head else ("MQA" if kv_heads == 1 else "GQA")
    # Training FLOPs/token (fwd+bwd), standard 6N + attention term (Kaplan/nanoGPT convention).
    # N excludes learned position embeddings (no matmul) but keeps the tied token embedding (lm_head matmul).
    pos_emb = config.model.seq_len * config.model.n_embd if config.model.position == "learned" else 0
    flops_per_token = 6 * (total - pos_emb) + 12 * config.model.n_blocks * n_head * head_dim * config.model.seq_len
    return {
        "architecture": config.model.architecture,
        "dense": config.model.ffn != "moe",
        "positions": config.model.position,
        "norm": config.model.norm,
        "ffn": config.model.ffn,
        "params_total": total,
        "params_embedding": embedding,
        "params_non_embedding": total - embedding,
        "n_head": n_head,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "attention_kind": attention_kind,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_bytes_at_seq_len": kv_bytes_per_token * config.model.seq_len,
        "seq_len": config.model.seq_len,
        "flops_per_token": flops_per_token,
    }


def compute_mfu(flops_per_token: float, tokens_per_sec: float, device_peak_flops: float | None) -> float | None:
    """Model FLOPs Utilization = achieved FLOP/s / device peak FLOP/s. Needs the device peak."""
    if not device_peak_flops:
        return None
    return (flops_per_token * tokens_per_sec) / device_peak_flops


def format_architecture_summary(summary: dict) -> list[str]:
    """Render the architecture summary as log lines."""
    density = "dense" if summary["dense"] else "sparse"
    return [
        f"architecture: {summary['architecture']} ({density}) | positions={summary['positions']} | "
        f"norm={summary['norm']} | ffn={summary['ffn']}",
        f"params: {human_count(summary['params_total'])} total | "
        f"{human_count(summary['params_embedding'])} embedding | "
        f"{human_count(summary['params_non_embedding'])} non-embedding | "
        f"~{human_bytes(summary['params_total'] * 4)} fp32",
        f"attention: n_head={summary['n_head']} kv_heads={summary['num_key_value_heads']} "
        f"head_dim={summary['head_dim']} ({summary['attention_kind']}) | "
        f"KV-cache {human_bytes(summary['kv_bytes_per_token'])}/token, "
        f"{human_bytes(summary['kv_bytes_at_seq_len'])} @ seq_len={summary['seq_len']} (fp16)",
        f"compute: ~{summary['flops_per_token'] / 1e9:.2f} GFLOP/token (train fwd+bwd)",
    ]


class JsonlMetricsWriter:
    """Append-only JSONL writer for metric records; a falsy path disables it."""

    def __init__(self, path: str | None):
        self.path = path
        self._handle = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        if self._handle is not None:
            self._handle.write(json.dumps(record) + "\n")
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
