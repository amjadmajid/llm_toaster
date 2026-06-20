"""Checkpoint save/load helpers for full training state."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    config=None,
    global_step: int = 0,
    tokens_seen: int = 0,
    best_metric: float | None = None,
    data_state: dict | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "rng_state": _rng_state(),
        "data_state": data_state or {},
        "best_metric": best_metric,
        "git_commit": git_commit(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str, model, optimizer=None, scheduler=None, scaler=None, device: str = "cpu", strict: bool = True
) -> dict:
    # weights_only=False is required because checkpoints carry optimizer/scheduler
    # state, RNG state, and the config dict. Only load checkpoints you trust.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    _restore_rng_state(checkpoint.get("rng_state", {}))
    return checkpoint


def rotate_checkpoints(output_dir: str, pattern: str = "step_*.pt", save_total_limit: int = 3) -> None:
    checkpoints = sorted(Path(output_dir).glob(pattern), key=lambda p: p.stat().st_mtime)
    for checkpoint in checkpoints[:-save_total_limit]:
        checkpoint.unlink(missing_ok=True)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _rng_state() -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    if not state:
        return
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu())
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
