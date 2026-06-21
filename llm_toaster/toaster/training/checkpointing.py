"""Checkpoint save/load helpers for full training state.

Checkpoint payload schema (``format_version`` 1):
    model, optimizer, scheduler, scaler, config, global_step, tokens_seen,
    rng_state, data_state, best_metric, git_commit, wall_clock_s,
    tokenizer_info, format_version.

Checkpoints are written atomically (temp file -> fsync -> rename) so an interruption
mid-write can never corrupt the live checkpoint.
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 1


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
    wall_clock_s: float = 0.0,
    tokenizer_info: dict | None = None,
) -> None:
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
        "wall_clock_s": wall_clock_s,
        "tokenizer_info": tokenizer_info or {},
        "git_commit": git_commit(),
        "format_version": CHECKPOINT_FORMAT_VERSION,
    }
    atomic_save(payload, path)


def atomic_save(payload, path: str) -> None:
    """Write ``payload`` to ``path`` atomically: temp file in the same dir, fsync, then rename.

    ``os.replace`` is atomic within a filesystem, so a reader/resume always sees either the old
    checkpoint or the fully written new one -- never a half-written file from an interrupted save.
    """
    path = str(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_ckpt_", suffix=".pt")
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Remove the partial temp file on any failure/interruption (incl. KeyboardInterrupt).
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def load_checkpoint(
    path: str, model, optimizer=None, scheduler=None, scaler=None, device: str = "cpu", strict: bool = True
) -> dict:
    # weights_only=False is required because checkpoints carry optimizer/scheduler
    # state, RNG state, and the config dict. Only load checkpoints you trust.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    version = checkpoint.get("format_version", 1)
    if version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint {path} has format_version {version}, but this build only understands "
            f"up to {CHECKPOINT_FORMAT_VERSION}. Upgrade LLM Toaster to load it."
        )
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    _restore_rng_state(checkpoint.get("rng_state", {}))
    return checkpoint


def load_state_dict_any(path: str, device: str = "cpu") -> dict:
    """Return a model state_dict from any supported artifact.

    Accepts a full engine checkpoint (``{"model": ...}``), a legacy checkpoint
    (``{"model_state_dict": ...}``), or a bare ``state_dict`` (e.g. a ``.llm``
    inference file). Tries the safe ``weights_only`` path first.
    """
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        return obj["model"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return {key.replace("_orig_mod.", ""): value for key, value in obj["model_state_dict"].items()}
    return obj


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
    # RNG states must be CPU ByteTensors. torch.load(map_location="cuda") moves every
    # tensor in the checkpoint (including these) to the GPU, so move them back explicitly.
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu())
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    cuda_states = state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        cuda_states = [s.cpu() for s in cuda_states]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
