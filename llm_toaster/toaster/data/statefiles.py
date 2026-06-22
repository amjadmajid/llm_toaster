"""Atomic JSON state files (producer status, producer checkpoint, consumer cursor).

These tiny side files coordinate the prefetch producer and the trainer across processes. They
are written atomically (temp -> ``fsync`` -> ``os.replace``) so a reader never sees a half-written
file, and a missing/garbage file reads back as ``None`` rather than crashing the consumer.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(path: str | os.PathLike, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_state_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def read_json(path: str | os.PathLike) -> dict | None:
    """Return the parsed JSON object, or ``None`` if absent/unreadable/partway through a write."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
