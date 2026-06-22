"""Document stream openers, dispatched by ``source.type``.

``huggingface`` streams records from the Hub (lazy ``datasets`` import); ``local`` reads a
``.jsonl``/``.txt`` file on disk. The local source keeps ``prepare``/``prefetch`` fully usable
offline (and is what the test-suite and air-gapped users rely on). Both yield ``{text_field: str}``
records so the producer/packer are source-agnostic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from .shard_store import sha256_file


def open_local_stream(source) -> Iterator[dict]:
    """Yield ``{text_field: ...}`` records from a local ``.jsonl`` (one row each) or ``.txt``
    (one non-empty line each) file at ``source.dataset_name``."""
    path = Path(source.dataset_name)
    if not path.exists():
        raise FileNotFoundError(f"local source file not found: {path}")
    field = source.text_field
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield {field: row.get(field, row.get("text", ""))}
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield {field: line.rstrip("\n")}


def open_document_stream(source) -> Iterator[dict]:
    """Dispatch on ``source.type`` to an iterator of ``{text_field: str}`` records."""
    if source.type in {"huggingface", "hf"}:
        from .hf_source import open_hf_stream

        return iter(open_hf_stream(source))
    if source.type == "local":
        return open_local_stream(source)
    raise ValueError(f"unknown source.type {source.type!r} (expected 'huggingface' or 'local')")


def resolve_source_revision(source) -> str:
    """Return an immutable revision id for the source.

    Hugging Face -> the immutable commit SHA (network). Local -> a content hash of the file, so a
    changed local corpus is detected on resume.
    """
    if source.type in {"huggingface", "hf"}:
        from .hf_source import resolve_revision

        return resolve_revision(source.dataset_name, source.requested_revision or getattr(source, "revision", None))
    if source.type == "local":
        path = Path(source.dataset_name)
        if not path.exists():
            raise FileNotFoundError(f"local source file not found: {path}")
        digest, _ = sha256_file(path)
        return f"local:{digest[:16]}"
    raise ValueError(f"unknown source.type {source.type!r}")


def _env_start_method() -> str:
    """Multiprocessing start method for the producer; ``spawn`` by default (CUDA-safe)."""
    return os.environ.get("LLM_TOASTER_MP_START", "spawn")
