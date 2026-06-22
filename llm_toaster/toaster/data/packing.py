"""Deterministic tokenization, packing, and batch shaping shared by every data path.

One implementation of "turn documents into a contiguous token stream" and "cut a stream into
``(x, y)`` batches" is reused by shard materialization (prepared/prefetch) and direct streaming,
so a shard built offline and a batch streamed in-process are byte-for-byte consistent.

The next-token label shift happens here, exactly once: ``x = buf[:-1]``, ``y = buf[1:]``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

# Canary text whose token ids are folded into the tokenizer fingerprint. Encoding it captures
# *actual* tokenizer behavior, so a real GPT-2 BPE and the offline byte-fallback (or a changed
# vocab) produce different fingerprints and a mismatched resume is rejected.
_FINGERPRINT_PROBE = "The quick brown fox jumps over the lazy dog.\n0123456789"

# NumPy dtypes we are willing to store tokens as, with their exclusive vocab ceiling.
SUPPORTED_DTYPES: dict[str, int] = {
    "uint16": 1 << 16,
    "uint32": 1 << 32,
    "int32": 1 << 31,
}


def canonical_json(obj: object) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, stable separators.

    The single source of truth for every hash in the pipeline, so a manifest written on one
    machine fingerprints identically on another.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_canonical(obj: object) -> str:
    """``sha256:<hex>`` of the canonical JSON encoding of ``obj``."""
    return "sha256:" + sha256_hexdigest(canonical_json(obj).encode("utf-8"))


def dtype_for(name: str) -> np.dtype:
    if name not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported shard dtype {name!r}; supported: {sorted(SUPPORTED_DTYPES)}")
    return np.dtype(name)


def check_dtype_fits_vocab(dtype_name: str, vocab_size: int | None) -> None:
    """Reject e.g. ``dtype=uint16`` with a vocab that would overflow it."""
    ceiling = SUPPORTED_DTYPES.get(dtype_name)
    if ceiling is None:
        raise ValueError(f"Unsupported shard dtype {dtype_name!r}; supported: {sorted(SUPPORTED_DTYPES)}")
    if vocab_size is not None and vocab_size > ceiling:
        raise ValueError(
            f"dtype={dtype_name!r} holds ids in [0, {ceiling}) but vocab_size={vocab_size} "
            f"would overflow it. Use a wider dtype (e.g. uint32)."
        )


def tokenizer_fingerprint(tokenizer) -> str:
    """Stable ``sha256:`` fingerprint of a tokenizer's identity *and* behavior.

    Folds in the type/name/special-ids and the encoding of a fixed probe string so two
    tokenizers that encode differently (real BPE vs byte-fallback, or a changed vocab) never
    share a fingerprint.
    """
    try:
        probe = list(tokenizer.encode(_FINGERPRINT_PROBE))
    except Exception:  # pragma: no cover - extremely defensive; encode should not fail
        probe = []
    payload = {
        "class": type(tokenizer).__name__,
        "name": getattr(tokenizer, "name", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "probe": probe,
    }
    return hash_canonical(payload)


@dataclass(frozen=True)
class TokenPacker:
    """Encodes documents into a contiguous token stream with the configured conventions.

    EOT is *prepended* to each document (the established LLM Toaster convention, matching the
    offline ``tokenizer_lib`` pipeline), so it acts as a delimiter between packed documents.
    """

    tokenizer: object
    add_eot: bool = True
    dtype_name: str = "uint16"
    packing: str = "contiguous"

    def __post_init__(self) -> None:
        if self.packing != "contiguous":
            raise ValueError(f"Only packing='contiguous' is implemented, got {self.packing!r}")
        check_dtype_fits_vocab(self.dtype_name, getattr(self.tokenizer, "vocab_size", None))

    @property
    def dtype(self) -> np.dtype:
        return dtype_for(self.dtype_name)

    @property
    def eot_id(self) -> int:
        eot = getattr(self.tokenizer, "eos_token_id", None)
        return 0 if eot is None else int(eot)

    def encode_document(self, text: str) -> np.ndarray:
        """Tokenize one document, prepending the EOT delimiter when configured."""
        ids = [self.eot_id] if self.add_eot else []
        ids.extend(int(t) for t in self.tokenizer.encode(text))
        return np.asarray(ids, dtype=self.dtype)


def make_batch(tokens: np.ndarray, position: int, batch_size: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut one ``(x, y)`` batch out of a token stream with a single next-token shift.

    Reads ``B*T + 1`` tokens starting at ``position`` and returns ``x = buf[:-1]`` reshaped to
    ``(B, T)`` and ``y = buf[1:]`` reshaped to ``(B, T)`` -- so target ``[i]`` is input ``[i+1]``.
    Raises ``ValueError`` if the stream does not hold a full window from ``position``.
    """
    need = batch_size * seq_len + 1
    buf = tokens[position : position + need]
    if len(buf) < need:
        raise ValueError(
            f"token window starting at {position} has only {len(buf)} tokens, "
            f"fewer than B*T+1 = {need} (B={batch_size}, T={seq_len})"
        )
    buf = np.asarray(buf)
    x = buf[:-1].reshape(batch_size, seq_len)
    y = buf[1:].reshape(batch_size, seq_len)
    return x, y
