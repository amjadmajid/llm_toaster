"""``ManifestShardSource`` -- the canonical trainer-facing pretraining batch source.

Reads immutable, manifest-described token shards. ``prepared`` and ``prefetch`` modes use
*exactly* this class over *exactly* the same manifest/shard contract; the only difference is
who writes the shards (an upfront ``prepare`` job vs. a background producer) and the exhaustion
policy (``stop`` vs. ``wait``).

Key behaviors:
- the manifest -- never a directory listing -- is the source of truth; ``wait`` refreshes it,
- exhaustion is explicit (``stop`` raises :class:`DataExhausted`; ``repeat`` wraps and counts
  passes; ``wait`` blocks for newly sealed shards with a timeout),
- ``.npy`` shards are memory-mapped (released when moving between shards); ``.txt``/``.tokens``
  fixtures still work,
- ``state_dict``/``load_state_dict`` resume the exact next batch and validate dataset identity.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .errors import (
    DataExhausted,
    DataWaitTimeout,
    ProducerFailedError,
    ResumeIncompatibleError,
    ShardError,
)
from .manifest import Manifest, ShardEntry, load_manifest, resolve_shard_path
from .packing import make_batch
from .protocol import PretrainBatchInfo

logger = logging.getLogger(__name__)

SOURCE_MODE = "prepared"


def transform_identity_hash(manifest: Manifest) -> str:
    from .packing import hash_canonical

    return hash_canonical(manifest.transform.identity())


def verify_resume_compatible(manifest: Manifest, state: dict, split: str) -> None:
    """Reject a resume whose checkpoint is incompatible with the current manifest.

    Verifies, with specific messages: dataset identity, resolved source revision, tokenizer
    fingerprint, transform fingerprint, current shard id + checksum, and that the committed
    manifest prefix through the current shard is unchanged. Appended shards beyond the prefix
    are explicitly allowed (a growing prefetch manifest stays compatible).
    """
    if not state:
        return
    saved_id = state.get("dataset_id")
    if saved_id is not None and saved_id != manifest.dataset_id:
        raise ResumeIncompatibleError(
            f"dataset identity changed: checkpoint has dataset_id={saved_id!r}, manifest has {manifest.dataset_id!r}."
        )
    saved_rev = state.get("resolved_revision")
    if saved_rev is not None and saved_rev != manifest.source.resolved_revision:
        raise ResumeIncompatibleError(
            f"source revision changed: checkpoint pinned {saved_rev!r}, "
            f"manifest resolved {manifest.source.resolved_revision!r}."
        )
    saved_tok = state.get("tokenizer_fingerprint")
    if saved_tok is not None and saved_tok != manifest.tokenizer.fingerprint:
        raise ResumeIncompatibleError(
            "tokenizer fingerprint changed since the checkpoint; refusing to resume on different tokens."
        )
    saved_tf = state.get("transform_fingerprint")
    if saved_tf is not None and saved_tf != transform_identity_hash(manifest):
        raise ResumeIncompatibleError(
            "transform (add_eot/packing/dtype) changed since the checkpoint; refusing to resume."
        )
    shard_id = state.get("shard_id")
    if shard_id is not None:
        entry = manifest.find_shard(split, shard_id)
        if entry is None:
            raise ResumeIncompatibleError(
                f"checkpoint's current shard {shard_id!r} is not in split {split!r} of the manifest."
            )
        saved_sha = state.get("shard_sha256")
        if saved_sha is not None and saved_sha != entry.sha256:
            raise ResumeIncompatibleError(
                f"current shard {shard_id!r} checksum changed since the checkpoint; the shard was modified."
            )
        saved_prefix = state.get("manifest_prefix_fingerprint")
        if saved_prefix is not None and saved_prefix != manifest.prefix_fingerprint(split, shard_id):
            raise ResumeIncompatibleError(
                f"the committed manifest prefix through {shard_id!r} changed since the checkpoint "
                f"(an earlier shard was altered/removed). Appended shards are fine; rewriting history is not."
            )


class ManifestShardSource:
    """Resumable batch source over a manifest split. Implements ``PretrainBatchSource``."""

    def __init__(
        self,
        manifest_path: str,
        split: str,
        batch_size: int,
        seq_len: int,
        *,
        exhaustion: str = "stop",
        shuffle: str = "none",
        seed: int = 1337,
        mmap: bool = True,
        wait_timeout_s: float = 300.0,
        poll_interval_s: float = 0.5,
        status_provider=None,
        manifest: Manifest | None = None,
    ):
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        if exhaustion not in {"stop", "repeat", "wait"}:
            raise ValueError(f"unknown exhaustion policy {exhaustion!r}")
        if shuffle not in {"none", "shard"}:
            raise ValueError(f"unknown shuffle policy {shuffle!r}")
        if shuffle == "shard" and exhaustion == "wait":
            raise ValueError("shuffle='shard' is incompatible with a growing (wait) manifest")
        self.manifest_path = str(manifest_path)
        self.split = split
        self.B = batch_size
        self.T = seq_len
        self.exhaustion = exhaustion
        self.shuffle = shuffle
        self.seed = seed
        self.mmap = mmap
        self.wait_timeout_s = wait_timeout_s
        self.poll_interval_s = poll_interval_s
        self.status_provider = status_provider

        self.manifest = manifest if manifest is not None else load_manifest(self.manifest_path)
        self.pass_index = 0
        self.repeated = False
        self._order: list[int] = []
        self._order_pos = 0
        self._tokens: np.ndarray | None = None
        self._current_entry: ShardEntry | None = None
        self.current_position = 0
        # Counters for stats() (low-cardinality only).
        self.unique_tokens_emitted = 0
        self.batches_emitted = 0
        self._build_order()
        self._waited_seconds_total = 0.0

    # ----- ordering -------------------------------------------------------
    def _entries(self) -> list[ShardEntry]:
        return self.manifest.split(self.split).shards

    def _build_order(self) -> None:
        n = len(self._entries())
        if self.shuffle == "shard":
            self._order = list(np.random.default_rng(self.seed).permutation(n))
        else:
            self._order = list(range(n))
        self._order_pos = 0

    def _extend_order_after_refresh(self, previous_n: int) -> None:
        """Append newly sealed shards (``none`` shuffle only) to the reading order."""
        n = len(self._entries())
        if n > previous_n and self.shuffle == "none":
            self._order.extend(range(previous_n, n))

    # ----- shard loading --------------------------------------------------
    def _release_tokens(self) -> None:
        tokens = self._tokens
        self._tokens = None
        mmap = getattr(tokens, "_mmap", None)
        if mmap is not None:
            try:
                mmap.close()
            except (BufferError, ValueError):  # pragma: no cover - mmap already released
                pass

    def _load_order_position(self, order_pos: int, *, verify: bool = False) -> None:
        entry = self._entries()[self._order[order_pos]]
        path = resolve_shard_path(self.manifest_path, entry)
        self._release_tokens()
        from .shard_store import read_shard

        tokens = read_shard(path, mmap=self.mmap, expected_sha256=entry.sha256 if verify else None)
        if tokens.shape[0] < self.B * self.T + 1:
            raise ShardError(
                f"shard {entry.id!r} has {tokens.shape[0]} tokens, fewer than B*T+1 = "
                f"{self.B * self.T + 1} required for one batch (B={self.B}, T={self.T})."
            )
        self._tokens = tokens
        self._current_entry = entry
        self._order_pos = order_pos
        self.current_position = 0

    def _ensure_loaded(self) -> None:
        if self._tokens is None:
            if not self._order:
                self._on_no_more_shards(first=True)
            self._load_order_position(self._order_pos)

    # ----- exhaustion handling -------------------------------------------
    def _advance_shard(self) -> None:
        """Move to the next shard in order, applying the exhaustion policy at the end."""
        if self._order_pos + 1 < len(self._order):
            self._load_order_position(self._order_pos + 1)
            return
        self._on_no_more_shards(first=False)

    def _on_no_more_shards(self, *, first: bool) -> None:
        if self.exhaustion == "wait":
            self._wait_for_more_shards(first=first)
            return
        if self.exhaustion == "repeat":
            self._begin_new_pass()
            return
        # stop
        raise DataExhausted(
            f"unique pretraining data exhausted after {self.pass_index} pass(es) over split {self.split!r}",
            pass_index=self.pass_index,
        )

    def _begin_new_pass(self) -> None:
        self.pass_index += 1
        self.repeated = True
        self._build_order()  # deterministic re-permutation (or range) for the new pass
        if not self._order:
            raise DataExhausted("no shards available to repeat", pass_index=self.pass_index)
        logger.info("data pass %d begins (repeat) over split %r", self.pass_index, self.split)
        self._load_order_position(0)

    def _refresh_manifest(self) -> bool:
        """Reload the manifest from disk (the source of truth, never a directory scan)."""
        previous_n = len(self._entries())
        try:
            self.manifest = load_manifest(self.manifest_path)
        except Exception as exc:  # pragma: no cover - defensive; surfaced as a clear error
            logger.warning("manifest refresh failed: %s", exc)
            return False
        self._extend_order_after_refresh(previous_n)
        return len(self._entries()) > previous_n

    def _wait_for_more_shards(self, *, first: bool) -> None:
        deadline = time.monotonic() + self.wait_timeout_s
        target_pos = self._order_pos if first else self._order_pos + 1
        while True:
            if self._refresh_manifest() and target_pos < len(self._order):
                self._load_order_position(target_pos)
                return
            status = self._producer_status()
            if status == "failed":
                raise ProducerFailedError(self._producer_error() or "the prefetch producer reported failure")
            if status == "complete" and target_pos >= len(self._order):
                raise DataExhausted(
                    f"producer complete and split {self.split!r} fully consumed", pass_index=self.pass_index
                )
            if time.monotonic() >= deadline:
                raise DataWaitTimeout(
                    f"waited {self.wait_timeout_s:.0f}s for a new shard in split {self.split!r} but none arrived "
                    f"(producer status: {status})."
                )
            wait_start = time.monotonic()
            time.sleep(self.poll_interval_s)
            self._waited_seconds_total += time.monotonic() - wait_start

    def _producer_status(self) -> str | None:
        if self.status_provider is None:
            return None
        try:
            return self.status_provider().get("status")
        except Exception:  # pragma: no cover - status is advisory
            return None

    def _producer_error(self) -> str | None:
        if self.status_provider is None:
            return None
        try:
            return self.status_provider().get("error")
        except Exception:  # pragma: no cover
            return None

    # ----- protocol -------------------------------------------------------
    def next_batch(self) -> tuple[np.ndarray, np.ndarray, PretrainBatchInfo]:
        self._ensure_loaded()
        assert self._tokens is not None and self._current_entry is not None
        need = self.B * self.T + 1
        if self.current_position + need > self._tokens.shape[0]:
            self._advance_shard()  # drop the shard tail; batches never straddle shard boundaries
        x, y = make_batch(self._tokens, self.current_position, self.B, self.T)
        self.current_position += self.B * self.T
        self.unique_tokens_emitted += self.B * self.T
        self.batches_emitted += 1
        info = PretrainBatchInfo(
            shard_id=self._current_entry.id,
            token_offset=self.current_position,
            pass_index=self.pass_index,
            repeated=self.repeated,
            source_mode=SOURCE_MODE,
        )
        # Copy the small window out of the (read-only) memory map so torch gets a writable array.
        return np.array(x), np.array(y), info

    def reset(self) -> None:
        self._release_tokens()
        self.pass_index = 0
        self.repeated = False
        self.current_position = 0
        self._current_entry = None
        self._build_order()

    def close(self) -> None:
        self._release_tokens()

    def state_dict(self) -> dict:
        entry = self._current_entry
        return {
            "source_mode": SOURCE_MODE,
            "split": self.split,
            "dataset_id": self.manifest.dataset_id,
            "dataset_fingerprint": self.manifest.dataset_fingerprint,
            "resolved_revision": self.manifest.source.resolved_revision,
            "tokenizer_fingerprint": self.manifest.tokenizer.fingerprint,
            "transform_fingerprint": transform_identity_hash(self.manifest),
            "manifest_generation": self.manifest.generation,
            "shard_id": entry.id if entry else None,
            "shard_sha256": entry.sha256 if entry else None,
            "manifest_prefix_fingerprint": (self.manifest.prefix_fingerprint(self.split, entry.id) if entry else None),
            "token_offset": self.current_position,
            "pass_index": self.pass_index,
            "repeated": self.repeated,
            "unique_tokens_seen": self.unique_tokens_emitted,
        }

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        verify_resume_compatible(self.manifest, state, self.split)
        self.pass_index = int(state.get("pass_index", 0))
        self.repeated = bool(state.get("repeated", self.pass_index > 0))
        shard_id = state.get("shard_id")
        if shard_id is None:
            return
        # Locate the saved shard within the (possibly shuffled) order and seek to its offset.
        target_index = next((i for i, e in enumerate(self._entries()) if e.id == shard_id), None)
        if target_index is None:  # pragma: no cover - verify_resume_compatible already guards this
            raise ResumeIncompatibleError(f"shard {shard_id!r} not found on resume")
        order_pos = self._order.index(target_index)
        self._load_order_position(order_pos, verify=True)
        self.current_position = int(state.get("token_offset", 0))
        self.unique_tokens_emitted = int(state.get("unique_tokens_seen", 0))

    def consumed_shard_count(self) -> int:
        """How many train shards have been fully passed (used to bound the prefetch queue)."""
        return self._order_pos

    def stats(self) -> dict:
        return {
            "source_mode": SOURCE_MODE,
            "dataset_id": self.manifest.dataset_id,
            "dataset_fingerprint": self.manifest.dataset_fingerprint,
            "manifest_generation": self.manifest.generation,
            "current_shard_id": self._current_entry.id if self._current_entry else None,
            "data_pass": self.pass_index,
            "data_wait_s": round(self._waited_seconds_total, 3),
        }
