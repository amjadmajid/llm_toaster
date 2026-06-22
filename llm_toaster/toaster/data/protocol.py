"""The canonical pretraining batch-source interface.

Every pretraining data path -- materialized shards (:class:`ManifestShardSource`), a legacy
shard directory (:class:`LegacyShardDirSource`), or in-process HF streaming
(:class:`HFDirectTokenSource`) -- implements :class:`PretrainBatchSource`. The
:class:`TrainingEngine` only ever talks to this protocol, so a future object-store or
MosaicML-Streaming adapter can be added without touching the engine.

Each batch carries a :class:`PretrainBatchInfo` describing *where* it came from, which the
engine uses for accounting (unique vs repeated tokens), logging, and resume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class PretrainBatchInfo:
    """Provenance for a single pretraining batch.

    Attributes:
        shard_id: Stable id of the shard the batch came from (``None`` for direct streaming).
        token_offset: Token cursor *after* this batch within the current shard/stream.
        pass_index: How many full passes over the unique data have completed (0 on the first).
        repeated: ``True`` once the source has wrapped at least once (``repeat`` mode only).
        source_mode: ``prepared`` | ``prefetch`` | ``direct`` | ``legacy``.
    """

    shard_id: str | None
    token_offset: int
    pass_index: int
    repeated: bool
    source_mode: str


@runtime_checkable
class PretrainBatchSource(Protocol):
    """A resumable source of ``(inputs, targets, info)`` pretraining batches.

    Implementations must apply the single next-token label shift exactly once
    (``x = tokens[:-1]``, ``y = tokens[1:]``); the engine applies no further shift.
    """

    def next_batch(self) -> tuple[np.ndarray, np.ndarray, PretrainBatchInfo]:
        """Return the next ``(x, y, info)``.

        Raises:
            DataExhausted: under ``stop`` when no more unique data is available.
            DataWaitTimeout / ProducerFailedError: under ``wait`` when nothing new arrives.
        """
        ...

    def state_dict(self) -> dict:
        """Serializable cursor + identity for exact resume."""
        ...

    def load_state_dict(self, state: dict) -> None:
        """Restore the cursor saved by :meth:`state_dict` (validates identity where applicable)."""
        ...

    def reset(self) -> None:
        """Return to the very beginning (shard 0 / start of stream), pass 0."""
        ...

    def close(self) -> None:
        """Release file handles, memory maps, and any background resources."""
        ...

    def stats(self) -> dict:
        """Lightweight, low-cardinality counters for metrics/observability."""
        ...
