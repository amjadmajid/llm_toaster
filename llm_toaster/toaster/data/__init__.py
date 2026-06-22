"""Manifest-backed pretraining data pipeline.

Public surface (heavy/optional bits like the producer and HF helpers are imported lazily by
their callers, so importing this package never pulls in ``datasets``):

    Document source -> deterministic tokenization & packing -> immutable token shards + manifest
        -> PretrainBatchSource (sampling/batching/resume) -> TrainingEngine

The SFT loader lives in ``data.adapters`` and is intentionally *not* routed through the
pretraining ``PretrainBatchSource`` interface.
"""

from __future__ import annotations

from .errors import (
    ChecksumMismatchError,
    ConcurrentProducerError,
    DataError,
    DataExhausted,
    DataWaitTimeout,
    HFDependencyError,
    ManifestError,
    ManifestVersionError,
    ProducerFailedError,
    ResumeIncompatibleError,
    ShardError,
)
from .legacy import LegacyShardDirSource, find_legacy_shards, migrate_legacy_directory
from .manifest import (
    MANIFEST_FORMAT_VERSION,
    Manifest,
    ShardEntry,
    SourceSpec,
    SplitState,
    TokenizerSpec,
    TransformSpec,
    load_manifest,
    resolve_shard_path,
)
from .packing import TokenPacker, make_batch, tokenizer_fingerprint
from .protocol import PretrainBatchInfo, PretrainBatchSource
from .shard_source import ManifestShardSource, verify_resume_compatible
from .shard_store import ShardStore, read_shard

__all__ = [
    "ChecksumMismatchError",
    "ConcurrentProducerError",
    "DataError",
    "DataExhausted",
    "DataWaitTimeout",
    "HFDependencyError",
    "LegacyShardDirSource",
    "MANIFEST_FORMAT_VERSION",
    "Manifest",
    "ManifestError",
    "ManifestShardSource",
    "ManifestVersionError",
    "PretrainBatchInfo",
    "PretrainBatchSource",
    "ProducerFailedError",
    "ResumeIncompatibleError",
    "ShardEntry",
    "ShardError",
    "ShardStore",
    "SourceSpec",
    "SplitState",
    "TokenPacker",
    "TokenizerSpec",
    "TransformSpec",
    "find_legacy_shards",
    "load_manifest",
    "make_batch",
    "migrate_legacy_directory",
    "read_shard",
    "resolve_shard_path",
    "tokenizer_fingerprint",
    "verify_resume_compatible",
]
