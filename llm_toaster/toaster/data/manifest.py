"""Versioned, append-only JSON manifest describing an immutable token-shard dataset.

The manifest is the single source of truth a trainer consults -- never a directory listing.
It separates an **immutable identity** (``dataset_fingerprint`` over source + tokenizer +
transform) from a **mutable append generation** (``generation``, bumped on every committed
update), so a prefetch producer can keep appending sealed shards while an earlier checkpoint
remains resume-compatible as long as its committed prefix is unchanged.

Design rules enforced here:
- existing shard entries are never mutated in place; updates are append-only,
- only top-level status/counter fields change (``generation``, ``updated_at``, per-split
  ``complete``/``tokens``),
- ``generation`` strictly increases on each committed update,
- all reads validate structure with contextual :class:`ManifestError` messages carrying the
  full key path,
- manifests are published atomically (temp file -> ``fsync`` -> ``os.replace``).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .errors import ManifestError, ManifestVersionError
from .packing import SUPPORTED_DTYPES, hash_canonical

MANIFEST_FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


def utcnow_iso() -> str:
    """Timezone-aware UTC timestamp, e.g. ``2026-06-22T10:30:00+00:00``."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require(condition: bool, key_path: str, message: str) -> None:
    if not condition:
        raise ManifestError(f"manifest[{key_path}]: {message}")


def _get(mapping: dict, key: str, key_path: str, *, expected: type | tuple[type, ...]) -> object:
    if not isinstance(mapping, dict):
        raise ManifestError(f"manifest[{key_path}]: expected a mapping, got {type(mapping).__name__}")
    if key not in mapping:
        raise ManifestError(f"manifest[{key_path}.{key}]: required field is missing")
    value = mapping[key]
    if not isinstance(value, expected):
        names = expected.__name__ if isinstance(expected, type) else "/".join(t.__name__ for t in expected)
        raise ManifestError(f"manifest[{key_path}.{key}]: expected {names}, got {type(value).__name__}")
    return value


@dataclass
class SourceSpec:
    type: str = "huggingface"
    dataset_name: str = ""
    config_name: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None
    split: str = "train"
    text_field: str = "text"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "dataset_name": self.dataset_name,
            "config_name": self.config_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "split": self.split,
            "text_field": self.text_field,
        }

    @staticmethod
    def from_dict(data: dict, key_path: str = "source") -> "SourceSpec":
        _require(isinstance(data, dict), key_path, "must be a mapping")
        return SourceSpec(
            type=str(_get(data, "type", key_path, expected=str)),
            dataset_name=str(data.get("dataset_name", "")),
            config_name=data.get("config_name"),
            requested_revision=data.get("requested_revision"),
            resolved_revision=data.get("resolved_revision"),
            split=str(data.get("split", "train")),
            text_field=str(data.get("text_field", "text")),
        )

    def identity(self) -> dict:
        """Identity fields folded into ``dataset_fingerprint`` (mutable counters excluded)."""
        return {
            "type": self.type,
            "dataset_name": self.dataset_name,
            "config_name": self.config_name,
            "resolved_revision": self.resolved_revision,
            "split": self.split,
            "text_field": self.text_field,
        }


@dataclass
class TokenizerSpec:
    type: str = "tiktoken"
    name: str = "gpt2"
    vocab_size: int | None = None
    fingerprint: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "vocab_size": self.vocab_size,
            "fingerprint": self.fingerprint,
        }

    @staticmethod
    def from_dict(data: dict, key_path: str = "tokenizer") -> "TokenizerSpec":
        _require(isinstance(data, dict), key_path, "must be a mapping")
        return TokenizerSpec(
            type=str(data.get("type", "tiktoken")),
            name=str(data.get("name", "gpt2")),
            vocab_size=data.get("vocab_size"),
            fingerprint=data.get("fingerprint"),
        )


@dataclass
class TransformSpec:
    add_eot: bool = True
    packing: str = "contiguous"
    shard_tokens: int = 10_000_000
    dtype: str = "uint16"

    def to_dict(self) -> dict:
        return {
            "add_eot": self.add_eot,
            "packing": self.packing,
            "shard_tokens": self.shard_tokens,
            "dtype": self.dtype,
        }

    @staticmethod
    def from_dict(data: dict, key_path: str = "transform") -> "TransformSpec":
        _require(isinstance(data, dict), key_path, "must be a mapping")
        dtype = str(data.get("dtype", "uint16"))
        _require(dtype in SUPPORTED_DTYPES, f"{key_path}.dtype", f"unsupported dtype {dtype!r}")
        return TransformSpec(
            add_eot=bool(data.get("add_eot", True)),
            packing=str(data.get("packing", "contiguous")),
            shard_tokens=int(data.get("shard_tokens", 10_000_000)),
            dtype=dtype,
        )

    def identity(self) -> dict:
        """Transform fields that change token *content/packing semantics* (not boundary size)."""
        return {"add_eot": self.add_eot, "packing": self.packing, "dtype": self.dtype}


@dataclass
class ShardEntry:
    id: str
    index: int
    split: str
    path: str
    tokens: int
    dtype: str
    bytes: int
    sha256: str
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "index": self.index,
            "split": self.split,
            "path": self.path,
            "tokens": self.tokens,
            "dtype": self.dtype,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict, key_path: str) -> "ShardEntry":
        _require(isinstance(data, dict), key_path, "must be a mapping")
        dtype = str(_get(data, "dtype", key_path, expected=str))
        _require(dtype in SUPPORTED_DTYPES, f"{key_path}.dtype", f"unsupported dtype {dtype!r}")
        tokens = int(_get(data, "tokens", key_path, expected=int))
        _require(tokens > 0, f"{key_path}.tokens", f"must be positive, got {tokens}")
        nbytes = int(_get(data, "bytes", key_path, expected=int))
        _require(nbytes > 0, f"{key_path}.bytes", f"must be positive, got {nbytes}")
        index = int(_get(data, "index", key_path, expected=int))
        _require(index >= 0, f"{key_path}.index", f"must be non-negative, got {index}")
        return ShardEntry(
            id=str(_get(data, "id", key_path, expected=str)),
            index=index,
            split=str(_get(data, "split", key_path, expected=str)),
            path=str(_get(data, "path", key_path, expected=str)),
            tokens=tokens,
            dtype=dtype,
            bytes=nbytes,
            sha256=str(_get(data, "sha256", key_path, expected=str)),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class SplitState:
    complete: bool = False
    tokens: int = 0
    shards: list[ShardEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "complete": self.complete,
            "tokens": self.tokens,
            "shards": [s.to_dict() for s in self.shards],
        }

    @staticmethod
    def from_dict(data: dict, key_path: str) -> "SplitState":
        _require(isinstance(data, dict), key_path, "must be a mapping")
        shards_raw = data.get("shards", [])
        _require(isinstance(shards_raw, list), f"{key_path}.shards", "must be a list")
        shards = [ShardEntry.from_dict(s, f"{key_path}.shards[{i}]") for i, s in enumerate(shards_raw)]
        return SplitState(
            complete=bool(data.get("complete", False)),
            tokens=int(data.get("tokens", 0)),
            shards=shards,
        )


class Manifest:
    """In-memory view of a manifest file with append-only mutation helpers."""

    def __init__(
        self,
        *,
        dataset_id: str,
        source: SourceSpec,
        tokenizer: TokenizerSpec,
        transform: TransformSpec,
        splits: dict[str, SplitState] | None = None,
        generation: int = 0,
        created_at: str | None = None,
        updated_at: str | None = None,
        format_version: int = MANIFEST_FORMAT_VERSION,
        dataset_fingerprint: str | None = None,
    ):
        self.format_version = format_version
        self.dataset_id = dataset_id
        self.source = source
        self.tokenizer = tokenizer
        self.transform = transform
        self.splits: dict[str, SplitState] = splits if splits is not None else {}
        self.generation = generation
        self.created_at = created_at or utcnow_iso()
        self.updated_at = updated_at or self.created_at
        self.dataset_fingerprint = dataset_fingerprint or self.compute_dataset_fingerprint()

    # ----- identity -------------------------------------------------------
    def compute_dataset_fingerprint(self) -> str:
        """Immutable fingerprint over source identity + tokenizer fingerprint + transform semantics."""
        return hash_canonical(
            {
                "source": self.source.identity(),
                "tokenizer": self.tokenizer.fingerprint,
                "transform": self.transform.identity(),
            }
        )

    def split(self, name: str) -> SplitState:
        return self.splits.setdefault(name, SplitState())

    def next_index(self, split: str) -> int:
        shards = self.splits.get(split, SplitState()).shards
        return shards[-1].index + 1 if shards else 0

    def shard_id_for(self, split: str, index: int) -> str:
        return f"{split}-{index:06d}"

    def prefix_fingerprint(self, split: str, through_shard_id: str | None) -> str:
        """Hash of the committed shard prefix up to and including ``through_shard_id``.

        Resume verifies this is unchanged; later appended shards do not affect it, so a growing
        prefetch manifest stays compatible with an earlier checkpoint.
        """
        prefix: list[dict] = []
        for entry in self.splits.get(split, SplitState()).shards:
            prefix.append({"id": entry.id, "sha256": entry.sha256, "tokens": entry.tokens})
            if entry.id == through_shard_id:
                break
        return hash_canonical({"split": split, "prefix": prefix})

    def find_shard(self, split: str, shard_id: str) -> ShardEntry | None:
        for entry in self.splits.get(split, SplitState()).shards:
            if entry.id == shard_id:
                return entry
        return None

    # ----- append-only mutation ------------------------------------------
    def append_shard(self, entry: ShardEntry) -> None:
        """Append a sealed shard. Bumps ``generation`` and the split token counter.

        Rejects duplicate ids/paths (any split), non-monotonic/non-contiguous indexes, and
        appends to a split already marked complete.
        """
        for split_name, state in self.splits.items():
            for existing in state.shards:
                if existing.id == entry.id:
                    raise ManifestError(f"duplicate shard id {entry.id!r} (already in split {split_name!r})")
                if existing.path == entry.path:
                    raise ManifestError(f"duplicate shard path {entry.path!r} (already in split {split_name!r})")
        state = self.split(entry.split)
        if state.complete:
            raise ManifestError(f"split {entry.split!r} is marked complete; cannot append {entry.id!r}")
        expected_index = self.next_index(entry.split)
        if entry.index != expected_index:
            raise ManifestError(
                f"shard {entry.id!r} index {entry.index} is not the next contiguous index "
                f"{expected_index} for split {entry.split!r}"
            )
        state.shards.append(entry)
        state.tokens += entry.tokens
        self.generation += 1
        self.updated_at = utcnow_iso()

    def mark_complete(self, split: str, complete: bool = True) -> None:
        self.split(split).complete = complete
        self.generation += 1
        self.updated_at = utcnow_iso()

    # ----- serialization --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "generation": self.generation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "transform": self.transform.to_dict(),
            "splits": {name: state.to_dict() for name, state in self.splits.items()},
        }

    @staticmethod
    def from_dict(data: dict) -> "Manifest":
        if not isinstance(data, dict):
            raise ManifestError(f"manifest: top-level must be a mapping, got {type(data).__name__}")
        version = data.get("format_version", 1)
        if not isinstance(version, int):
            raise ManifestError("manifest[format_version]: must be an integer")
        if version > MANIFEST_FORMAT_VERSION:
            raise ManifestVersionError(
                f"manifest format_version {version} is newer than this build understands "
                f"(<= {MANIFEST_FORMAT_VERSION}); upgrade LLM Toaster."
            )
        splits_raw = data.get("splits", {})
        _require(isinstance(splits_raw, dict), "splits", "must be a mapping of split-name -> state")
        splits = {name: SplitState.from_dict(state, f"splits.{name}") for name, state in splits_raw.items()}
        manifest = Manifest(
            dataset_id=str(_get(data, "dataset_id", "", expected=str)),
            source=SourceSpec.from_dict(_get(data, "source", "", expected=dict)),
            tokenizer=TokenizerSpec.from_dict(_get(data, "tokenizer", "", expected=dict)),
            transform=TransformSpec.from_dict(_get(data, "transform", "", expected=dict)),
            splits=splits,
            generation=int(data.get("generation", 0)),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            format_version=version,
            dataset_fingerprint=data.get("dataset_fingerprint"),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        """Structural invariants: contiguous monotonic indexes, unique ids/paths, sane counters."""
        _require(bool(self.dataset_id), "dataset_id", "must be non-empty")
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        for split_name, state in self.splits.items():
            running = 0
            for position, entry in enumerate(state.shards):
                key = f"splits.{split_name}.shards[{position}]"
                _require(entry.split == split_name, f"{key}.split", f"is {entry.split!r}, expected {split_name!r}")
                _require(
                    entry.index == position,
                    f"{key}.index",
                    f"is {entry.index}, expected contiguous index {position}",
                )
                _require(entry.id not in seen_ids, f"{key}.id", f"duplicate shard id {entry.id!r}")
                _require(entry.path not in seen_paths, f"{key}.path", f"duplicate shard path {entry.path!r}")
                seen_ids.add(entry.id)
                seen_paths.add(entry.path)
                running += entry.tokens
            _require(
                state.tokens == running,
                f"splits.{split_name}.tokens",
                f"recorded {state.tokens} but shards sum to {running}",
            )

    # ----- atomic IO ------------------------------------------------------
    def save_atomic(self, path: str | os.PathLike) -> None:
        """Publish the manifest atomically: temp file -> fsync -> ``os.replace``."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_manifest_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise


def load_manifest(path: str | os.PathLike) -> Manifest:
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {path} is not valid JSON: {exc}") from exc
    return Manifest.from_dict(data)


def resolve_shard_path(manifest_path: str | os.PathLike, entry: ShardEntry) -> Path:
    """Resolve a shard's (normally relative) path against the manifest's directory."""
    shard_path = Path(entry.path)
    if shard_path.is_absolute():
        return shard_path
    return Path(manifest_path).resolve().parent / shard_path
