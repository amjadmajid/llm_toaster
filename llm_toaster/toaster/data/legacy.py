"""Compatibility shims for the pre-manifest data layout.

During the deprecation period we must (a) keep training over an old shard *directory* working
unchanged, and (b) offer a one-shot migration that registers those existing files into a
manifest **without retokenizing**.

``LegacyShardDirSource`` deliberately preserves the original ``DataLoaderLite`` semantics --
including silent modulo wrapping -- so existing configs behave identically (the engine emits a
single deprecation warning pointing at the migration command). New runs should use a manifest.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from .errors import ShardError
from .manifest import Manifest, ShardEntry, SourceSpec, TokenizerSpec, TransformSpec, utcnow_iso
from .packing import make_batch, tokenizer_fingerprint
from .protocol import PretrainBatchInfo
from .shard_store import read_shard, sha256_file

logger = logging.getLogger(__name__)

SOURCE_MODE = "legacy"
SUPPORTED_EXTS = (".npy", ".txt", ".tokens")


def find_legacy_shards(directory: str | os.PathLike, split: str | None = None) -> list[Path]:
    """Sorted shard files in a legacy directory, optionally filtered to a split substring."""
    directory = Path(directory)
    if not directory.exists():
        return []
    shards = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix in SUPPORTED_EXTS and (split is None or split in p.name)
    ]
    return sorted(shards)


class LegacyShardDirSource:
    """Reads a legacy shard directory with the original wrapping behavior. Deprecated."""

    def __init__(self, data_root: str, split: str, batch_size: int, seq_len: int):
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        self.data_root = str(data_root)
        self.split = split
        self.B = batch_size
        self.T = seq_len
        self.shards = [str(p) for p in find_legacy_shards(data_root, split)]
        if not self.shards:
            raise FileNotFoundError(f"No '{split}' shards found in {data_root}")
        self.current_shard = 0
        self.current_position = 0
        self.pass_index = 0
        self.repeated = False
        self._tokens = read_shard(self.shards[0], mmap=False)

    def _advance_shard(self) -> None:
        nxt = (self.current_shard + 1) % len(self.shards)
        if nxt <= self.current_shard:  # wrapped back to the start
            self.pass_index += 1
            self.repeated = True
            logger.info("legacy loader wrapped to pass %d (split %r)", self.pass_index, self.split)
        self.current_shard = nxt
        self._tokens = read_shard(self.shards[self.current_shard], mmap=False)
        self.current_position = 0

    def next_batch(self) -> tuple[np.ndarray, np.ndarray, PretrainBatchInfo]:
        need = self.B * self.T + 1
        if self.current_position + need > len(self._tokens):
            self._advance_shard()
        if self.current_position + need > len(self._tokens):
            raise ShardError(
                f"shard {self.shards[self.current_shard]!r} has only {len(self._tokens)} tokens, "
                f"fewer than B*T+1 = {need} (B={self.B}, T={self.T})."
            )
        x, y = make_batch(self._tokens, self.current_position, self.B, self.T)
        self.current_position += self.B * self.T
        info = PretrainBatchInfo(
            shard_id=os.path.basename(self.shards[self.current_shard]),
            token_offset=self.current_position,
            pass_index=self.pass_index,
            repeated=self.repeated,
            source_mode=SOURCE_MODE,
        )
        return np.asarray(x), np.asarray(y), info

    def reset(self) -> None:
        self.current_shard = 0
        self.current_position = 0
        self.pass_index = 0
        self.repeated = False
        self._tokens = read_shard(self.shards[0], mmap=False)

    def close(self) -> None:
        self._tokens = None

    def state_dict(self) -> dict:
        return {
            "source_mode": SOURCE_MODE,
            "current_shard": self.current_shard,
            "current_position": self.current_position,
            "pass_index": self.pass_index,
        }

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.current_shard = int(state.get("current_shard", 0)) % len(self.shards)
        self.pass_index = int(state.get("pass_index", 0))
        self.repeated = self.pass_index > 0
        self._tokens = read_shard(self.shards[self.current_shard], mmap=False)
        self.current_position = int(state.get("current_position", 0))

    def stats(self) -> dict:
        return {
            "source_mode": SOURCE_MODE,
            "current_shard_id": os.path.basename(self.shards[self.current_shard]),
            "data_pass": self.pass_index,
        }


def migrate_legacy_directory(
    data_dir: str | os.PathLike,
    manifest_path: str | os.PathLike,
    *,
    dataset_id: str | None = None,
    tokenizer=None,
    add_eot: bool = True,
    overwrite: bool = False,
) -> Manifest:
    """Register existing shard files into a manifest **without retokenizing**.

    Files are referenced in place by relative path (nothing is rewritten or moved). ``train``/
    ``val`` splits are detected from filenames; ``val`` is normalized to the ``validation`` split.
    Each shard's token count, byte size, and sha256 are recorded so resume/validation work.
    """
    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not overwrite:
        raise ShardError(f"manifest already exists at {manifest_path}; pass overwrite=True to replace it")

    split_files: dict[str, list[Path]] = {"train": [], "validation": []}
    for path in find_legacy_shards(data_dir):
        if "val" in path.name:
            split_files["validation"].append(path)
        elif "train" in path.name:
            split_files["train"].append(path)
        else:
            split_files["train"].append(path)  # default un-tagged shards to train
    if not split_files["train"] and not split_files["validation"]:
        raise ShardError(f"no legacy shard files found in {data_dir}")

    dtype_name = _infer_dtype(split_files)
    fingerprint = tokenizer_fingerprint(tokenizer) if tokenizer is not None else None
    manifest = Manifest(
        dataset_id=dataset_id or f"legacy-{data_dir.name}",
        source=SourceSpec(type="legacy", dataset_name=str(data_dir), resolved_revision=None),
        tokenizer=TokenizerSpec(
            type=getattr(getattr(tokenizer, "__class__", None), "__name__", "unknown"),
            name=getattr(tokenizer, "name", "unknown"),
            vocab_size=getattr(tokenizer, "vocab_size", None),
            fingerprint=fingerprint,
        ),
        transform=TransformSpec(add_eot=add_eot, packing="contiguous", shard_tokens=0, dtype=dtype_name),
    )
    manifest_dir = manifest_path.resolve().parent
    for split, files in split_files.items():
        for path in files:
            tokens = read_shard(path, mmap=True)
            ntokens = int(tokens.shape[0])
            if ntokens <= 0:
                raise ShardError(f"legacy shard {path} has no tokens")
            sha256_hex, nbytes = sha256_file(path)
            index = manifest.next_index(split)
            rel = os.path.relpath(path.resolve(), manifest_dir).replace(os.sep, "/")
            manifest.append_shard(
                ShardEntry(
                    id=manifest.shard_id_for(split, index),
                    index=index,
                    split=split,
                    path=rel,
                    tokens=ntokens,
                    dtype=dtype_name,
                    bytes=nbytes,
                    sha256=sha256_hex,
                    created_at=utcnow_iso(),
                )
            )
        if files:
            manifest.mark_complete(split)
    manifest.save_atomic(manifest_path)
    logger.info(
        "migrated %d train + %d validation legacy shards into %s",
        len(split_files["train"]),
        len(split_files["validation"]),
        manifest_path,
    )
    return manifest


def _infer_dtype(split_files: dict[str, list[Path]]) -> str:
    for files in split_files.values():
        for path in files:
            if path.suffix == ".npy":
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                name = str(arr.dtype)
                return name if name in {"uint16", "uint32", "int32"} else "int32"
    return "int32"  # text fixtures are read as int64 but stored small; int32 is the safe label
