"""Atomic, append-only shard publication.

A consumer must never observe a partially written shard, so :meth:`ShardStore.publish_shard`
follows a strict sequence:

1. write tokens to a ``.partial`` file in the *destination* filesystem,
2. ``flush`` + ``fsync`` the bytes,
3. compute the sha256 checksum,
4. atomically ``os.replace`` into the final ``.npy`` name,
5. append the sealed :class:`ShardEntry` to the in-memory manifest,
6. publish the manifest atomically.

The manifest is only mutated *after* the rename succeeds, so a crash before step 4 leaves no
committed entry and only a ``.partial`` file (ignored by the trainer, cleaned up on restart).
A single-writer lock guards against accidental concurrent producers corrupting the manifest.
"""

from __future__ import annotations

import logging
import os
import socket
import tempfile
from pathlib import Path

import numpy as np

from .errors import ChecksumMismatchError, ConcurrentProducerError, ShardError
from .manifest import MANIFEST_FILENAME, Manifest, ShardEntry, utcnow_iso
from .packing import dtype_for

logger = logging.getLogger(__name__)

SHARDS_SUBDIR = "shards"
LOCK_FILENAME = ".writer.lock"


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> tuple[str, int]:
    """Return ``(sha256_hex, num_bytes)`` for a file, read in bounded chunks."""
    import hashlib

    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


class WriterLock:
    """Best-effort single-writer lock via an ``O_CREAT | O_EXCL`` lock file.

    Holds ``pid``/``host`` so a *stale* lock from a dead process on this host is reclaimed,
    while a live concurrent writer is rejected with :class:`ConcurrentProducerError`.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._held = False

    def acquire(self) -> "WriterLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            self._reclaim_or_reject()
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}@{socket.gethostname()} {utcnow_iso()}\n")
        self._held = True
        return self

    def _reclaim_or_reject(self) -> None:
        try:
            content = self.path.read_text(encoding="utf-8").strip()
            ident = content.split()[0]
            pid_str, _, host = ident.partition("@")
            pid = int(pid_str)
        except (OSError, ValueError, IndexError):
            # Unreadable/garbage lock -> treat as stale and reclaim.
            self.path.unlink(missing_ok=True)
            return
        if host == socket.gethostname() and not _pid_alive(pid):
            logger.warning("Reclaiming stale writer lock at %s (dead pid %s)", self.path, pid)
            self.path.unlink(missing_ok=True)
            return
        raise ConcurrentProducerError(
            f"another writer holds {self.path} ({content!r}). Only one producer/preparer may write "
            f"a shard store at a time. If this is stale, remove the lock file and retry."
        )

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def __enter__(self) -> "WriterLock":
        return self.acquire()

    def __exit__(self, *_exc) -> None:
        self.release()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ShardStore:
    """Owns the on-disk layout (``<store_dir>/manifest.json`` + ``<store_dir>/shards/``)."""

    def __init__(self, store_dir: str | os.PathLike, manifest_path: str | os.PathLike | None = None):
        self.store_dir = Path(store_dir)
        self.manifest_path = Path(manifest_path) if manifest_path else self.store_dir / MANIFEST_FILENAME
        self.shards_dir = self.store_dir / SHARDS_SUBDIR

    def lock(self) -> WriterLock:
        return WriterLock(self.store_dir / LOCK_FILENAME)

    def cleanup_partials(self) -> int:
        """Remove leftover ``.partial`` / temp files from an interrupted write. Returns the count."""
        removed = 0
        if not self.shards_dir.exists():
            return 0
        for pattern in ("*.partial", ".tmp_manifest_*", ".tmp_shard_*"):
            for path in self.shards_dir.glob(pattern):
                path.unlink(missing_ok=True)
                removed += 1
        for path in self.store_dir.glob(".tmp_manifest_*"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def publish_shard(self, manifest: Manifest, tokens: np.ndarray, split: str) -> ShardEntry:
        """Run the full atomic publication sequence and return the sealed entry."""
        if tokens.size == 0:
            raise ShardError("refusing to publish an empty shard")
        index = manifest.next_index(split)
        shard_id = manifest.shard_id_for(split, index)
        dtype_name = manifest.transform.dtype
        tokens = np.ascontiguousarray(tokens, dtype=dtype_for(dtype_name))

        self.shards_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.shards_dir / f"{shard_id}.npy"
        self._guard_no_overwrite(manifest, split, final_path)

        # 1-3: write to a .partial handle (np.save won't append .npy to an open file), fsync, checksum.
        fd, tmp = tempfile.mkstemp(dir=str(self.shards_dir), prefix=f".tmp_shard_{shard_id}_", suffix=".partial")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, tokens, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            sha256_hex, nbytes = sha256_file(tmp)
            # 4: atomic rename into the final shard name.
            os.replace(tmp, final_path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

        # 5-6: only now mutate + publish the manifest (a sealed shard always has a published entry).
        rel_path = os.path.relpath(final_path, self.manifest_path.resolve().parent)
        entry = ShardEntry(
            id=shard_id,
            index=index,
            split=split,
            path=rel_path.replace(os.sep, "/"),
            tokens=int(tokens.size),
            dtype=dtype_name,
            bytes=nbytes,
            sha256=sha256_hex,
            created_at=utcnow_iso(),
        )
        manifest.append_shard(entry)
        manifest.save_atomic(self.manifest_path)
        logger.info("sealed shard %s (%d tokens, gen=%d)", shard_id, entry.tokens, manifest.generation)
        return entry

    def _guard_no_overwrite(self, manifest: Manifest, split: str, final_path: Path) -> None:
        """Never clobber a *committed* shard; an orphan from a failed publish is reclaimable.

        Commitment is judged against both the in-memory manifest and the authoritative on-disk
        manifest, so a re-run with a stale/empty manifest still cannot overwrite a shard that was
        already published (the "lost manifest" case).
        """
        if not final_path.exists():
            return
        rel = os.path.relpath(final_path, self.manifest_path.resolve().parent).replace(os.sep, "/")
        manifests = [manifest]
        if self.manifest_path.exists():
            from .manifest import load_manifest

            try:
                manifests.append(load_manifest(self.manifest_path))
            except Exception:  # pragma: no cover - a corrupt on-disk manifest is handled elsewhere
                pass
        committed = any(entry.path == rel for m in manifests for state in m.splits.values() for entry in state.shards)
        if committed:
            raise ShardError(
                f"refusing to overwrite existing shard {final_path}. Shards are immutable; "
                f"to rebuild, choose a fresh store_dir or remove the old dataset explicitly."
            )
        logger.warning("removing orphan shard file %s (not in manifest; likely a failed prior publish)", final_path)
        final_path.unlink(missing_ok=True)


def read_shard(path: str | os.PathLike, *, mmap: bool = True, expected_sha256: str | None = None) -> np.ndarray:
    """Load a shard, optionally memory-mapped, verifying the checksum when provided."""
    path = Path(path)
    if not path.exists():
        raise ShardError(f"shard file missing: {path}")
    if expected_sha256 is not None:
        actual, _ = sha256_file(path)
        if actual != expected_sha256:
            raise ChecksumMismatchError(
                f"shard {path} sha256 {actual} does not match the manifest's {expected_sha256}; "
                f"the file is corrupt or was modified."
            )
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    # Tiny text fixtures: integer token ids separated by whitespace or commas.
    with path.open("r", encoding="utf-8") as handle:
        raw = handle.read().replace(",", " ").split()
    return np.asarray([int(token) for token in raw], dtype=np.int64)
