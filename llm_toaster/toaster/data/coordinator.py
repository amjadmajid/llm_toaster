"""``PrefetchCoordinator`` -- supervises the background producer from the trainer process.

It launches the producer in a separate process (``spawn`` start method, so CUDA/torch state is
never forked), waits until enough data is ready, exposes the producer's status to the wait-mode
:class:`ManifestShardSource`, publishes the trainer's consumer cursor to bound the queue, and
guarantees the subprocess is torn down on normal exit, error, SIGINT, and SIGTERM.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import time
from pathlib import Path

from .errors import DataWaitTimeout, ProducerFailedError
from .manifest import load_manifest
from .producer import (
    CONSUMER_FILENAME,
    PRODUCER_CKPT_FILENAME,
    STATUS_FILENAME,
    run_producer_process,
)
from .statefiles import atomic_write_json, read_json

logger = logging.getLogger(__name__)


class PrefetchCoordinator:
    """Owns the producer subprocess and the trainer-side status/consumer files."""

    def __init__(self, spec: dict, *, target=run_producer_process, start_method: str = "spawn"):
        self.spec = spec
        self.store_dir = Path(spec["store_dir"])
        self.manifest_path = Path(spec.get("manifest_path") or self.store_dir / "manifest.json")
        self.status_path = self.store_dir / STATUS_FILENAME
        self.consumer_path = self.store_dir / CONSUMER_FILENAME
        self.ckpt_path = self.store_dir / PRODUCER_CKPT_FILENAME
        self._target = target
        self._ctx = mp.get_context(start_method)
        self._stop_event = self._ctx.Event()
        self._process: mp.process.BaseProcess | None = None
        self._closed = False

    # ----- lifecycle ------------------------------------------------------
    def start(self) -> "PrefetchCoordinator":
        if self._process is not None:
            return self
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._process = self._ctx.Process(
            target=self._target, args=(self.spec, self._stop_event), name="llm-toaster-producer", daemon=False
        )
        self._process.start()
        atexit.register(self.stop)
        logger.info("prefetch producer started (pid=%s) -> %s", self._process.pid, self.store_dir)
        return self

    def status(self) -> dict:
        data = read_json(self.status_path) or {"status": "starting"}
        if self._process is not None and not self._process.is_alive():
            # Process gone but no terminal status recorded -> treat a non-zero exit as failure.
            if data.get("status") not in {"complete", "failed"} and (self._process.exitcode or 0) != 0:
                data = {**data, "status": "failed", "error": f"producer exited with code {self._process.exitcode}"}
        return data

    def update_consumer(self, consumed_train_shards: int) -> None:
        atomic_write_json(self.consumer_path, {"consumed_train_shards": int(consumed_train_shards)})

    def wait_until_ready(self, min_ready_shards: int, timeout_s: float, poll_interval_s: float = 0.5) -> None:
        """Block until validation is frozen and ``min_ready_shards`` train shards exist.

        Raises :class:`ProducerFailedError` if the producer fails and :class:`DataWaitTimeout`
        if the data does not become ready in time -- never waits forever.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.status()
            if status.get("status") == "failed":
                raise ProducerFailedError(f"prefetch producer failed before data was ready: {status.get('error')}")
            ready = self._ready(min_ready_shards)
            if ready:
                return
            if status.get("status") == "complete" and not ready:
                # Producer finished but couldn't reach the readiness bar (e.g. tiny source).
                if self._train_shards() >= min_ready_shards and self._validation_ready():
                    return
                raise ProducerFailedError(
                    "producer completed without materializing the requested validation/train data; "
                    "check the source budget."
                )
            if time.monotonic() >= deadline:
                raise DataWaitTimeout(
                    f"waited {timeout_s:.0f}s for prefetch readiness "
                    f"(need validation + {min_ready_shards} train shards); producer status "
                    f"{status.get('status')!r}."
                )
            time.sleep(poll_interval_s)

    def _ready(self, min_ready_shards: int) -> bool:
        return self._validation_ready() and self._train_shards() >= min_ready_shards

    def _validation_ready(self) -> bool:
        """Validation must be a *frozen* (complete) split before training starts."""
        if not self.spec.get("val_shards", 0):
            return True  # no validation requested
        if not self.manifest_path.exists():
            return False
        try:
            return load_manifest(self.manifest_path).split("validation").complete
        except Exception:  # pragma: no cover - manifest may be mid-publish
            return False

    def _train_shards(self) -> int:
        if not self.manifest_path.exists():
            return 0
        try:
            return load_manifest(self.manifest_path).next_index("train")
        except Exception:  # pragma: no cover
            return 0

    def stop(self, timeout_s: float = 10.0) -> None:
        """Signal the producer to stop and ensure the subprocess is gone (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        process = self._process
        if process is None:
            return
        process.join(timeout_s)
        if process.is_alive():
            logger.warning("prefetch producer did not stop in %.1fs; terminating", timeout_s)
            process.terminate()
            process.join(timeout_s)
        if process.is_alive():  # pragma: no cover - last resort
            process.kill()
            process.join()
        logger.info("prefetch producer stopped (exitcode=%s)", process.exitcode)

    def __enter__(self) -> "PrefetchCoordinator":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()
