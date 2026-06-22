"""Bounded background producer: HF stream -> tokenize/pack -> atomic immutable shards.

The producer writes *exactly* the same manifest-backed shards a ``prepare`` job would, so the
trainer consumes prefetch and prepared data through the identical :class:`ManifestShardSource`.

Guarantees:
- validation is materialized first and frozen (its split marked complete) before train shards,
- production pauses when sealed-but-unconsumed train shards reach ``prefetch_shards`` and resumes
  as the trainer drains the queue (the trainer publishes a consumer cursor),
- a producer checkpoint is written atomically after every sealed shard; a crash during an
  *unsealed* (partial) shard discards it and regenerates from the last sealed boundary, never
  duplicating or altering a committed manifest entry,
- failures set a ``failed`` status (with traceback) and are re-raised -- never swallowed.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass

import numpy as np

from .errors import ShardError
from .manifest import Manifest, SourceSpec, TokenizerSpec, TransformSpec, load_manifest
from .packing import TokenPacker, tokenizer_fingerprint
from .shard_source import transform_identity_hash
from .shard_store import ShardStore
from .statefiles import atomic_write_json, read_json

logger = logging.getLogger(__name__)

STATUS_FILENAME = ".producer_status.json"
PRODUCER_CKPT_FILENAME = ".producer_ckpt.json"
CONSUMER_FILENAME = ".consumer.json"


@dataclass
class ProducerBudgets:
    """How many uniform ``shard_tokens`` shards to materialize per split."""

    val_shards: int
    train_shards: int


class ShardProducer:
    """Drives one pass over a document stream into validation-then-train shards."""

    def __init__(
        self,
        store: ShardStore,
        manifest: Manifest,
        packer: TokenPacker,
        source_factory,
        text_field: str,
        budgets: ProducerBudgets,
        *,
        prefetch_shards: int = 3,
        stop_event: threading.Event | None = None,
        poll_interval_s: float = 0.25,
    ):
        self.store = store
        self.manifest = manifest
        self.packer = packer
        self.source_factory = source_factory
        self.text_field = text_field
        self.budgets = budgets
        self.shard_tokens = manifest.transform.shard_tokens
        if self.shard_tokens <= 0:
            raise ValueError("transform.shard_tokens must be positive for the producer")
        self.prefetch_shards = max(1, prefetch_shards)
        self.stop_event = stop_event
        self.poll_interval_s = poll_interval_s

        self.status_path = store.store_dir / STATUS_FILENAME
        self.ckpt_path = store.store_dir / PRODUCER_CKPT_FILENAME
        self.consumer_path = store.store_dir / CONSUMER_FILENAME

        self._records = None
        self._buffer: list[int] = []
        self._records_consumed = 0
        self._source_exhausted = False

    # ----- status / checkpoint -------------------------------------------
    def _set_status(self, status: str, **extra) -> None:
        payload = {
            "status": status,
            "updated_at": time.time(),
            "last_train_shard": self.manifest.next_index("train") - 1,
            "last_val_shard": self.manifest.next_index("validation") - 1,
            "manifest_generation": self.manifest.generation,
        }
        payload.update(extra)
        atomic_write_json(self.status_path, payload)

    def _write_ckpt(self) -> None:
        atomic_write_json(
            self.ckpt_path,
            {
                "resolved_revision": self.manifest.source.resolved_revision,
                "tokenizer_fingerprint": self.manifest.tokenizer.fingerprint,
                "transform_fingerprint": transform_identity_hash(self.manifest),
                "records_consumed": self._records_consumed,
                "carry_tokens": list(self._buffer),
                "next_val_index": self.manifest.next_index("validation"),
                "next_train_index": self.manifest.next_index("train"),
                "validation_complete": self.manifest.split("validation").complete,
                "train_complete": self.manifest.split("train").complete,
                "manifest_generation": self.manifest.generation,
            },
        )

    # ----- stream helpers -------------------------------------------------
    def _stopped(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def _wait(self, seconds: float) -> None:
        if self.stop_event is not None:
            self.stop_event.wait(seconds)
        else:  # pragma: no cover - producers always run with a stop event in practice
            time.sleep(seconds)

    def _pull_record(self) -> bool:
        try:
            record = next(self._records)
        except StopIteration:
            self._source_exhausted = True
            return False
        if self.text_field not in record:
            raise ShardError(f"record missing text_field {self.text_field!r}; keys: {sorted(record)}")
        self._buffer.extend(int(t) for t in self.packer.encode_document(record[self.text_field]))
        self._records_consumed += 1
        return True

    def _accumulate_one_shard(self) -> np.ndarray | None:
        """Fill one uniform ``shard_tokens`` payload from carry + stream, or ``None`` at stream end."""
        while len(self._buffer) < self.shard_tokens:
            if not self._pull_record():
                return None
        payload = np.asarray(self._buffer[: self.shard_tokens], dtype=self.packer.dtype)
        del self._buffer[: self.shard_tokens]
        return payload

    # ----- restart / fast-forward ----------------------------------------
    def _restore_or_init(self) -> None:
        # The on-disk manifest is authoritative for already-committed shards on restart.
        if self.store.manifest_path.exists():
            existing = load_manifest(self.store.manifest_path)
            if existing.dataset_fingerprint != self.manifest.dataset_fingerprint:
                raise ShardError(
                    f"existing manifest at {self.store.manifest_path} has a different dataset fingerprint; "
                    f"refusing to append a mismatched dataset. Use a fresh store_dir."
                )
            self.manifest = existing
        self._records = self.source_factory()
        ckpt = read_json(self.ckpt_path)
        if ckpt:
            self._records_consumed = int(ckpt.get("records_consumed", 0))
            self._buffer = [int(t) for t in ckpt.get("carry_tokens", [])]
            for _ in range(self._records_consumed):  # deterministic skip to the sealed boundary
                if next(self._records, None) is None:
                    break
            self._fast_forward("validation", int(ckpt.get("next_val_index", 0)))
            self._fast_forward("train", int(ckpt.get("next_train_index", 0)))

    def _fast_forward(self, split: str, ckpt_next_index: int) -> None:
        """Regenerate (and drop) shards the manifest committed after the last ckpt write.

        Covers the tiny crash window between ``publish_shard`` and the ckpt write: the shard is in
        the manifest but the ckpt is one behind. Replaying rebuilds the exact carry without ever
        re-publishing a committed shard.
        """
        committed = self.manifest.next_index(split)
        for _ in range(max(0, committed - ckpt_next_index)):
            if self._accumulate_one_shard() is None:  # pragma: no cover - stream too short to replay
                raise ShardError(f"cannot fast-forward split {split!r}: source shorter than the committed manifest")
        logger.info("fast-forwarded split %r to committed index %d", split, committed)

    # ----- production -----------------------------------------------------
    def _respect_queue_bound(self) -> None:
        while not self._stopped():
            consumed = int((read_json(self.consumer_path) or {}).get("consumed_train_shards", 0))
            ahead = self.manifest.next_index("train") - consumed
            if ahead < self.prefetch_shards:
                return
            self._set_status("running", paused=True, queue_depth=ahead)
            self._wait(self.poll_interval_s)

    def _produce_split(self, split: str, shards_target: int) -> None:
        while self.manifest.next_index(split) < shards_target and not self._stopped():
            if split == "train":
                self._respect_queue_bound()
                if self._stopped():
                    return
            payload = self._accumulate_one_shard()
            if payload is None:
                logger.warning(
                    "source exhausted before %s budget (%d/%d shards)",
                    split,
                    self.manifest.next_index(split),
                    shards_target,
                )
                break
            self.store.publish_shard(self.manifest, payload, split)
            self._write_ckpt()
            self._set_status("running", paused=False)
        if not self.manifest.split(split).complete and (
            self.manifest.next_index(split) >= shards_target or self._source_exhausted
        ):
            self.manifest.mark_complete(split)
            self.manifest.save_atomic(self.store.manifest_path)
            self._write_ckpt()

    def run(self) -> None:
        """Run to completion (or until ``stop_event``). Sets a ``failed`` status on any error."""
        self._set_status("starting")
        try:
            with self.store.lock():
                self.store.cleanup_partials()
                self._restore_or_init()
                self._set_status("running")
                self._produce_split("validation", self.budgets.val_shards)
                if self._stopped():
                    self._set_status("running", note="stopped before train completion")
                    return
                self._produce_split("train", self.budgets.train_shards)
                status = "complete" if self.manifest.split("train").complete else "running"
                self._set_status(status)
        except BaseException as exc:  # noqa: BLE001 - re-raised; we must record the failure first
            self._set_status("failed", error=str(exc), traceback=traceback.format_exc())
            logger.exception("shard producer failed")
            raise


def build_tokenizer_from_spec(tok_type: str, name: str, path: str | None):
    from ..tokenizers import HFTokenizer, TiktokenTokenizer

    tok_type = (tok_type or "tiktoken").lower()
    if tok_type in {"tiktoken", "gpt2"}:
        return TiktokenTokenizer(name or "gpt2")
    if tok_type in {"hf", "huggingface"}:
        return HFTokenizer(path or name)
    raise ValueError(f"producer cannot build tokenizer of type {tok_type!r}")


def run_producer_process(spec: dict, stop_event: threading.Event | None = None) -> None:
    """Subprocess entrypoint: reconstruct everything from ``spec`` and run the producer.

    ``spec`` is a plain dict (picklable across the spawn boundary). HF deps are imported lazily
    inside :func:`~.hf_source.open_hf_stream`.
    """
    store = ShardStore(spec["store_dir"], spec.get("manifest_path"))
    source = SourceSpec.from_dict(spec["source"])
    tokenizer = build_tokenizer_from_spec(spec["tokenizer_type"], spec["tokenizer_name"], spec.get("tokenizer_path"))
    transform = TransformSpec.from_dict(spec["transform"])
    packer = TokenPacker(tokenizer, add_eot=transform.add_eot, dtype_name=transform.dtype, packing=transform.packing)

    if store.manifest_path.exists():
        manifest = load_manifest(store.manifest_path)
    else:
        manifest = Manifest(
            dataset_id=spec["dataset_id"],
            source=source,
            tokenizer=TokenizerSpec(
                type=source.type and spec["tokenizer_type"],
                name=spec["tokenizer_name"],
                vocab_size=getattr(tokenizer, "vocab_size", None),
                fingerprint=tokenizer_fingerprint(tokenizer),
            ),
            transform=transform,
        )

    from .hf_source import open_hf_stream

    def source_factory():
        return iter(open_hf_stream(source))

    producer = ShardProducer(
        store,
        manifest,
        packer,
        source_factory,
        source.text_field,
        ProducerBudgets(val_shards=spec["val_shards"], train_shards=spec["train_shards"]),
        prefetch_shards=spec.get("prefetch_shards", 3),
        stop_event=stop_event,
    )
    producer.run()
