"""Hugging Face integration: lazy helpers + the experimental ``HFDirectTokenSource``.

``datasets`` is an *optional* dependency, imported lazily so the core training path stays
light. Direct mode tokenizes and packs HF records in the training process with no persistent
train shards -- explicitly experimental and constrained (single process, ``num_workers=0``, no
buffered shuffle) until exact resume is demonstrated. It reuses the same :class:`TokenPacker`
and :func:`make_batch` as shard materialization, so its token stream matches the prepared path.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from .errors import DataExhausted, HFDependencyError
from .packing import TokenPacker, make_batch, tokenizer_fingerprint
from .protocol import PretrainBatchInfo

logger = logging.getLogger(__name__)

SOURCE_MODE = "direct"
# IterableDataset.state_dict()/load_state_dict() (resumable streaming) landed in datasets 2.19.0.
MIN_DATASETS_VERSION_FOR_STATE = (2, 19, 0)


def require_datasets(need_state_api: bool = False):
    """Import ``datasets`` lazily with an actionable error; optionally require the state API."""
    try:
        import datasets  # noqa: F401
    except ModuleNotFoundError as exc:
        raise HFDependencyError(
            "Hugging Face 'datasets' is required for source streaming. Install the data extra:\n"
            "    pip install -e '.[data]'"
        ) from exc
    if need_state_api:
        version = getattr(datasets, "__version__", "0")
        parts = tuple(int(p) for p in version.split(".")[:3] if p.isdigit())
        if parts < MIN_DATASETS_VERSION_FOR_STATE:
            need = ".".join(str(p) for p in MIN_DATASETS_VERSION_FOR_STATE)
            raise HFDependencyError(
                f"datasets {version} does not support resumable IterableDataset state "
                f"(need >= {need}). Upgrade: pip install -U 'datasets>={need}'."
            )
    return datasets


def resolve_revision(dataset_name: str, requested_revision: str | None) -> str:
    """Resolve a (possibly ``None``/branch) revision to an immutable commit SHA.

    Used by ``prepare``/``prefetch`` so shards record exactly which dataset snapshot they came
    from. Raises an actionable error if the Hub is unreachable.
    """
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:  # pragma: no cover - shipped with datasets
        raise HFDependencyError("huggingface_hub is required to resolve a dataset revision.") from exc
    try:
        info = HfApi().dataset_info(dataset_name, revision=requested_revision)
    except Exception as exc:
        raise HFDependencyError(
            f"could not resolve revision for dataset {dataset_name!r} "
            f"(requested {requested_revision!r}): {exc}. Check the name and network access."
        ) from exc
    sha = getattr(info, "sha", None)
    if not sha:  # pragma: no cover - HfApi always returns a sha on success
        raise HFDependencyError(f"Hub returned no commit SHA for {dataset_name!r}")
    return sha


def open_hf_stream(source, *, need_state_api: bool = True):
    """Open a streaming, revision-pinned :class:`datasets.IterableDataset` for ``source``.

    ``source`` is a :class:`~.manifest.SourceSpec`-like object. Returns the iterable; the caller
    iterates records and reads ``record[source.text_field]``.
    """
    datasets = require_datasets(need_state_api=need_state_api)
    revision = source.resolved_revision or source.requested_revision
    iterable = datasets.load_dataset(
        source.dataset_name,
        name=source.config_name,
        split=source.split,
        streaming=True,
        revision=revision,
    )
    return iterable


class HFDirectTokenSource:
    """Experimental in-process HF token stream. Implements ``PretrainBatchSource``.

    Resume restores the pending token buffer, records consumed, and (where the installed
    ``datasets`` supports it) the iterable's own state. Bitwise-exact resume is *not* claimed.
    """

    def __init__(
        self,
        iterable,
        packer: TokenPacker,
        text_field: str,
        batch_size: int,
        seq_len: int,
        *,
        exhaustion: str = "stop",
        resolved_revision: str | None = None,
        dataset_id: str = "hf-direct",
        require_state_api: bool = True,
        warn: bool = True,
    ):
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        if exhaustion == "wait":
            raise ValueError("direct mode does not support exhaustion='wait' (no shard producer)")
        if resolved_revision is None:
            raise ValueError(
                "direct mode requires an immutable resolved source revision; resolve it first "
                "(scripts/data.py records one) so the stream is reproducible."
            )
        self._has_state_api = hasattr(iterable, "state_dict") and hasattr(iterable, "load_state_dict")
        if require_state_api and not self._has_state_api:
            raise HFDependencyError(
                "the provided HF iterable lacks state_dict()/load_state_dict(); resumable direct "
                f"streaming needs datasets >= {'.'.join(map(str, MIN_DATASETS_VERSION_FOR_STATE))}."
            )
        if warn:
            message = (
                "direct mode is EXPERIMENTAL: weaker reproducibility than manifest-backed "
                "prepared/prefetch modes. Validation must use a fixed materialized manifest."
            )
            warnings.warn(message, stacklevel=2)
            logger.warning(message)

        self.dataset = iterable
        self.packer = packer
        self.text_field = text_field
        self.B = batch_size
        self.T = seq_len
        self.exhaustion = exhaustion
        self.resolved_revision = resolved_revision
        self.dataset_id = dataset_id
        self._iter = iter(self.dataset)
        self._buffer: list[int] = []
        self.records_consumed = 0
        self.emitted_tokens = 0
        self.pass_index = 0
        self.repeated = False

    def _pull_record(self) -> bool:
        """Append one document's tokens to the buffer. Returns ``False`` at stream end."""
        try:
            record = next(self._iter)
        except StopIteration:
            return False
        if self.text_field not in record:
            raise KeyError(f"record is missing text_field {self.text_field!r}; keys: {sorted(record)}")
        self._buffer.extend(int(t) for t in self.packer.encode_document(record[self.text_field]))
        self.records_consumed += 1
        return True

    def _refill_or_exhaust(self, need: int) -> None:
        while len(self._buffer) < need:
            if self._pull_record():
                continue
            if self.exhaustion == "repeat":
                self.pass_index += 1
                self.repeated = True
                logger.info("direct stream wrapped to pass %d (repeat)", self.pass_index)
                self._iter = iter(self.dataset)
                if not self._pull_record():
                    raise DataExhausted("direct stream is empty", pass_index=self.pass_index)
            else:
                raise DataExhausted("direct HF stream exhausted", pass_index=self.pass_index)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray, PretrainBatchInfo]:
        need = self.B * self.T + 1
        self._refill_or_exhaust(need)
        window = np.asarray(self._buffer[:need])
        x, y = make_batch(window, 0, self.B, self.T)
        del self._buffer[: self.B * self.T]  # keep the boundary token (contiguous overlap)
        self.emitted_tokens += self.B * self.T
        info = PretrainBatchInfo(
            shard_id=None,
            token_offset=self.emitted_tokens,
            pass_index=self.pass_index,
            repeated=self.repeated,
            source_mode=SOURCE_MODE,
        )
        return np.asarray(x), np.asarray(y), info

    def reset(self) -> None:
        self._iter = iter(self.dataset)
        self._buffer = []
        self.records_consumed = 0
        self.emitted_tokens = 0
        self.pass_index = 0
        self.repeated = False

    def close(self) -> None:
        self._buffer = []

    def state_dict(self) -> dict:
        state = {
            "source_mode": SOURCE_MODE,
            "dataset_id": self.dataset_id,
            "resolved_revision": self.resolved_revision,
            "tokenizer_fingerprint": tokenizer_fingerprint(self.packer.tokenizer),
            "transform_fingerprint": None,
            "pending_buffer": list(self._buffer),
            "records_consumed": self.records_consumed,
            "emitted_tokens": self.emitted_tokens,
            "pass_index": self.pass_index,
            "repeated": self.repeated,
        }
        if self._has_state_api:
            try:
                state["iterable_state"] = self.dataset.state_dict()
            except Exception as exc:  # pragma: no cover - advisory
                logger.warning("could not capture HF iterable state: %s", exc)
        return state

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        saved_rev = state.get("resolved_revision")
        if saved_rev is not None and saved_rev != self.resolved_revision:
            from .errors import ResumeIncompatibleError

            raise ResumeIncompatibleError(
                f"direct-mode source revision changed: checkpoint {saved_rev!r} vs {self.resolved_revision!r}"
            )
        saved_tok = state.get("tokenizer_fingerprint")
        if saved_tok is not None and saved_tok != tokenizer_fingerprint(self.packer.tokenizer):
            from .errors import ResumeIncompatibleError

            raise ResumeIncompatibleError("direct-mode tokenizer fingerprint changed since the checkpoint")
        self._buffer = [int(t) for t in state.get("pending_buffer", [])]
        self.records_consumed = int(state.get("records_consumed", 0))
        self.emitted_tokens = int(state.get("emitted_tokens", 0))
        self.pass_index = int(state.get("pass_index", 0))
        self.repeated = bool(state.get("repeated", self.pass_index > 0))
        iterable_state = state.get("iterable_state")
        if iterable_state is not None and self._has_state_api:
            self.dataset.load_state_dict(iterable_state)
        self._iter = iter(self.dataset)

    def stats(self) -> dict:
        return {
            "source_mode": SOURCE_MODE,
            "dataset_id": self.dataset_id,
            "data_pass": self.pass_index,
            "source_records_consumed": self.records_consumed,
        }
