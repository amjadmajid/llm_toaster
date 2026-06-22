"""Build trainer-facing :class:`PretrainBatchSource`s from a :class:`ConfigHandler`.

This is the single place that maps ``data.materialization.mode`` to a concrete source:

- ``prepared``  -> :class:`ManifestShardSource` over an existing manifest (stop/repeat).
- ``prefetch``  -> start a :class:`PrefetchCoordinator`, wait for readiness, then a wait-mode
  :class:`ManifestShardSource` that discovers appended shards via manifest refresh.
- ``direct``    -> experimental :class:`HFDirectTokenSource` (validation still from a manifest).
- legacy config -> deprecated :class:`LegacyShardDirSource` over a shard *directory*.

The engine owns the returned coordinator's lifecycle (it calls ``stop()`` in its ``finally``).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from .coordinator import PrefetchCoordinator
from .document_streams import _env_start_method, resolve_source_revision
from .errors import DataError, ShardError
from .legacy import LegacyShardDirSource, find_legacy_shards
from .manifest import SourceSpec, load_manifest
from .packing import TokenPacker
from .shard_source import ManifestShardSource

logger = logging.getLogger(__name__)


def manifest_path_for(config) -> str | None:
    data = config.data
    if data.manifest_path:
        return str(data.manifest_path)
    store = data.materialization.store_dir
    if store:
        return str(Path(store) / "manifest.json")
    return None


def _legacy_data_dir(config) -> str | None:
    return config.training.data_dir or config.data.materialization.store_dir or config.data.tokenized_data


def _is_legacy(config) -> bool:
    return getattr(config, "_data_is_legacy", False)


def _to_source_spec(source, resolved_revision: str | None) -> SourceSpec:
    return SourceSpec(
        type=source.type,
        dataset_name=source.dataset_name,
        config_name=source.config_name,
        requested_revision=source.revision,
        resolved_revision=resolved_revision,
        split=source.split,
        text_field=source.text_field,
    )


def dataset_id_for(config) -> str:
    source = config.data.source
    base = (source.dataset_name or "dataset").rstrip("/").split("/")[-1]
    parts = [base]
    if source.config_name:
        parts.append(str(source.config_name))
    parts.append(config.tokenizer.name or config.tokenizer.type)
    return "-".join(parts)


def tokens_per_step(config) -> int:
    return config.training.batch_size * config.model.seq_len * config.training.n_batches


def planned_train_shards(config) -> int:
    shard_tokens = max(1, config.data.transform.shard_tokens)
    if config.training.max_tokens is not None:
        target = config.training.max_tokens
    else:
        target = config.training.max_iter * tokens_per_step(config)
    return max(config.data.materialization.min_ready_shards, math.ceil(target / shard_tokens))


def planned_val_shards(config) -> int:
    validation = config.data.validation
    shard_tokens = max(1, config.data.transform.shard_tokens)
    if validation.shards is not None:
        return int(validation.shards)
    if validation.tokens is not None:
        return max(1, math.ceil(validation.tokens / shard_tokens))
    return 0


def make_producer_spec(config, resolved_revision: str) -> dict:
    data = config.data
    store_dir = data.materialization.store_dir or str(Path(manifest_path_for(config)).parent)
    return {
        "store_dir": store_dir,
        "manifest_path": manifest_path_for(config),
        "dataset_id": dataset_id_for(config),
        "source": _to_source_spec(data.source, resolved_revision).to_dict(),
        "tokenizer_type": config.tokenizer.type,
        "tokenizer_name": config.tokenizer.name,
        "tokenizer_path": config.tokenizer.path,
        "transform": {
            "add_eot": data.transform.add_eot,
            "packing": data.transform.packing,
            "shard_tokens": data.transform.shard_tokens,
            "dtype": data.transform.dtype,
        },
        "val_shards": planned_val_shards(config),
        "train_shards": planned_train_shards(config),
        "prefetch_shards": data.materialization.prefetch_shards,
    }


def build_pretrain_train_source(config, tokenizer):
    """Return ``(source, coordinator_or_None)`` for the configured pretraining mode."""
    data = config.data
    B, T = config.training.batch_size, config.model.seq_len
    mode = data.materialization.mode

    if mode == "direct":
        return _build_direct_source(config, tokenizer), None

    manifest_path = manifest_path_for(config)
    if _is_legacy(config) and (manifest_path is None or not Path(manifest_path).exists()):
        data_dir = _legacy_data_dir(config)
        logger.warning(
            "Using the deprecated legacy shard directory %s (no manifest). Migrate with "
            "`python scripts/data.py migrate-legacy`. See docs/data-pipeline.md.",
            data_dir,
        )
        return LegacyShardDirSource(data_dir, "train", B, T), None

    coordinator = None
    if mode == "prefetch":
        coordinator = _start_prefetch(config)

    manifest_path = manifest_path_for(config)
    if manifest_path is None or not Path(manifest_path).exists():
        _raise_missing_manifest(config, manifest_path)

    exhaustion = "wait" if mode == "prefetch" else data.sampling.exhaustion
    source = ManifestShardSource(
        manifest_path,
        "train",
        B,
        T,
        exhaustion=exhaustion,
        shuffle=data.sampling.shuffle,
        seed=data.sampling.seed,
        wait_timeout_s=data.materialization.wait_timeout_s,
        status_provider=coordinator.status if coordinator else None,
    )
    return source, coordinator


def build_validation_source(config):
    """Fixed, materialized validation source (never grows during training); ``None`` if absent."""
    if config.evaluation.eval_every_steps <= 0:
        return None
    data = config.data
    B, T = config.training.batch_size, config.model.seq_len

    manifest_path = manifest_path_for(config)
    if _is_legacy(config) and (manifest_path is None or not Path(manifest_path).exists()):
        try:
            return LegacyShardDirSource(_legacy_data_dir(config), "val", B, T)
        except FileNotFoundError:
            return None

    val_manifest = data.validation.manifest_path or manifest_path
    if not val_manifest or not Path(val_manifest).exists():
        return None
    try:
        manifest = load_manifest(val_manifest)
    except DataError as exc:
        logger.info("validation manifest unavailable; continuing without validation: %s", exc)
        return None
    if not manifest.split("validation").shards:
        return None
    return ManifestShardSource(
        val_manifest, "validation", B, T, exhaustion="stop", shuffle="none", seed=data.sampling.seed
    )


def _build_direct_source(config, tokenizer):
    from .hf_source import HFDirectTokenSource, open_hf_stream

    data = config.data
    resolved = data.source.revision or resolve_source_revision(_to_source_spec(data.source, None))
    source_spec = _to_source_spec(data.source, resolved)
    packer = TokenPacker(
        tokenizer,
        add_eot=data.transform.add_eot,
        dtype_name=data.transform.dtype,
        packing=data.transform.packing,
    )
    iterable = open_hf_stream(source_spec, need_state_api=True)
    return HFDirectTokenSource(
        iterable,
        packer,
        data.source.text_field,
        config.training.batch_size,
        config.model.seq_len,
        exhaustion=data.sampling.exhaustion,
        resolved_revision=resolved,
        dataset_id=dataset_id_for(config),
    )


def _start_prefetch(config) -> PrefetchCoordinator:
    resolved = config.data.source.revision or resolve_source_revision(_to_source_spec(config.data.source, None))
    spec = make_producer_spec(config, resolved)
    coordinator = PrefetchCoordinator(spec, start_method=_env_start_method())
    coordinator.start()
    coordinator.wait_until_ready(
        min_ready_shards=config.data.materialization.min_ready_shards,
        timeout_s=config.data.materialization.wait_timeout_s,
    )
    return coordinator


def _raise_missing_manifest(config, manifest_path: str | None) -> None:
    store = config.data.materialization.store_dir or _legacy_data_dir(config)
    if store and find_legacy_shards(store):
        raise ShardError(
            f"found unmanifested legacy shards in {store} but no manifest at {manifest_path}. "
            f"Migrate them once: `python scripts/data.py migrate-legacy --data-dir {store} "
            f"--manifest {manifest_path}`."
        )
    raise ShardError(
        f"no manifest at {manifest_path}. Materialize data first: "
        f"`python scripts/data.py prepare --config <your_config.yaml>` (prepared mode), "
        f"or use materialization.mode=prefetch to produce shards during training."
    )
