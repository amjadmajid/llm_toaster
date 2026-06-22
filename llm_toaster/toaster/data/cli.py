"""``scripts/data.py`` backend: prepare / inspect / validate / migrate-legacy.

Prepare reuses the same :class:`ShardProducer` the prefetch path uses (run in the foreground with
the queue bound disabled), so there is a single materialization implementation. It is append-safe,
refuses to clobber unmanifested legacy shards, and supports a dry-run plan.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import ConfigHandler
from ..tokenizers import build_tokenizer
from .document_streams import open_document_stream, resolve_source_revision
from .errors import ShardError
from .legacy import find_legacy_shards, migrate_legacy_directory
from .manifest import Manifest, SourceSpec, TokenizerSpec, TransformSpec, load_manifest, resolve_shard_path
from .packing import TokenPacker, dtype_for, tokenizer_fingerprint
from .producer import ProducerBudgets, ShardProducer
from .shard_store import ShardStore, read_shard
from .sources import (
    dataset_id_for,
    manifest_path_for,
    planned_train_shards,
    planned_val_shards,
    tokens_per_step,
)

logger = logging.getLogger(__name__)


def _store_dir_for(config) -> str:
    store = config.data.materialization.store_dir
    if store:
        return str(store)
    manifest_path = manifest_path_for(config)
    if manifest_path:
        return str(Path(manifest_path).parent)
    raise ShardError("set data.materialization.store_dir or data.manifest_path to prepare data")


def _source_spec(config, resolved_revision: str | None) -> SourceSpec:
    s = config.data.source
    return SourceSpec(
        type=s.type,
        dataset_name=s.dataset_name,
        config_name=s.config_name,
        requested_revision=s.revision,
        resolved_revision=resolved_revision,
        split=s.split,
        text_field=s.text_field,
    )


def plan_for(config, train_shards: int, val_shards: int) -> dict:
    shard_tokens = config.data.transform.shard_tokens
    itemsize = dtype_for(config.data.transform.dtype).itemsize
    tps = tokens_per_step(config)
    return {
        "dataset_id": dataset_id_for(config),
        "training_tokens": train_shards * shard_tokens,
        "validation_tokens": val_shards * shard_tokens,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "approx_storage_bytes": (train_shards + val_shards) * shard_tokens * itemsize,
        "tokens_per_step": tps,
        "max_steps": train_shards * shard_tokens // max(1, tps),
    }


def prepare_from_config(
    config, *, dry_run: bool = False, train_shards: int | None = None, val_shards: int | None = None
):
    """Materialize shards for ``config`` (foreground). Returns the manifest (or the plan on dry-run)."""
    tokenizer = build_tokenizer(config)
    if config.model.vocab_size is None:
        config.model.vocab_size = tokenizer.vocab_size
    train_shards = planned_train_shards(config) if train_shards is None else train_shards
    val_shards = planned_val_shards(config) if val_shards is None else val_shards

    plan = plan_for(config, train_shards, val_shards)
    for line in _format_plan(config, plan):
        logger.info(line)
    if dry_run:
        return plan

    store_dir = _store_dir_for(config)
    manifest_path = manifest_path_for(config) or str(Path(store_dir) / "manifest.json")
    store = ShardStore(store_dir, manifest_path)
    if not Path(manifest_path).exists() and find_legacy_shards(store_dir):
        raise ShardError(
            f"{store_dir} contains unmanifested legacy shards. Migrate them once instead of re-tokenizing:\n"
            f"  python scripts/data.py migrate-legacy --data-dir {store_dir} --manifest {manifest_path}"
        )

    resolved = config.data.source.revision or resolve_source_revision(_source_spec(config, None))
    source = _source_spec(config, resolved)
    transform = TransformSpec(
        add_eot=config.data.transform.add_eot,
        packing=config.data.transform.packing,
        shard_tokens=config.data.transform.shard_tokens,
        dtype=config.data.transform.dtype,
    )
    manifest = Manifest(
        dataset_id=dataset_id_for(config),
        source=source,
        tokenizer=TokenizerSpec(
            type=config.tokenizer.type,
            name=config.tokenizer.name,
            vocab_size=getattr(tokenizer, "vocab_size", None),
            fingerprint=tokenizer_fingerprint(tokenizer),
        ),
        transform=transform,
    )
    packer = TokenPacker(tokenizer, add_eot=transform.add_eot, dtype_name=transform.dtype, packing=transform.packing)
    producer = ShardProducer(
        store,
        manifest,
        packer,
        lambda: open_document_stream(source),
        source.text_field,
        ProducerBudgets(val_shards=val_shards, train_shards=train_shards),
        prefetch_shards=train_shards + 1,  # foreground prepare: never pause on the queue bound
    )
    producer.run()
    logger.info("prepared %s (generation %d)", manifest_path, load_manifest(manifest_path).generation)
    return load_manifest(manifest_path)


def cmd_prepare(args) -> int:
    config = ConfigHandler.from_yaml(args.config)
    prepare_from_config(config, dry_run=args.dry_run)
    return 0


def cmd_inspect(args) -> int:
    manifest = load_manifest(args.manifest)
    print(f"dataset_id:          {manifest.dataset_id}")
    print(f"dataset_fingerprint: {manifest.dataset_fingerprint}")
    print(f"generation:          {manifest.generation}")
    print(f"format_version:      {manifest.format_version}")
    src = manifest.source
    print(
        f"source:              {src.type} {src.dataset_name}@{src.resolved_revision} "
        f"config={src.config_name} split={src.split}"
    )
    tok = manifest.tokenizer
    print(f"tokenizer:           {tok.type}/{tok.name} vocab={tok.vocab_size}")
    tf = manifest.transform
    print(
        f"transform:           add_eot={tf.add_eot} packing={tf.packing} dtype={tf.dtype} "
        f"shard_tokens={tf.shard_tokens}"
    )
    for name, state in manifest.splits.items():
        ids = [s.id for s in state.shards]
        span = f"{ids[0]}..{ids[-1]}" if ids else "(none)"
        print(f"split {name:>10}: complete={state.complete} tokens={state.tokens:,} shards={len(ids)} [{span}]")
    return 0


def cmd_validate(args) -> int:
    manifest = load_manifest(args.manifest)  # structural validation
    problems = 0
    for name, state in manifest.splits.items():
        for entry in state.shards:
            path = resolve_shard_path(args.manifest, entry)
            try:
                read_shard(path, mmap=False, expected_sha256=entry.sha256)
            except Exception as exc:  # noqa: BLE001 - report and continue across all shards
                problems += 1
                print(f"FAIL {name} {entry.id}: {exc}")
    if problems:
        print(f"validation FAILED: {problems} problem(s)")
        return 1
    print(f"validation OK: {sum(len(s.shards) for s in manifest.splits.values())} shard(s) verified")
    return 0


def cmd_migrate(args) -> int:
    tokenizer = None
    if args.config:
        tokenizer = build_tokenizer(ConfigHandler.from_yaml(args.config))
    migrate_legacy_directory(args.data_dir, args.manifest, tokenizer=tokenizer, overwrite=args.overwrite)
    print(f"migrated {args.data_dir} -> {args.manifest}")
    return 0


def _format_plan(config, plan: dict) -> list[str]:
    mb = plan["approx_storage_bytes"] / (1024**2)
    return [
        f"plan: dataset={plan['dataset_id']} | mode={config.data.materialization.mode}",
        f"  training tokens:   {plan['training_tokens']:,} ({plan['train_shards']} shards)",
        f"  validation tokens: {plan['validation_tokens']:,} ({plan['val_shards']} shards)",
        f"  approx storage:    {mb:,.1f} MB ({config.data.transform.dtype})",
        f"  optimizer steps:   up to {plan['max_steps']:,} ({plan['tokens_per_step']:,} tokens/step)",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data.py", description="LLM Toaster manifest-backed data tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Materialize token shards described by a config.")
    p_prepare.add_argument("--config", required=True, help="Path to a training YAML config.")
    p_prepare.add_argument("--dry-run", action="store_true", help="Print the plan and exit (no shards written).")
    p_prepare.set_defaults(func=cmd_prepare)

    p_inspect = sub.add_parser("inspect", help="Print a manifest summary.")
    p_inspect.add_argument("--manifest", required=True)
    p_inspect.set_defaults(func=cmd_inspect)

    p_validate = sub.add_parser("validate", help="Verify every shard exists and matches its checksum.")
    p_validate.add_argument("--manifest", required=True)
    p_validate.set_defaults(func=cmd_validate)

    p_migrate = sub.add_parser("migrate-legacy", help="Register an existing shard directory into a manifest.")
    p_migrate.add_argument("--data-dir", required=True)
    p_migrate.add_argument("--manifest", required=True)
    p_migrate.add_argument("--config", default=None, help="Optional config to record tokenizer identity.")
    p_migrate.add_argument("--overwrite", action="store_true", help="Replace an existing manifest.")
    p_migrate.set_defaults(func=cmd_migrate)
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)
