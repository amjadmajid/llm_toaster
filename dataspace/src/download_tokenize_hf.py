"""DEPRECATED compatibility wrapper around the manifest-backed data pipeline.

Historically this script tokenized a Hugging Face dataset into loose ``shard_XXXXXX_train.npy``
files and carved a validation split by renaming. That layout had no manifest, overwrote files on
re-run, and produced misleading validation proportions on small shard counts.

It now delegates to the canonical pipeline (``llm_toaster.toaster.data.cli.prepare_from_config``),
which writes immutable, manifest-described shards atomically and is append-safe. Prefer the new CLI:

    python scripts/data.py prepare --config config/default_config.yaml

``--stream`` here means "stream the source while materializing shards" (the pipeline always streams
the source); it never meant trainer-time streaming. For that, use ``materialization.mode: prefetch``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.data.cli import prepare_from_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_config(dataset_name, remote_name, output_dir, shard_size) -> ConfigHandler:
    config = ConfigHandler()
    config.data.source.type = "huggingface"
    config.data.source.dataset_name = dataset_name
    config.data.source.config_name = remote_name
    config.data.source.split = "train"
    config.data.transform.shard_tokens = int(shard_size)
    config.data.transform.dtype = "uint16"
    config.data.materialization.store_dir = str(output_dir)
    config.data.manifest_path = str(Path(output_dir) / "manifest.json")
    return config


def _budgets(split_ratio: float, max_shards: int | None) -> tuple[int, int]:
    """Translate the legacy split_ratio/max_shards into explicit (train_shards, val_shards)."""
    if max_shards is not None:
        val_shards = max(1, round(max_shards * (1.0 - split_ratio)))
        train_shards = max(1, max_shards - val_shards)
        return train_shards, val_shards
    # Whole-split materialization: one validation shard, train until the source ends.
    return 1_000_000_000, 1


def download_and_tokenize(
    dataset_name,
    remote_name,
    split_ratio,
    output_dir,
    shard_size=int(1e8),
    stream=False,
    max_shards=None,
):
    """Compatibility entry point. Materializes manifest-backed shards; returns True on success."""
    logger.warning(
        "download_tokenize_hf is deprecated; use `python scripts/data.py prepare --config <cfg>`. "
        "Delegating to the manifest-backed pipeline (append-safe, atomic)."
    )
    if not stream:
        logger.info("the new pipeline always streams the source; '--stream' is now the only behavior.")
    config = _build_config(dataset_name, remote_name, output_dir, shard_size)
    train_shards, val_shards = _budgets(split_ratio, max_shards)
    try:
        prepare_from_config(config, train_shards=train_shards, val_shards=val_shards)
    except Exception as exc:  # noqa: BLE001 - surface a clean True/False to legacy callers
        logger.error("preparation failed: %s", exc)
        return False
    return True


def _resolve_output_dir(tokenized_data: str) -> Path:
    """Resolve a relative output dir against the repo root, not the CWD."""
    output_dir = Path(tokenized_data)
    if output_dir.is_absolute():
        return output_dir
    return Path(__file__).resolve().parents[2] / output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DEPRECATED: tokenize an HF dataset into manifest-backed shards "
        "(use scripts/data.py prepare instead)."
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the source while materializing shards (the pipeline always streams; kept for compatibility).",
    )
    parser.add_argument("--max-shards", type=int, default=None, metavar="N", help="Stop after N total shards.")
    parser.add_argument("--output-dir", type=str, default=None, metavar="PATH", help="Where to write the shard store.")
    parser.add_argument("--shard-size", type=int, default=None, metavar="N", help="Tokens per shard.")
    args = parser.parse_args()
    if args.max_shards is not None and args.max_shards < 1:
        parser.error("--max-shards must be >= 1")
    if args.shard_size is not None and args.shard_size < 1:
        parser.error("--shard-size must be >= 1")

    defaults = ConfigHandler().data
    requested_out = args.output_dir if args.output_dir is not None else defaults.tokenized_data
    output_dir = str(_resolve_output_dir(requested_out))
    shard_size = args.shard_size if args.shard_size is not None else defaults.shard_size

    ok = download_and_tokenize(
        dataset_name=defaults.dataset_name,
        remote_name=defaults.remote_name,
        split_ratio=defaults.split_ratio,
        output_dir=output_dir,
        shard_size=shard_size,
        stream=args.stream,
        max_shards=args.max_shards,
    )
    sys.exit(0 if ok else 1)
