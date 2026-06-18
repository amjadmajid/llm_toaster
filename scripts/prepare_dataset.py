#!/usr/bin/env python
"""Prepare simple text/JSONL data into tokenized .npy train/val shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.tokenizers import build_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize a text or JSONL dataset into LLM Toaster shards.")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--input", required=True, help="Input .txt or .jsonl file. JSONL rows may contain a text field.")
    parser.add_argument("--output-dir", default=None, help="Output shard directory; defaults to training.data_dir.")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    config = ConfigHandler.from_yaml(args.config)
    tokenizer = build_tokenizer(config)
    texts = list(_iter_texts(Path(args.input)))
    token_ids = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text))
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            token_ids.append(int(eos_id))
    if len(token_ids) < 2:
        raise ValueError("Need at least two tokens to create a language-modeling shard")

    output_dir = Path(args.output_dir or config.training.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = max(1, int(len(token_ids) * (1.0 - args.val_ratio)))
    train = np.asarray(token_ids[:split], dtype=np.int32)
    val = np.asarray(token_ids[split:] or token_ids[-min(len(token_ids), 128) :], dtype=np.int32)
    np.save(output_dir / "train_000.npy", train)
    np.save(output_dir / "val_000.npy", val)
    print(f"Wrote {len(train)} train tokens and {len(val)} val tokens to {output_dir}")


def _iter_texts(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield row.get("text") or row.get("prompt") or row.get("instruction") or json.dumps(row)
    else:
        yield path.read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
