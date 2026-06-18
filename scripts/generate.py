#!/usr/bin/env python
"""Generate text from a checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.training.engine import TrainingEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with an LLM Toaster checkpoint.")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=35)
    args = parser.parse_args()

    config = ConfigHandler.from_yaml(args.config)
    engine = TrainingEngine(config)
    tokenizer = engine.setup_tokenizer()
    model = engine.setup_model()
    checkpoint = args.checkpoint or engine.default_checkpoint_path
    if Path(checkpoint).exists():
        engine.load_checkpoint(checkpoint)
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=engine.device)
    out = model.generate_text(
        ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(out[0].detach().cpu().tolist()))


if __name__ == "__main__":
    main()
