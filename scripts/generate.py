#!/usr/bin/env python
"""Generate text from a checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.generation import generate
from llm_toaster.toaster.training.engine import TrainingEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with an LLM Toaster checkpoint.")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=35)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    config = ConfigHandler.from_yaml(args.config)
    engine = TrainingEngine(config)
    tokenizer = engine.setup_tokenizer()
    model = engine.setup_model()
    checkpoint = args.checkpoint or engine.default_checkpoint_path
    if Path(checkpoint).exists():
        engine.load_checkpoint(checkpoint)
    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            engine.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
        )
    )


if __name__ == "__main__":
    main()
