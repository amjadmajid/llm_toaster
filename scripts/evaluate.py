#!/usr/bin/env python
"""Evaluate a checkpoint with the modular engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.training.engine import TrainingEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an LLM Toaster checkpoint.")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = ConfigHandler.from_yaml(args.config)
    engine = TrainingEngine(config)
    engine.setup_tokenizer()
    engine.setup_model()
    engine.setup_dataloaders()
    engine.setup_optimizer()
    engine.setup_scheduler()
    checkpoint = args.checkpoint or engine.default_checkpoint_path
    if Path(checkpoint).exists():
        engine.load_checkpoint(checkpoint)
    metric = engine.eval_step()
    if metric is None:
        print("No validation loader is available for this config.")
    else:
        print(f"validation_loss={metric:.6f}")


if __name__ == "__main__":
    main()
