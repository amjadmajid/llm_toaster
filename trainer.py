"""Compatibility wrapper around the modular TrainingEngine."""

from __future__ import annotations

import argparse
import logging

import torch

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.training.engine import TrainingEngine

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Toaster training wrapper")
    parser.add_argument("-ct", "--continue-training", action="store_true", help="Resume from training.ckpt")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to YAML config")
    parser.add_argument("--mode", choices=["pretrain", "finetune", "sft"], help="Override training.mode")
    args = parser.parse_args()

    config = ConfigHandler.from_yaml(args.config)
    if args.mode:
        config.training.mode = "finetune" if args.mode == "sft" else args.mode
        config.finetune.enabled = config.training.mode == "finetune"
    if args.continue_training:
        config.checkpointing.resume_from_checkpoint = config.training.ckpt
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    TrainingEngine(config).train()


if __name__ == "__main__":
    main()
