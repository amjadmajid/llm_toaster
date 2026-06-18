"""Extract an inference-only state_dict from a training checkpoint."""

import argparse
import logging
from pathlib import Path

import torch

from config import ConfigHandler
from model import TransformerModel
from utils import load_checkpoint_

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    config.training.device = device
    return TransformerModel(
        n_head=config.training.n_head,
        vocab_size=config.training.vocab_size or 50304,
        n_embd=config.training.n_embd,
        seq_len=config.training.seq_len,
        device=device,
        dropout_rate=config.training.dropout_rate,
        n_blocks=config.training.n_blocks,
        decoder=True,
    ).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract an inference model from a checkpoint")
    parser.add_argument("--config", default="checkpoints/base_config.yaml", help="Checkpoint config to load")
    parser.add_argument("--checkpoint", help="Checkpoint path. Defaults to training.ckpt from --config")
    parser.add_argument("--output", default="model/babyGPT/babyGPT_base.llm", help="Output state_dict path")
    parser.add_argument("--output-config", default="model/babyGPT/babyGPT_base.yaml", help="Output config path")
    args = parser.parse_args()

    config = ConfigHandler.from_yaml(args.config)
    model = build_model(config)
    checkpoint = args.checkpoint or config.training.ckpt
    load_checkpoint_(model, None, None, checkpoint, config.training.device, inference=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    config.to_yaml(args.output_config)
    logger.info("Saved inference model to %s and config to %s", output, args.output_config)
