"""Extract an inference-only state_dict from a training checkpoint.

Loads a full engine checkpoint (model/optimizer/scheduler/scaler/config) and
writes just the model weights plus its config, ready for ``inference.py``.
"""

import argparse
import logging
from pathlib import Path

import torch

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.models.registry import build_model
from llm_toaster.toaster.tokenizers import build_tokenizer
from llm_toaster.toaster.training.checkpointing import load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract an inference model from a checkpoint")
    parser.add_argument("--config", default="checkpoints/base_config.yaml", help="Checkpoint config to load")
    parser.add_argument("--checkpoint", help="Checkpoint path. Defaults to training.ckpt from --config")
    parser.add_argument("--output", default="model/babyGPT/babyGPT_base.llm", help="Output state_dict path")
    parser.add_argument("--output-config", default="model/babyGPT/babyGPT_base.yaml", help="Output config path")
    args = parser.parse_args()

    config = ConfigHandler.from_yaml(args.config)
    device = _default_device()
    config.training.device = device
    if config.model.vocab_size is None:
        config.model.vocab_size = build_tokenizer(config).vocab_size

    model = build_model(config).to(device)
    checkpoint_path = args.checkpoint or config.training.ckpt
    # strict=False tolerates a `_orig_mod.` prefix from torch.compile or LoRA params.
    load_checkpoint(checkpoint_path, model, device=device, strict=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    config.to_yaml(args.output_config)
    logger.info("Saved inference model to %s and config to %s", output, args.output_config)


if __name__ == "__main__":
    main()
