"""Run inference on a trained LLM Toaster model.

Uses the same model registry and tokenizer as training (via the modular engine)
so a model produced by ``extract_inference_model.py`` loads and tokenizes
identically here.
"""

import argparse
import logging

import torch

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.generation import generate
from llm_toaster.toaster.models.registry import build_model
from llm_toaster.toaster.tokenizers import build_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Run inference on a Transformer model.")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt for the model")
    parser.add_argument("--config", default="model/babyGPT/babyGPT_base.yaml", help="Model config path")
    parser.add_argument("--model", default="model/babyGPT/babyGPT_base.llm", help="Model state_dict path")
    parser.add_argument("--max-new-tokens", type=int, help="Override generated token count")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=35, help="Top-k sampling cutoff; use 0 to disable")
    parser.add_argument("--top-p", type=float, help="Nucleus sampling cutoff")
    args = parser.parse_args()

    config = ConfigHandler.from_yaml(args.config)
    device = _default_device()
    config.training.device = device

    tokenizer = build_tokenizer(config)
    if config.model.vocab_size is None:
        config.model.vocab_size = tokenizer.vocab_size
    model = build_model(config).to(device)
    # Inference artifacts are plain state_dicts, so the safe loader path applies.
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))

    text = generate(
        model,
        tokenizer,
        args.prompt,
        device,
        max_new_tokens=args.max_new_tokens or config.inference.generate_max_length,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()
