import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from config import ConfigHandler
from model import TransformerModel
from tokenizer_lib import gpt2_decode, gpt2_encode, init_gpt2_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inference(config, model, prompt, *, max_new_tokens=None, temperature=1.0, top_k=35, top_p=None):
    model.eval()
    init_gpt2_tokenizer()
    input_ids = gpt2_encode(prompt, dtype=np.int32)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(config.training.device)
    if input_ids.size(1) > config.training.seq_len:
        input_ids = input_ids[:, -config.training.seq_len:]
        print(f"Input truncated to the last {config.training.seq_len} tokens.")
    with torch.no_grad():
        output = model.generate_text(
            input_ids,
            max_length=max_new_tokens or config.inference.generate_max_length,
            topk=top_k,
            temperature=temperature,
            top_p=top_p,
        )
    print(gpt2_decode(output[0].cpu().tolist(), require_eot=False))


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
    config.training.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = TransformerModel(
        n_head=config.training.n_head,
        vocab_size=config.training.vocab_size or 50304,
        n_embd=config.training.n_embd,
        seq_len=config.training.seq_len,
        device=config.training.device,
        dropout_rate=config.training.dropout_rate,
        n_blocks=config.training.n_blocks,
        decoder=True,
    ).to(config.training.device)
    model.load_state_dict(torch.load(Path(args.model), map_location=config.training.device))
    inference(
        config,
        model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
