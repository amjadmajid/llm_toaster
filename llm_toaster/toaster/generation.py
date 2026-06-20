"""Single text-generation path shared by inference.py and scripts/generate.py.

Both entrypoints build the model via the registry and tokenize via
``build_tokenizer`` so training and inference always agree on the model
architecture and tokenization.
"""

from __future__ import annotations

import torch

from .tokenizers import BaseTokenizer


def generate(
    model,
    tokenizer: BaseTokenizer,
    prompt: str,
    device: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    """Generate a continuation for ``prompt`` and return the decoded string."""
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output = model.generate_text(
        input_ids,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0].detach().cpu().tolist())
