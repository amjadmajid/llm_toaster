"""Feed-forward blocks and factory."""

from __future__ import annotations

import torch
import torch.nn as nn


class GELUFFN(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        out_proj = nn.Linear(4 * n_embd, n_embd)
        out_proj._is_residual_projection = True  # depth-scaled init (GPT-2 style)
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            out_proj,
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        hidden = 4 * n_embd
        self.gate = nn.Linear(n_embd, hidden)
        self.up = nn.Linear(n_embd, hidden)
        self.down = nn.Linear(hidden, n_embd)
        self.down._is_residual_projection = True  # depth-scaled init (GPT-2 style)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x)))


class MoEFFN(nn.Module):
    """MoE placeholder so TransformerBlock does not need future API changes."""

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        raise NotImplementedError("MoE is a planned extension and is not implemented yet.")


def build_ffn(kind: str, n_embd: int, dropout: float) -> nn.Module:
    if kind in {"gelu", "geglu"}:
        return GELUFFN(n_embd, dropout)
    if kind == "swiglu":
        return SwiGLUFFN(n_embd, dropout)
    if kind == "moe":
        return MoEFFN(n_embd, dropout)
    raise ValueError(f"Unknown FFN kind {kind!r}")
