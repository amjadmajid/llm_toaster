"""Feed-forward blocks and factory.

All variants expand to a hidden width of ``mult * n_embd`` (``model.ffn_mult``). Note that
the gated variants (GEGLU, SwiGLU) carry two input projections (gate + up) at that hidden
width, so at equal ``mult`` they have more parameters than the plain GELU MLP. Architecture
comparisons therefore equalise *total* model parameters via the matched-parameter solver
rather than assuming equal ``mult`` means equal size. The output projection of every variant
is flagged ``_is_residual_projection`` for GPT-2 depth-scaled init.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELUFFN(nn.Module):
    """Plain MLP: Linear -> GELU -> Linear (the classic GPT feed-forward)."""

    def __init__(self, n_embd: int, dropout: float = 0.0, mult: int = 4):
        super().__init__()
        hidden = mult * n_embd
        out_proj = nn.Linear(hidden, n_embd)
        out_proj._is_residual_projection = True
        self.net = nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(), out_proj, nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GatedFFN(nn.Module):
    """Gated FFN: down(act(gate(x)) * up(x)). Subclasses pick the gate activation."""

    activation: staticmethod

    def __init__(self, n_embd: int, dropout: float = 0.0, mult: int = 4):
        super().__init__()
        hidden = mult * n_embd
        self.gate = nn.Linear(n_embd, hidden)
        self.up = nn.Linear(n_embd, hidden)
        self.down = nn.Linear(hidden, n_embd)
        self.down._is_residual_projection = True
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.activation(self.gate(x)) * self.up(x)))


class SwiGLUFFN(_GatedFFN):
    """Gated FFN with SiLU/Swish gate (LLaMA-style)."""

    activation = staticmethod(F.silu)


class GEGLUFFN(_GatedFFN):
    """Gated FFN with GELU gate (a real GEGLU, not the plain GELU MLP)."""

    activation = staticmethod(F.gelu)


class MoEFFN(nn.Module):
    """MoE placeholder so TransformerBlock does not need future API changes."""

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        raise NotImplementedError("MoE is a planned extension and is not implemented yet.")


def build_ffn(kind: str, n_embd: int, dropout: float, mult: int = 4) -> nn.Module:
    if kind == "gelu":
        return GELUFFN(n_embd, dropout, mult)
    if kind == "geglu":
        return GEGLUFFN(n_embd, dropout, mult)
    if kind == "swiglu":
        return SwiGLUFFN(n_embd, dropout, mult)
    if kind == "moe":
        return MoEFFN(n_embd, dropout, mult)
    raise ValueError(f"Unknown FFN kind {kind!r}")
