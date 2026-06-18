"""Attention backends for decoder-only Transformer blocks."""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Causal self-attention supporting eager, SDPA, MQA, and GQA layouts."""

    def __init__(self, config, attention_config, causal: bool = True):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.num_key_value_heads = config.num_key_value_heads or config.n_head
        if self.n_head % self.num_key_value_heads != 0:
            raise ValueError("n_head must be divisible by num_key_value_heads")
        self.head_dim = config.n_embd // config.n_head
        self.backend = attention_config.backend
        self.sdpa_kernel = attention_config.sdpa_kernel
        self.causal = causal

        kv_width = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, kv_width)
        self.v_proj = nn.Linear(config.n_embd, kv_width)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(attention_config.dropout or config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q = self._shape_projection(self.q_proj(x), batch, seq_len, self.n_head)
        k = self._shape_projection(self.k_proj(x), batch, seq_len, self.num_key_value_heads)
        v = self._shape_projection(self.v_proj(x), batch, seq_len, self.num_key_value_heads)
        k, v = self._repeat_kv_if_needed(k, v)

        if self.backend == "eager":
            out = self._eager_attention(q, k, v)
        elif self.backend.startswith("sdpa"):
            out = self._sdpa_attention(q, k, v)
        elif self.backend in {"flash_attn_2", "xformers"}:
            raise ImportError(f"Optional attention backend {self.backend!r} is not installed or wired yet.")
        else:
            raise ValueError(f"Unknown attention backend {self.backend!r}")
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.o_proj(out)

    def _shape_projection(self, tensor: torch.Tensor, batch: int, seq_len: int, heads: int) -> torch.Tensor:
        return tensor.view(batch, seq_len, heads, self.head_dim).transpose(1, 2)

    def _repeat_kv_if_needed(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_key_value_heads == self.n_head:
            return k, v
        repeats = self.n_head // self.num_key_value_heads
        return k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)

    def _eager_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        seq_len = q.size(-2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        return probs @ v

    def _sdpa_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dropout_p = self.dropout.p if self.training else 0.0
        with _sdpa_kernel_context(self.backend, self.sdpa_kernel):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=self.causal)


@contextlib.contextmanager
def _sdpa_kernel_context(backend: str, sdpa_kernel: str):
    """Best-effort SDPA kernel selector with graceful fallback across PyTorch versions."""
    kernel_name = sdpa_kernel
    if backend == "sdpa_flash":
        kernel_name = "flash"
    elif backend == "sdpa_mem_efficient":
        kernel_name = "mem_efficient"
    elif backend == "sdpa_math":
        kernel_name = "math"
    attention_module = getattr(torch.nn, "attention", None)
    if kernel_name == "auto" or attention_module is None or not hasattr(attention_module, "sdpa_kernel"):
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        mapping = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH,
        }
        with sdpa_kernel(mapping[kernel_name]):
            yield
    except Exception:
        yield
