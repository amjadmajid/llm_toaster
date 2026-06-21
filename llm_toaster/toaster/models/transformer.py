"""Decoder-only Transformer language model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import build_ffn
from .norms import build_norm


class TransformerBlock(nn.Module):
    def __init__(self, model_config, attention_config):
        super().__init__()
        self.norm1 = build_norm(model_config.norm, model_config.n_embd)
        self.attention = MultiHeadAttention(model_config, attention_config, causal=True)
        self.norm2 = build_norm(model_config.norm, model_config.n_embd)
        self.feed_forward = build_ffn(
            model_config.ffn, model_config.n_embd, model_config.dropout_rate, model_config.ffn_mult
        )
        self.dropout = nn.Dropout(model_config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class TransformerModel(nn.Module):
    """Small decoder-only language model used for pretraining and SFT."""

    def __init__(self, model_config, attention_config):
        super().__init__()
        vocab_size = model_config.vocab_size or 50304
        self.seq_len = model_config.seq_len
        self.position = model_config.position
        self.token_embeddings = nn.Embedding(vocab_size, model_config.n_embd)
        # Learned absolute positions add an embedding table; RoPE injects positions inside
        # attention; "none" uses no positional signal at all.
        self.position_embeddings = (
            nn.Embedding(model_config.seq_len, model_config.n_embd) if model_config.position == "learned" else None
        )
        self.TransformerBlocks = nn.ModuleList(
            TransformerBlock(model_config, attention_config) for _ in range(model_config.n_blocks)
        )
        self.norm = build_norm(model_config.norm, model_config.n_embd)
        self.lm_head = nn.Linear(model_config.n_embd, vocab_size, bias=False)
        if model_config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        self._init_parameters(model_config.n_blocks)

    def _init_parameters(self, n_blocks: int) -> None:
        """GPT-2 style init: N(0, 0.02) weights, zero biases, with residual output
        projections additionally scaled by 1/sqrt(2*n_blocks) so the residual stream
        doesn't blow up with depth. Default PyTorch init (embeddings ~N(0,1)) makes the
        starting loss far exceed ln(vocab); this brings it down to ~ln(vocab)."""
        self.apply(_init_weights)
        residual_std = 0.02 / math.sqrt(2 * n_blocks)
        for module in self.modules():
            if getattr(module, "_is_residual_projection", False):
                nn.init.normal_(module.weight, mean=0.0, std=residual_std)

    def forward(self, input_indices: torch.Tensor) -> torch.Tensor:
        _batch, seq_len = input_indices.shape
        if seq_len > self.seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds configured seq_len {self.seq_len}")
        x = self.token_embeddings(input_indices)
        if self.position_embeddings is not None:
            positions = torch.arange(seq_len, device=input_indices.device)
            x = x + self.position_embeddings(positions)
        for block in self.TransformerBlocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate_text(
        self,
        start_indices: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        was_training = self.training
        self.eval()
        generated = start_indices
        try:
            for _ in range(max_length):
                logits = self(generated[:, -self.seq_len :])[:, -1, :] / temperature
                logits = _filter_top_k(logits, top_k)
                logits = _filter_top_p(logits, top_p)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
        finally:
            self.train(was_training)
        return generated


def _filter_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    values = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values
    return logits.masked_fill(logits < values[:, [-1]], float("-inf"))


def _filter_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or not 0.0 < top_p < 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # Drop tokens once the cumulative mass exceeds top_p, but always keep the first.
    remove = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
