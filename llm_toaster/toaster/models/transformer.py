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

    def forward(self, x: torch.Tensor, layer_cache=None, start_pos: int = 0):
        attn_out, new_cache = self.attention(self.norm1(x), layer_cache, start_pos)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x, new_cache


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
        """Full-sequence forward used for training/eval (no KV-cache)."""
        return self._forward(input_indices)[0]

    def _forward(self, input_indices: torch.Tensor, caches=None, start_pos: int = 0):
        """Shared forward. ``caches`` is a per-layer list of (k, v) for incremental decode;
        ``start_pos`` is the absolute position of the first token. Returns (logits, new_caches)."""
        _batch, seq_len = input_indices.shape
        if start_pos + seq_len > self.seq_len:
            raise ValueError(f"Position {start_pos + seq_len} exceeds configured seq_len {self.seq_len}")
        x = self.token_embeddings(input_indices)
        if self.position_embeddings is not None:
            positions = torch.arange(start_pos, start_pos + seq_len, device=input_indices.device)
            x = x + self.position_embeddings(positions)
        new_caches = []
        for index, block in enumerate(self.TransformerBlocks):
            layer_cache = caches[index] if caches is not None else None
            x, new_cache = block(x, layer_cache, start_pos)
            new_caches.append(new_cache)
        return self.lm_head(self.norm(x)), new_caches

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
        """Reference generation: re-runs the full forward each step (no KV-cache)."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        was_training = self.training
        self.eval()
        generated = start_indices
        try:
            for _ in range(max_length):
                logits = self(generated[:, -self.seq_len :])[:, -1, :]
                next_token = _sample_next(logits, temperature, top_k, top_p)
                generated = torch.cat([generated, next_token], dim=1)
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
        finally:
            self.train(was_training)
        return generated

    @torch.no_grad()
    def generate_cached(
        self,
        start_indices: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """KV-cached generation: prefill the prompt once, then feed one token at a time.

        Reuses cached keys/values so per-token attention is O(context) instead of the full
        forward's O(context^2) — the realistic deployment decode path. Produces identical tokens
        to ``generate_text`` for the same sampling decision (verified greedily in tests). Bounded by
        ``seq_len`` (no eviction). Note: at small scale / on CPU, the Python overhead of one forward
        per token can outweigh the compute saving; the win shows at larger models / longer contexts.
        """
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        was_training = self.training
        self.eval()
        generated = start_indices
        caches = [(None, None)] * len(self.TransformerBlocks)  # (None, None) = caching on, empty
        try:
            logits, caches = self._forward(generated[:, -self.seq_len :], caches, start_pos=0)
            for _ in range(max_length):
                next_token = _sample_next(logits[:, -1, :], temperature, top_k, top_p)
                generated = torch.cat([generated, next_token], dim=1)
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                if generated.shape[1] >= self.seq_len:  # filled the trained context window
                    break
                # next_token sits at absolute position (len-1); cache holds 0..len-2 so far.
                logits, caches = self._forward(next_token, caches, start_pos=generated.shape[1] - 1)
        finally:
            self.train(was_training)
        return generated


def _sample_next(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """Sample one next token from (B, vocab) logits. top_k=1 is greedy (deterministic argmax)."""
    logits = _filter_top_p(_filter_top_k(logits / temperature, top_k), top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


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
