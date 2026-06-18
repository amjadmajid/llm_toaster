"""Decoder-only Transformer language model."""

from __future__ import annotations

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
        self.feed_forward = build_ffn(model_config.ffn, model_config.n_embd, model_config.dropout_rate)
        self.dropout = nn.Dropout(model_config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class TransformerModel(nn.Module):
    """Small decoder-only language model used for pretraining and SFT."""

    def __init__(self, model_config, attention_config):
        super().__init__()
        if model_config.position == "rope":
            raise NotImplementedError("RoPE is reserved in config but not implemented yet.")
        vocab_size = model_config.vocab_size or 50304
        self.seq_len = model_config.seq_len
        self.token_embeddings = nn.Embedding(vocab_size, model_config.n_embd)
        self.position_embeddings = nn.Embedding(model_config.seq_len, model_config.n_embd)
        self.TransformerBlocks = nn.ModuleList(
            TransformerBlock(model_config, attention_config) for _ in range(model_config.n_blocks)
        )
        self.norm = build_norm(model_config.norm, model_config.n_embd)
        self.lm_head = nn.Linear(model_config.n_embd, vocab_size, bias=False)
        if model_config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, input_indices: torch.Tensor) -> torch.Tensor:
        _batch, seq_len = input_indices.shape
        if seq_len > self.seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds configured seq_len {self.seq_len}")
        positions = torch.arange(seq_len, device=input_indices.device)
        x = self.token_embeddings(input_indices) + self.position_embeddings(positions)
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
                if top_k is not None and top_k > 0:
                    values = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
        finally:
            self.train(was_training)
        return generated
