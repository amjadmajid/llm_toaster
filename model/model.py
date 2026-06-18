from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class FlashAttention(nn.Module):
    """Self-attention mechanism backed by PyTorch scaled-dot-product attention."""

    def __init__(self, n_head: int, n_embd: int, attn_pdrop: float, causal: bool):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding size must be divisible by number of heads"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.causal = causal
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.output_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(attn_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v)]
        dropout_p = self.dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=self.causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.output_proj(y)


class SelfAttention(nn.Module):
    """Reference self-attention implementation with optional causal masking."""

    def __init__(self, n_head: int, n_embd: int, seq_len: int, attn_pdrop: float, causal: bool, device: str):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding size must be divisible by number of heads"
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = nn.Dropout(attn_pdrop)
        self.causal = causal
        if self.causal:
            self.register_buffer("bias", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x).chunk(3, dim=2)
        q, k, v = [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in qkv]
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        if self.causal:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class TransformerBlock(nn.Module):
    """Transformer block consisting of self-attention and feed-forward layers."""

    def __init__(self, n_head: int, n_embd: int, seq_len: int, dropout_rate: float, device: str, decoder: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.attention = FlashAttention(n_head, n_embd, dropout_rate, causal=decoder)
        self.feed_forward = FeedForwardLayer(n_embd, n_embd * 4, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class TransformerModel(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        n_head: int,
        vocab_size: int,
        n_embd: int,
        seq_len: int,
        device: str,
        dropout_rate: float = 0.0,
        n_blocks: int = 4,
        decoder: bool = False,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(seq_len, n_embd)
        self.TransformerBlocks = nn.ModuleList(
            [TransformerBlock(n_head, n_embd, seq_len, dropout_rate, device, decoder) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.token_embeddings.weight = self.lm_head.weight
        self.seq_len = seq_len
        self.decoder = decoder

    def forward(self, input_indices: torch.Tensor) -> torch.Tensor:
        B, T = input_indices.size()
        if T > self.seq_len:
            raise ValueError(f"Input sequence length {T} exceeds configured seq_len {self.seq_len}")
        token_emb = self.token_embeddings(input_indices)
        positions = torch.arange(T, device=input_indices.device)
        position_emb = self.position_embeddings(positions)
        x = token_emb + position_emb
        for block in self.TransformerBlocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate_text(
        self,
        start_indices: torch.Tensor,
        max_length: int,
        topk: Optional[int] = 35,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate token IDs with temperature, top-k, and optional nucleus sampling."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        was_training = self.training
        self.eval()
        generated_indices = start_indices
        try:
            for _ in range(max_length):
                input_indices = generated_indices[:, -self.seq_len:]
                logits = self(input_indices)[:, -1, :] / temperature
                logits = _filter_logits(logits, topk=topk, top_p=top_p)
                probabilities = torch.softmax(logits, dim=-1)
                next_index = torch.multinomial(probabilities, num_samples=1)
                generated_indices = torch.cat((generated_indices, next_index), dim=1)
                if eos_token_id is not None and torch.all(next_index == eos_token_id):
                    break
        finally:
            self.train(was_training)
        return generated_indices


class FeedForwardLayer(nn.Module):
    """Feed-forward neural network layer."""

    def __init__(self, n_embd: int, hidden_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _filter_logits(logits: torch.Tensor, topk: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
    if topk is not None and topk > 0:
        topk = min(topk, logits.size(-1))
        threshold = torch.topk(logits, topk, dim=-1).values[:, [-1]]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits
