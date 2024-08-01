import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Self-attention mechanism with optional causal masking.

    Args:
        n_head (int): Number of attention heads.
        n_embd (int): Embedding size.
        seq_len (int): Maximum sequence length.
        attn_pdrop (float): Dropout rate.
        causal (bool): If True, apply causal masking.
        device (str): Device for training.
    """
    def __init__(self, n_head: int, n_embd: int, seq_len: int, attn_pdrop: float, causal: bool, device: str):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding size must be divisible by number of heads"
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # Projecting q, k, v in one transformation
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
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class TransformerBlock(nn.Module):
    """
    Transformer block consisting of self-attention and feed-forward layers.

    Args:
        n_head (int): Number of attention heads.
        n_embd (int): Embedding size.
        seq_len (int): Maximum sequence length.
        dropout_rate (float): Dropout rate.
        device (str): Device for training.
        decoder (bool): If True, applies causal masking for the self-attention layer.
    """
    def __init__(self, n_head: int, n_embd: int, seq_len: int, dropout_rate: float, device: str, decoder: bool):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd, seq_len, dropout_rate, causal=decoder, device=device)
        self.feed_forward = FeedForwardLayer(n_embd, n_embd * 4, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class TransformerModel(nn.Module):
    """
    Transformer model with optional encoder/decoder functionality.

    Args:
        n_head (int): Number of attention heads.
        vocab_size (int): Size of the vocabulary.
        n_embd (int): Embedding size.
        seq_len (int): Maximum sequence length.
        device (str): Device for training.
        dropout_rate (float): Dropout rate.
        n_blocks (int): Number of transformer blocks.
        decoder (bool): If True, model is used as a decoder with causal masking.
    """
    def __init__(self, n_head: int, vocab_size: int, n_embd: int, seq_len: int, device: str, dropout_rate: float = 0.0, n_blocks: int = 4, decoder: bool = False):
        super(TransformerModel, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(seq_len, n_embd)
        self.TransformerBlocks = nn.ModuleList(
            [TransformerBlock(n_head, n_embd, seq_len, dropout_rate, device, decoder) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(n_embd)
        self.output_layer = nn.Linear(n_embd, vocab_size)

        # Weight sharing scheme
        self.token_embeddings.weight = self.output_layer.weight

        self.device = device
        self.seq_len = seq_len
        self.decoder = decoder

    def forward(self, input_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_indices (torch.Tensor): Input tensor of token indices.

        Returns:
            torch.Tensor: Logits for each token in the vocabulary.
        """
        token_emb = self.token_embeddings(input_indices)
        position_emb = self.position_embeddings(torch.arange(input_indices.size(1), device=self.device))
        x = token_emb + position_emb
        for transformer_block in self.TransformerBlocks:
            x = transformer_block(x)
        x = self.norm(x)
        logits = self.output_layer(x)
        return logits

    def generate_text(self, start_indices: torch.Tensor, max_length: int, topk: int = 35) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            start_indices (torch.Tensor): Starting indices for generation.
            max_length (int): Maximum length of the generated sequence.
            topk (int): Number of top probabilities to sample from.

        Returns:
            torch.Tensor: Generated token indices.
        """
        self.eval()
        generated_indices = start_indices
        for _ in range(max_length):
            # Ensure the input length does not exceed the sequence length
            input_indices = generated_indices[:, -self.seq_len:]

            # Get the logits from the model
            logits = self(input_indices)
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)
            
            # Perform top-k sampling
            topk_p, topk_i = torch.topk(probabilities, topk, dim=-1)
            
            # Sample from the top-k probabilities
            next_index = torch.multinomial(topk_p, num_samples=1)
            
            # Gather the actual token indices
            next_index = torch.gather(topk_i, 1, next_index)
            
            # Append the sampled token to the generated sequence
            generated_indices = torch.cat((generated_indices, next_index), dim=1)
        
        self.train()
        return generated_indices

class FeedForwardLayer(nn.Module):
    """
    Feed-forward neural network layer.

    Args:
        n_embd (int): Embedding size.
        hidden_dim (int): Size of the hidden layer.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_embd: int, hidden_dim: int, dropout_rate: float = 0.0):
        super(FeedForwardLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
