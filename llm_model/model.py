import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ENLARGE_FACTOR = 4

class TransformerModel(nn.Module):
    """
    Simple Transformer-based Language Model.

    Args:
        n_head (int): Number of attention heads.
        vocab_size (int): Size of the vocabulary.
        n_embd (int): Embedding size.
        seq_len (int): Maximum sequence length.
        device (str): Device for training.
        dropout_rate (float): Dropout rate.
        n_blocks (int): Number of transformer blocks.
    """

    def __init__(self, n_head, vocab_size, n_embd, seq_len, device, dropout_rate=0.0, n_blocks=4):
        super(TransformerModel, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(seq_len, n_embd)
        self.TransformerBlocks = nn.ModuleList([TransformerBlock(n_head, n_embd, seq_len, dropout_rate) for _ in range(n_blocks)])
        self.norm =  nn.LayerNorm(n_embd)
        self.output_layer = nn.Linear(n_embd, vocab_size)


        # weights sharing scheme
        self.token_embeddings.weight = self.output_layer.weight
        
        self.device = device
        self.seq_len = seq_len

    def forward(self, input_indices):
        """
        Forward pass of the model.

        Args:
            input_indices (torch.Tensor): Input tensor of token indices.

        Returns:
            torch.Tensor: Logits for each token in the vocabulary.
        """
        # logger.info(f"Model input shape: {input_indices.shape}")
        token_emb = self.token_embeddings(input_indices)
        position_emb = self.position_embeddings(torch.arange(input_indices.size(1), device=self.device))
        x = token_emb + position_emb
        for transformer_block in self.TransformerBlocks:
            x = transformer_block(x)
        x = self.norm(x)
        logits = self.output_layer(x)
        return logits
    
    def generate(self, start_indices, max_length):
        """
        Generate text using the model.

        Args:
            start_indices (torch.Tensor): Starting indices for generation.
            max_length (int): Maximum length of the generated sequence.

        Returns:
            torch.Tensor: Generated token indices.
        """
        self.eval()
        generated_indices = start_indices
        for _ in range(max_length):
            logits = self(generated_indices[:, -self.seq_len:])
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            generated_indices = torch.cat((generated_indices, next_index), dim=1)
        self.train()
        return generated_indices

class TransformerBlock(nn.Module):
    """
    Transformer block consisting of self-attention and feed-forward layers.

    Args:
        n_head (int): Number of attention heads.
        n_embd (int): Embedding size.
        seq_len (int): Maximum sequence length.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_head, n_embd, seq_len, dropout_rate=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.attention = CausalSelfAttention(n_head, n_embd, seq_len)
        self.feed_forward = FeedForward(n_embd, n_embd * ENLARGE_FACTOR, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention mechanism.

    Args:
        n_head (int): Number of attention heads.
        n_embd (int): Embedding size.
        seq_len (int): Maximum sequence length.
    """
    def __init__(self, n_head, n_embd, seq_len):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd) # 3 * n_embd is for projecting the q k v in one transformation
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class FeedForward(nn.Module):
    """
    Feed-forward neural network layer.

    Args:
        n_embd (int): Embedding size.
        hidden_dim (int): Size of the hidden layer.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_embd, hidden_dim, dropout_rate=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,  hidden_dim * ENLARGE_FACTOR),
            nn.GELU(),
            nn.Linear(hidden_dim * ENLARGE_FACTOR, n_embd),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        return self.net(x)
