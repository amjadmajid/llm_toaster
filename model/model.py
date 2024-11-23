# model.py

import torch
import torch.nn as nn
from torch.nn import functional as F

class FlashAttention(nn.Module):
    """
    Self-attention mechanism with optional causal masking and attention masking.

    Args:
        n_head (int): Number of attention heads.
        n_embd (int): Embedding size.
        seq_len (int): Maximum sequence length.
        attn_pdrop (float): Dropout rate for attention probabilities.
        causal (bool): Whether to apply causal masking.
    """
    def __init__(self, 
                 n_head: int,       
                 n_embd: int,       
                 seq_len: int,      
                 attn_pdrop: float, 
                 causal: bool, 
                 device: str
                 ):     
        super().__init__()
        assert n_embd % n_head == 0, "Embedding size must be divisible by number of heads"
        
        self.n_head = n_head
        self.head_dim = n_embd // n_head  # Dimension per head.
        self.causal = causal
        self.n_embd = n_embd

        # Linear layers for projecting input to Q, K, V and output projection.
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Dropout layer for regularization.
        self.dropout = nn.Dropout(attn_pdrop)

        # Register causal mask if needed.
        if self.causal:
            self.register_buffer(
                "bias", 
                torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
            )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, C).
            attention_mask (torch.Tensor): Optional attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, C).
        """
        batch_size, seq_len, C = x.size()

        # Compute queries, keys, and values.
        qkv = self.c_attn(x).chunk(3, dim=2)  # Shape: (batch_size, seq_len, 3 * n_embd) -> 3 tensors of (batch_size, seq_len, n_embd)
        q, k, v = [t.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2) for t in qkv]
        # Shape after processing: (batch_size, n_head, seq_len, head_dim).

        # Handle attention mask.
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]  # Shape: (batch_size, 1, 1, seq_len)
            attn_mask = (
                self.bias[:, :, :seq_len, :seq_len].logical_and(extended_attention_mask) 
                if self.causal 
                else extended_attention_mask
            )
        else:
            attn_mask = self.bias[:, :, :seq_len, :seq_len] if self.causal else None

        # Apply scaled dot-product attention.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask) 

        # Reshape back to (batch_size, seq_len, seq_len).
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)

        # Apply output projection.
        y = self.c_proj(y)
        return y



class SelfAttention(nn.Module):
    """
    Self-attention mechanism with optional causal masking and attention masking.

    Args:
        n_head (int): Number of attention heads
        n_embd (int): Embedding size.
        seq_len (int): Maximum sequence length
        attn_pdrop (float): Dropout rate
        causal (bool): If True, apply causal masking
        device (str): Device for training.
    """
    def __init__(self, 
                 n_head: int,       # Number of attention heads.
                 n_embd: int,       # Embedding size.
                 seq_len: int,      # Sequence length.
                attn_pdrop: float,  # Dropout rate for attention probabilities.
                causal: bool,       # Whether to apply causal masking.
                device: str):       # Device on which tensors are stored.
        super().__init__()
        assert n_embd % n_head == 0, "Embedding size must be divisible by number of heads"
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # projecting input embeddings to queries (Q), keys (K), and values (V) in one step. The output shape is (batch_size, seq_len, 3 * n_embd) where batch_size is the batch size, seq_lenis the sequence length.
        self.c_proj = nn.Linear(n_embd, n_embd) # This linear transformation is important because it enables information sharing across the attention heads
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = nn.Dropout(attn_pdrop)
        self.causal = causal # when True the tokens after the current token is masked away (i.e., they do not communicate info)
        if self.causal:
            self.register_buffer( # bufferes are tensors that are not trainable 
                "bias", 
                torch.tril( # apply a lower triangular mask by zeroing out all elements above the main diagonal 
                    torch.ones(seq_len, seq_len, device=device) # create a sequare matrix
                    ).view(1, 1, seq_len, seq_len)) # reshape the matrix from 2D to 4D 

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.size()
        head_dim = n_embd // self.n_head

        qkv = self.c_attn(x).chunk(3, dim=2)
        q, k, v = [t.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2) for t in qkv]
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))

        # Combine masks
        if self.causal and attention_mask is not None:
            # Combine causal mask and attention mask (logical AND)
            combined_mask = self.bias[:, :, :seq_len, :seq_len].logical_and(attention_mask[:, None, None, :])
        elif self.causal:
            # Only causal mask
            combined_mask = self.bias[:, :, :seq_len, :seq_len]
        elif attention_mask is not None:
            # Only attention mask
            combined_mask = attention_mask[:, None, None, :]
        else:
            # No mask
            combined_mask = None

        # Apply combined mask
        if combined_mask is not None:
            att = att.masked_fill(combined_mask == 0, float('-inf'))


        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd) # concatenate the output of the multi-head attention 
        y = self.c_proj(y) # apply a linear transformation to mix info learned by different attention heads. 
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
        self.attention = FlashAttention(n_head, n_embd, seq_len, dropout_rate, causal=decoder, device=device)
        self.feed_forward = FeedForwardLayer(n_embd, n_embd * 4, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), attention_mask))
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
    def __init__(self, 
                 n_head: int, 
                 vocab_size: int, 
                 n_embd: int, 
                 seq_len: int, 
                 device: str, 
                 dropout_rate: float = 0.0, 
                 n_blocks: int = 4, 
                 decoder: bool = True
                 ):
        super(TransformerModel, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(seq_len, n_embd)
        self.TransformerBlocks = nn.ModuleList(
            [TransformerBlock(n_head, n_embd, seq_len, dropout_rate, device, decoder) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Weight sharing scheme
        self.lm_head.weight = self.token_embeddings.weight

        self.device = device
        self.seq_len = seq_len
        self.decoder = decoder

    def forward(self, input_indices: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_indices (torch.Tensor): Input tensor of token indices.
            attention_mask (torch.Tensor): Attention mask indicating valid tokens.

        Returns:
            torch.Tensor: Logits for each token in the vocabulary.
        """
        batch_size, seq_len= input_indices.size()
        position_ids = torch.arange(seq_len, device=input_indices.device).unsqueeze(0).expand(batch_size, seq_len)
        token_emb = self.token_embeddings(input_indices)
        position_emb = self.position_embeddings(position_ids)
        x = token_emb + position_emb
        for block in self.TransformerBlocks:
            x = block(x, attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


    def top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Filters logits to keep only the top k tokens.

        Args:
            logits (torch.Tensor): The logits tensor of shape (batch_size, vocab_size).
            k (int): The number of top tokens to keep.

        Returns:
            torch.Tensor: The filtered logits tensor.
        """
        v, _ = torch.topk(logits, k)
        threshold = v[:, -1].unsqueeze(1)
        logits = torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)
        return logits


    def generate_text(self, input_indices: torch.Tensor, max_length: int, tokenizer, top_k: int = 50, temperature: float = 1.0) -> torch.Tensor:

        self.eval()
        generated_indices = input_indices
        with torch.no_grad():
            for _ in range(max_length):
                input_ids = generated_indices[:, -self.seq_len:]
                attention_mask = torch.ones_like(input_ids).to(self.device)

                logits = self(input_ids, attention_mask)
                logits = logits[:, -1, :] / temperature  # Adjust temperature

                filtered_logits = self.top_k_logits(logits, top_k)
                probabilities = torch.softmax(filtered_logits, dim=-1)

                next_token_id = torch.multinomial(probabilities, num_samples=1)
                generated_indices = torch.cat((generated_indices, next_token_id), dim=1)

                # Stop if EOS token is generated
                if next_token_id.item() == tokenizer.eos_token_id:
                    break

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
