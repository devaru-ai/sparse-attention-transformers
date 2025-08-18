import torch
import torch.nn as nn
import math

# ---------------------------
# 1. Embedding Layer
# ---------------------------
class EmbeddingLayer(nn.Module):
    """
    Token embedding layer: maps token indices to learned vector representations.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        x: (batch, seq_len) token indices
        returns: (batch, seq_len, d_model)
        """
        return self.embedding(x)


# ---------------------------
# 2. Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings using sine/cosine functions.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model) with positional encodings added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ---------------------------
# 3. Multi-Head Attention
# ---------------------------
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # each: (batch, seq_len, d_model)

        # Split into heads
        def split_heads(t):
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k, v = [split_heads(t) for t in (q, k, v)]

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(attn_output)


# ---------------------------
# 4. Feed-Forward Network
# ---------------------------
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


# ---------------------------
# 5. Layer Normalization
# ---------------------------
class LayerNorm(nn.Module):
    """
    Applies Layer Normalization.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.norm(x)


# ---------------------------
# 6. Residual Helper (Inline)
# ---------------------------
def residual_connection(x, sublayer):
    """
    Applies a residual (skip) connection: sublayer(x) + x.
    sublayer: a callable module
    """
    return x + sublayer(x)
