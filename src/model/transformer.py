import torch
import torch.nn as nn
from .layers import EmbeddingLayer, PositionalEncoding, MultiHeadAttention, FeedForward, LayerNorm

# ---------- Encoder Layer ----------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out = self.self_attn(x)  # add src_mask logic if needed
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# ---------- Encoder ----------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, d_ff, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)
        ])
    def forward(self, x, src_mask=None):
        x = self.embedding(x)           # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

# ---------- Decoder Layer ----------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        self_attn_out = self.self_attn(x)  # Add tgt_mask logic for causal masking
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out = self.cross_attn(x)  # Add logic to attend to memory/encoder output
        x = self.norm2(x + self.dropout(cross_attn_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x

# ---------- Decoder ----------
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, d_ff, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        x = self.embedding(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.out_proj(x)  # logits for next-token prediction or seq2seq tasks

# ---------- Full Transformer ----------
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size, tgt_vocab_size,
                 d_model=512, d_ff=2048,
                 num_heads=8, num_layers=6,
                 max_len=256, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, max_len, d_ff, num_heads, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, max_len, d_ff, num_heads, num_layers, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return out
