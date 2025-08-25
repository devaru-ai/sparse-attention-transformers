# reformer.py
import torch
import torch.nn as nn
from .lsh_attention import LSHAttention
from .reversible import ReversibleBlock
from .chunked_ffn import ChunkedFeedForward

class ReformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, n_buckets=64, n_hashes=4, chunk_size=64):
        super().__init__()
        self.block = ReversibleBlock(
            LSHAttention(d_model, num_heads, n_buckets, n_hashes),
            ChunkedFeedForward(d_model, d_ff, chunk_size),
            d_model
        )
    def forward(self, x1, x2):
        print(f"[ReformerLayer] x1/x2 shapes: {x1.shape}, {x2.shape}")
        return self.block(x1, x2)

class ReformerStack(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, d_ff, num_heads, num_layers,
                 n_buckets=64, n_hashes=4, chunk_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList([
            ReformerLayer(d_model, d_ff, num_heads, n_buckets, n_hashes, chunk_size)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        print("[ReformerStack] input shape:", x.shape)
        seq_len = x.size(1)
        x_emb = self.embedding(x) + self.pos_enc[:, :seq_len, :]
        print("[ReformerStack] embedded shape:", x_emb.shape)
        x1, x2 = x_emb, torch.zeros_like(x_emb)
        for i, layer in enumerate(self.layers):
            print(f"[ReformerStack] Layer {i+1}: x1/x2 shapes: {x1.shape}, {x2.shape}")
            x1, x2 = layer(x1, x2)
            print(f"[ReformerStack] Layer {i+1} output: x1/x2 shapes: {x1.shape}, {x2.shape}")
        out = (x1 + x2) / 2
        print("[ReformerStack] output shape:", out.shape)
        return out

class Reformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, d_ff=2048, num_heads=8, num_layers=6, max_len=256,
                 n_buckets=64, n_hashes=4, chunk_size=64):
        super().__init__()
        self.encoder = ReformerStack(src_vocab_size, d_model, max_len, d_ff, num_heads, num_layers,
                                    n_buckets, n_hashes, chunk_size)
        self.decoder = ReformerStack(tgt_vocab_size, d_model, max_len, d_ff, num_heads, num_layers,
                                    n_buckets, n_hashes, chunk_size)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        print("[Reformer] encoder input:", src.shape)
        memory = self.encoder(src)
        print("[Reformer] decoder input:", tgt.shape)
        tgt_rep = self.decoder(tgt)
        print("[Reformer] decoder output:", tgt_rep.shape)
        logits = self.out_proj(tgt_rep)
        print("[Reformer] logits shape:", logits.shape)
        return logits
