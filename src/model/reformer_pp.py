import torch
import torch.nn as nn
from .learnable_lsh_attention import LearnableLSHAttention
from .dynamic_local_global import DynamicLocalGlobalRouter
from .gated_reversible import GatedReversibleBlock

class ReformerPPBlock(nn.Module):
    def __init__(self, d_model, num_heads, bucket_size, seq_len, n_hashes=4, local_radius=4):
        super().__init__()
        self.attn = LearnableLSHAttention(d_model, num_heads, bucket_size, n_hashes)
        self.router = DynamicLocalGlobalRouter(seq_len, num_heads, local_radius, d_model // num_heads)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.revblock = GatedReversibleBlock(self._attn_with_router, self.ffn, d_model)

    def _attn_with_router(self, x):
        print(f"[ReformerPPBlock] Calling attention+router")
        out, reg_loss = self.attn(x, self.router, regularizer=True)
        self.reg_loss = reg_loss
        print(f"[ReformerPPBlock] Attention+Router done | out: {out.shape}, reg_loss: {reg_loss}")
        return out

    def forward(self, x1, x2):
        print(f"[ReformerPPBlock] x1: {x1.shape}, x2: {x2.shape}")
        y1, y2 = self.revblock(x1, x2)
        print(f"[ReformerPPBlock] y1: {y1.shape}, y2: {y2.shape}")
        return y1, y2, getattr(self, 'reg_loss', torch.tensor(0.0, device=x1.device))

class ReformerPP(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=128, d_ff=256, num_heads=4, num_layers=2,
                 max_len=512, bucket_size=64, n_hashes=4, local_radius=4):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.max_len = max_len
        self.d_model = d_model
        # Start with a long position encoding for any reasonable sequence
        self.register_buffer('pos_enc', torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([
            ReformerPPBlock(d_model, num_heads, bucket_size, max_len, n_hashes, local_radius)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        print(f"[ReformerPP] src shape: {src.shape}, tgt shape: {tgt.shape}")
        seq_len = src.size(1)
        # Dynamically expand position encoding if needed
        if seq_len > self.pos_enc.shape[1]:
            print(f"[ReformerPP] Expanding position encoding from {self.pos_enc.shape[1]} to {seq_len}")
            new_pos_enc = torch.zeros(1, seq_len, self.d_model, device=src.device, dtype=self.pos_enc.dtype)
            new_pos_enc[:, :self.pos_enc.shape[1], :] = self.pos_enc
            self.pos_enc = new_pos_enc
        x_emb = self.embedding(src) + self.pos_enc[:, :seq_len, :]
        print(f"[ReformerPP] embedded shape: {x_emb.shape}")
        x1, x2 = x_emb, torch.zeros_like(x_emb)
        total_reg_loss = torch.tensor(0.0, device=src.device)
        for i, block in enumerate(self.blocks):
            print(f"[ReformerPP] Block {i+1}")
            x1, x2, reg_loss = block(x1, x2)
            print(f"[ReformerPP] Block {i+1} done | x1/x2: {x1.shape}, {x2.shape}, reg_loss: {reg_loss}")
            total_reg_loss += reg_loss
        out = (x1 + x2) / 2
        if tgt is not None:
            target_len = tgt.size(1)
            print(f"[ReformerPP] Slicing output from {out.shape[1]} to target length {target_len}.")
            out = out[:, :target_len, :]
        logits = self.out_proj(out)
        print(f"[ReformerPP] logits shape: {logits.shape}, total_reg_loss: {total_reg_loss}")
        return logits, total_reg_loss
