# reversible.py
import torch
import torch.nn as nn

class ReversibleBlock(nn.Module):
    def __init__(self, attn, ff, d_model):
        super().__init__()
        self.attn = attn
        self.ff = ff
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        print("[ReversibleBlock] Input shapes:", x1.shape, x2.shape)
        y1 = x1 + self.attn(self.norm1(x2))
        print("[ReversibleBlock] y1 shape:", y1.shape)
        y2 = x2 + self.ff(self.norm2(y1))
        print("[ReversibleBlock] y2 shape:", y2.shape)
        return y1, y2

    def backward_pass(self, y1, y2):
        print("[ReversibleBlock] Backward pass input:", y1.shape, y2.shape)
        x2 = y2 - self.ff(self.norm2(y1))
        x1 = y1 - self.attn(self.norm1(x2))
        print("[ReversibleBlock] Backward pass output:", x1.shape, x2.shape)
        return x1, x2
