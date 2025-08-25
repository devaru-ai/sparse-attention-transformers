import torch
import torch.nn as nn

class GatedReversibleBlock(nn.Module):
    def __init__(self, layer_F, layer_G, d_model):
        super().__init__()
        self.F = layer_F
        self.G = layer_G
        self.gate1 = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x1, x2):
        print(f"[GatedReversibleBlock] x1 shape: {x1.shape}, x2 shape: {x2.shape}")
        f_out = self.F(x2)
        print(f"  F(x2) shape: {f_out.shape}")
        g1 = self.gate1(x1)
        print(f"  gate1(x1) shape: {g1.shape}")
        y1 = g1 * x1 + (1 - g1) * f_out
        print(f"  y1 shape: {y1.shape}")
        g_out = self.G(y1)
        print(f"  G(y1) shape: {g_out.shape}")
        g2 = self.gate2(x2)
        print(f"  gate2(x2) shape: {g2.shape}")
        y2 = g2 * x2 + (1 - g2) * g_out
        print(f"  y2 shape: {y2.shape}")
        return y1, y2
