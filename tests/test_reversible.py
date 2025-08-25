import torch
import torch.nn as nn

class DummyF(nn.Module):
    def forward(self, x):
        return x * 2

class DummyG(nn.Module):
    def forward(self, x):
        return x * 3

class ReversibleBlock(nn.Module):
    def __init__(self, F, G):
        super().__init__()
        self.F = F
        self.G = G
    def forward(self, x1, x2):
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        return y1, y2
    def backward_pass(self, y1, y2):
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return x1, x2

def test_dummy_reversible_block():
    x1 = torch.randn(2,3)
    x2 = torch.randn(2,3)
    block = ReversibleBlock(DummyF(), DummyG())
    y1, y2 = block.forward(x1, x2)
    x1_rev, x2_rev = block.backward_pass(y1, y2)
    assert torch.allclose(x1, x1_rev, atol=1e-6)
    assert torch.allclose(x2, x2_rev, atol=1e-6)
    print("Dummy reversible block passes reversibility test!")

if __name__ == "__main__":
    test_dummy_reversible_block()
