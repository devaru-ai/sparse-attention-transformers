import torch
from src.model.reformer import Reformer

def test_reformer_forward():
    src_vocab_size = 30
    tgt_vocab_size = 35
    batch, src_len, tgt_len = 2, 14, 11
    d_model, d_ff = 32, 64
    num_heads, n_layers = 4, 2
    max_len = 16

    model = Reformer(
        src_vocab_size, tgt_vocab_size,
        d_model=d_model, d_ff=d_ff,
        num_heads=num_heads, num_layers=n_layers, max_len=max_len
    )
    src = torch.randint(0, src_vocab_size, (batch, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch, tgt_len))
    out = model(src, tgt)
    assert out.shape == (batch, tgt_len, tgt_vocab_size), f"Expected {(batch, tgt_len, tgt_vocab_size)}, got {out.shape}"
    assert torch.isfinite(out).all(), "Non-finite values found in Reformer output"
    print("test_reformer_forward passed!")

if __name__ == "__main__":
    test_reformer_forward()
