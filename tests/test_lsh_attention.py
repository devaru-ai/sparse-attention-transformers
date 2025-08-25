import torch
from src.model.lsh_attention import LSHAttention

def test_lsh_attention_forward():
    batch, seq_len, d_model = 2, 16, 32
    num_heads = 4
    n_buckets = 8
    n_hashes = 2

    lsh_att = LSHAttention(d_model, num_heads=num_heads, n_buckets=n_buckets, n_hashes=n_hashes)
    x = torch.randn(batch, seq_len, d_model)
    out = lsh_att(x)
    assert out.shape == (batch, seq_len, d_model), f"Expected {(batch, seq_len, d_model)}, got {out.shape}"
    assert torch.isfinite(out).all(), "Non-finite values found in LSHAttention output"
    print("test_lsh_attention_forward passed!")

if __name__ == "__main__":
    test_lsh_attention_forward()
