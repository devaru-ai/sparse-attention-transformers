import torch
from src.model.chunked_ffn import ChunkedFeedForward

def test_chunked_ffn_forward():
    batch, seq_len, d_model, d_ff = 2, 20, 32, 64
    chunk_size = 5
    ff = ChunkedFeedForward(d_model, d_ff, chunk_size)
    x = torch.randn(batch, seq_len, d_model)
    out = ff(x)
    assert out.shape == (batch, seq_len, d_model)
    print("test_chunked_ffn_forward passed!")

if __name__ == "__main__":
    test_chunked_ffn_forward()
