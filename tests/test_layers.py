import torch
from src.model.layers import EmbeddingLayer, PositionalEncoding, MultiHeadAttention, FeedForward, LayerNorm

def test_embedding():
    vocab_size, d_model = 30, 16
    embed = EmbeddingLayer(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (2, 5))  # batch=2, seq_len=5
    out = embed(x)
    assert out.shape == (2, 5, d_model)
    print("EmbeddingLayer OK")

def test_positional_encoding():
    d_model = 16
    pe = PositionalEncoding(d_model)
    x = torch.zeros(2, 5, d_model)
    out = pe(x)
    assert out.shape == (2, 5, d_model)
    print("PositionalEncoding OK")

def test_multihead_attention():
    d_model, num_heads = 16, 4
    attn = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(2, 5, d_model)
    out = attn(x)
    assert out.shape == (2, 5, d_model)
    print("MultiHeadAttention OK")

def test_feedforward():
    d_model, d_ff = 16, 32
    ffn = FeedForward(d_model, d_ff)
    x = torch.randn(2, 5, d_model)
    out = ffn(x)
    assert out.shape == (2, 5, d_model)
    print("FeedForward OK")

def test_layernorm():
    d_model = 16
    norm = LayerNorm(d_model)
    x = torch.randn(2, 5, d_model)
    out = norm(x)
    assert out.shape == (2, 5, d_model)
    print("LayerNorm OK")

if __name__ == "__main__":
    test_embedding()
    test_positional_encoding()
    test_multihead_attention()
    test_feedforward()
    test_layernorm()
