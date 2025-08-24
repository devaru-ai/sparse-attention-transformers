import torch
from src.model.transformer import Transformer

def test_transformer_forward():
    vocab_size = 30
    src = torch.randint(0, vocab_size, (2, 10))    # batch=2, seq_len=10
    tgt = torch.randint(0, vocab_size, (2, 9))     # batch=2, seq_len=9 (decoder input)

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=16,
        d_ff=32,
        num_heads=4,
        num_layers=2,
        max_len=15
    )
    out = model(src, tgt)
    assert out.shape == (2, 9, vocab_size)
    print("Transformer Forward OK")

if __name__ == "__main__":
    test_transformer_forward()
