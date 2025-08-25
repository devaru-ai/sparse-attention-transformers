# chunked_ffn.py
import torch
import torch.nn as nn

class ChunkedFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, chunk_size=64):
        super().__init__()
        self.chunk_size = chunk_size
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("[ChunkedFeedForward] Input shape:", x.shape)
        batch, seq_len, d_model = x.size()
        out = torch.zeros_like(x)
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end, :]
            print(f" - Processing chunk {start}:{end} (shape: {chunk.shape})")
            out[:, start:end, :] = self.linear2(self.relu(self.linear1(chunk)))
        print("[ChunkedFeedForward] Output shape:", out.shape)
        return out
