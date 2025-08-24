import torch
import torch.nn as nn
import torch.optim as optim
from src.model.transformer import Transformer
from src.model.utils import load_vocab, load_encoded, pad_batch


src_vocab = load_vocab("data/processed/vocab.src")
tgt_vocab = load_vocab("data/processed/vocab.tgt")
src_train = load_encoded("data/processed/train.src")
tgt_train = load_encoded("data/processed/train.tgt")

batch_size = 32
n_epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(len(src_vocab), len(tgt_vocab), d_model=128, d_ff=256, num_heads=4, num_layers=2, max_len=128).to(device)


#model = Transformer(len(src_vocab), len(tgt_vocab), d_model=128, d_ff=256, num_heads=4, num_layers=2, max_len=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

def get_batches(src_list, tgt_list, batch_size):
    for i in range(0, len(src_list), batch_size):
        src = pad_batch(src_list[i:i+batch_size])
        tgt = pad_batch(tgt_list[i:i+batch_size])
        src = torch.tensor(src, dtype=torch.long, device=device)
        tgt = torch.tensor(tgt, dtype=torch.long, device=device)
        yield src, tgt

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for src_b, tgt_b in get_batches(src_train, tgt_train, batch_size):
        dec_in = tgt_b[:, :-1]
        dec_target = tgt_b[:, 1:]
        optimizer.zero_grad()
        logits = model(src_b, dec_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), f"../outputs/checkpoints/model_epoch{epoch+1}.pt")
