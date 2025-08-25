import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.model.registry import get_model
from src.model.utils import load_vocab, load_encoded, pad_batch
import os   

# Ensure checkpoint directory exists
checkpoint_dir = "../outputs/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['transformer'], default='transformer')
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_ff', type=int, default=256)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

src_vocab = load_vocab("data/processed/vocab.src")
tgt_vocab = load_vocab("data/processed/vocab.tgt")
#src_train = load_encoded("data/processed/train.src")
#tgt_train = load_encoded("data/processed/train.tgt")
src_train = load_encoded("data/processed/train_small.src")
tgt_train = load_encoded("data/processed/train_small.tgt")


device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(
    args.model,
    len(src_vocab), len(tgt_vocab),
    d_model=args.d_model,
    d_ff=args.d_ff,
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    max_len=args.max_len
).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

def get_batches(src_list, tgt_list, batch_size):
    for i in range(0, len(src_list), batch_size):
        src = pad_batch(src_list[i:i+batch_size])
        tgt = pad_batch(tgt_list[i:i+batch_size])
        src = torch.tensor(src, dtype=torch.long, device=device)
        tgt = torch.tensor(tgt, dtype=torch.long, device=device)
        yield src, tgt

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for src_b, tgt_b in get_batches(src_train, tgt_train, args.batch_size):
        dec_in = tgt_b[:, :-1]
        dec_target = tgt_b[:, 1:]
        optimizer.zero_grad()
        logits = model(src_b, dec_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), f"../outputs/checkpoints/model_epoch{epoch+1}_{args.model}.pt")
