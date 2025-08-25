import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.model.registry import get_model
from src.model.utils import load_vocab, load_encoded, pad_batch
import os
import time

# Ensure checkpoint directory exists
checkpoint_dir = "../outputs/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['transformer', 'reformer', 'reformer_pp'], default='transformer')
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_ff', type=int, default=256)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

print("="*50)
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"Model: {args.model.upper()}")
print(f"d_model: {args.d_model}, d_ff: {args.d_ff}, heads: {args.num_heads}, layers: {args.num_layers}")
print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")
print("="*50)

src_vocab = load_vocab("data/processed/vocab.src")
tgt_vocab = load_vocab("data/processed/vocab.tgt")
src_train = load_encoded("data/processed/train_small.src")
tgt_train = load_encoded("data/processed/train_small.tgt")

print(f"Loaded vocabs: src {len(src_vocab)}, tgt {len(tgt_vocab)}")
print(f"Loaded data lines: src {len(src_train)}, tgt {len(tgt_train)}")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Instantiating model...")
model = get_model(
    args.model,
    len(src_vocab), len(tgt_vocab),
    d_model=args.d_model,
    d_ff=args.d_ff,
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    max_len=args.max_len,
)
model = model.to(device)
param_total = sum(p.numel() for p in model.parameters())
print(model)
print(f"Total trainable parameters: {param_total}\n")

optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

def get_batches(src_list, tgt_list, batch_size):
    print("Batch generator activated.")
    for i in range(0, len(src_list), batch_size):
        src = pad_batch(src_list[i:i+batch_size])
        tgt = pad_batch(tgt_list[i:i+batch_size])
        src = torch.tensor(src, dtype=torch.long, device=device)
        tgt = torch.tensor(tgt, dtype=torch.long, device=device)
        print(f"Yielding batch {i//batch_size+1}: src {src.shape}, tgt {tgt.shape}")
        yield src, tgt

for epoch in range(args.epochs):
    print(f"\n============== EPOCH {epoch+1} / {args.epochs} ==============")
    model.train()
    total_loss = 0
    batch_i = 0
    t_epoch_start = time.time()
    for src_b, tgt_b in get_batches(src_train, tgt_train, args.batch_size):
        t_batch_start = time.time()
        print(f"\n[Batch {batch_i+1}] src: {src_b.shape}, tgt: {tgt_b.shape}")

        dec_in = tgt_b[:, :-1]
        dec_target = tgt_b[:, 1:]

        optimizer.zero_grad()
        print("  -> Forward pass...", flush=True)
        t0 = time.time()
        if args.model == "reformer_pp":
            logits, reg_loss = model(src_b, dec_in)
            print(f"       Done ({time.time()-t0:.2f}s) | logits: {logits.shape} | reg_loss: {getattr(reg_loss,'item',lambda:reg_loss)():.4f}", flush=True)
        else:
            logits = model(src_b, dec_in)
            reg_loss = torch.tensor(0.0, device=device)
            print(f"       Done ({time.time()-t0:.2f}s) | logits: {logits.shape}", flush=True)

        # SHAPE DEBUG AND SAFE LOSS
        print("Logits shape:", logits.shape, "dec_target shape:", dec_target.shape)
        # If shapes mismatched, forcibly slice both to minimum common length
        seq_len = min(logits.shape[1], dec_target.shape[1])
        if logits.shape[1] != dec_target.shape[1]:
            print(f"WARNING: Logits length {logits.shape[1]} does not match dec_target {dec_target.shape[1]}. Slicing both to {seq_len}.")
            logits = logits[:, :seq_len, :]
            dec_target = dec_target[:, :seq_len]

        print("  -> Loss computation...", flush=True)
        t0 = time.time()
        # Old code:
        # ce_loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
        # New safe code:
        ce_loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
        total_loss_val = ce_loss + reg_loss
        print(f"       CE: {ce_loss.item():.4f} | Reg: {reg_loss.item() if torch.is_tensor(reg_loss) else reg_loss:.4f} | Total: {total_loss_val.item():.4f} | ({time.time()-t0:.2f}s)", flush=True)

        print("  -> Backward...", flush=True)
        t0 = time.time()
        total_loss_val.backward()
        print(f"       Done ({time.time()-t0:.2f}s)", flush=True)

        max_grad = max((p.grad.abs().max().item() for p in model.parameters() if p.grad is not None), default=0.0)
        print(f"       Max grad in model: {max_grad:.6f}")

        print("  -> Optimizer step...", flush=True)
        t0 = time.time()
        optimizer.step()
        print(f"       Done ({time.time()-t0:.2f}s)", flush=True)

        total_loss += total_loss_val.item()
        batch_i += 1

        print(f"    [Batch {batch_i} DONE] Loss: {total_loss_val.item():.4f} | Batch time: {time.time()-t_batch_start:.2f}s", flush=True)

        # Optional debugging: break after first batch
        # if batch_i >= 1: break

    print(f"\nEpoch {epoch+1} total loss: {total_loss:.4f} | Time: {time.time()-t_epoch_start:.1f}s", flush=True)
    checkpoint_path = f"../outputs/checkpoints/model_epoch{epoch+1}_{args.model}.pt"
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(model.state_dict(), checkpoint_path)

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated(device=device)
        print(f"Epoch peak GPU memory usage: {mem//1024//1024:,} MB")

print("Training complete!")
