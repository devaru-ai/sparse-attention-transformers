import torch
from model.transformer import Transformer
from model.utils import load_vocab, load_encoded, pad_batch

src_vocab = load_vocab("../data/processed/vocab.src")
tgt_vocab = load_vocab("../data/processed/vocab.tgt")
src_test = load_encoded("../data/processed/test.src")
tgt_test = load_encoded("../data/processed/test.tgt")
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(len(src_vocab), len(tgt_vocab), d_model=128, d_ff=256, num_heads=4, num_layers=2, max_len=128).to(device)
model.load_state_dict(torch.load("../outputs/checkpoints/model_epoch10.pt"))
model.eval()

batch_size = 32

def decode_sequence(seq):
    return " ".join([inv_tgt_vocab.get(idx, "<unk>") for idx in seq if idx > 2])

with torch.no_grad():
    for i, (src_b, tgt_b) in enumerate(zip(
        *[iter(src_test)] * batch_size, *[iter(tgt_test)] * batch_size)):
        src_batch = pad_batch(src_b)
        tgt_batch = pad_batch(tgt_b)
        src_batch = torch.tensor(src_batch, dtype=torch.long, device=device)
        tgt_batch = torch.tensor(tgt_batch, dtype=torch.long, device=device)
        dec_in = tgt_batch[:, :-1]
        logits = model(src_batch, dec_in)
        preds = logits.argmax(-1).cpu().tolist()
        for p in preds:
            print(decode_sequence(p))
        if i > 2: break  # Only show a few batches for test
