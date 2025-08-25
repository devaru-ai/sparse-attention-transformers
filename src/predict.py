import argparse
import torch
from model.registry import get_model
from model.utils import load_vocab, pad_batch

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['transformer'], default='transformer')
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_ff', type=int, default=256)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--checkpoint', type=str, required=True)
args = parser.parse_args()

src_vocab = load_vocab("../data/processed/vocab.src")
tgt_vocab = load_vocab("../data/processed/vocab.tgt")
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
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
model.load_state_dict(torch.load(args.checkpoint))
model.eval()

def predict(input_sentence):
    tokens = input_sentence.lower().strip().split()
    inp_ids = [src_vocab.get(tok, 0) for tok in tokens] + [2]
    inp_tensor = torch.tensor(pad_batch([inp_ids]), dtype=torch.long, device=device)
    tgt_in = torch.tensor([[1]], dtype=torch.long, device=device)  # <sos>
    output = []
    for _ in range(50):
        logits = model(inp_tensor, tgt_in)
        next_tok = logits[0, -1].argmax().item()
        if next_tok == 2:  # <eos>
            break
        output.append(next_tok)
        tgt_in = torch.cat([tgt_in, torch.tensor([[next_tok]], device=device)], dim=1)
    return " ".join([inv_tgt_vocab.get(idx, "<unk>") for idx in output])

while True:
    sent = input("Enter a sentence: ")
    print("Translation:", predict(sent))
