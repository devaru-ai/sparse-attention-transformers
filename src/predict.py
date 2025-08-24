import torch
from model.transformer import Transformer
from model.utils import load_vocab, pad_batch

src_vocab = load_vocab("../data/processed/vocab.src")
tgt_vocab = load_vocab("../data/processed/vocab.tgt")
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(len(src_vocab), len(tgt_vocab), d_model=128, d_ff=256, num_heads=4, num_layers=2, max_len=128).to(device)
model.load_state_dict(torch.load("../outputs/checkpoints/model_epoch10.pt"))
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
