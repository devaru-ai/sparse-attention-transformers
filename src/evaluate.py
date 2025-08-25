import argparse
import torch
import random
import numpy as np
from model.registry import get_model
from model.utils import load_vocab, load_encoded, pad_batch

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['transformer'], default='transformer')
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_ff', type=int, default=256)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out', type=str, default=None, help='File to write accuracy result (optional)')
args = parser.parse_args()

set_seed(args.seed)

src_vocab = load_vocab("../data/processed/vocab.src")
tgt_vocab = load_vocab("../data/processed/vocab.tgt")
src_test = load_encoded("../data/processed/test.src")
tgt_test = load_encoded("../data/processed/test.tgt")
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

# --- Accumulate predictions and references ---
total_correct, total_tokens = 0, 0
all_pred_texts = []
all_ref_texts = []

start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

smoother = SmoothingFunction().method4  # smooth BLEU for short sentences

def decode_tokens(tokens, inv_vocab):
    """Helper to convert token indices to string tokens."""
    return [inv_vocab.get(idx, "<unk>") for idx in tokens if idx > 2]

with torch.no_grad():
    if start_time is not None: start_time.record()
    for i in range(0, len(src_test), args.batch_size):
        src_batch = pad_batch(src_test[i:i+args.batch_size])
        tgt_batch = pad_batch(tgt_test[i:i+args.batch_size])
        src_tensor = torch.tensor(src_batch, dtype=torch.long, device=device)
        tgt_tensor = torch.tensor(tgt_batch, dtype=torch.long, device=device)
        dec_in = tgt_tensor[:, :-1]
        dec_target = tgt_tensor[:, 1:]
        logits = model(src_tensor, dec_in)
        preds = logits.argmax(-1).cpu().tolist()
        refs = dec_target.cpu().tolist()

        # Token-level accuracy
        total_correct += (torch.tensor(preds) == torch.tensor(refs)).sum().item()
        total_tokens += torch.tensor(refs).numel()

        # Decode sentences for BLEU/ROUGE
        all_pred_texts.extend([decode_tokens(p, inv_tgt_vocab) for p in preds])
        all_ref_texts.extend([[decode_tokens(r, inv_tgt_vocab)] for r in refs])
    if end_time is not None: end_time.record()

accuracy = total_correct / total_tokens if total_tokens else 0.0
print(f"Token-level Accuracy: {accuracy:.5f}")

if start_time is not None and end_time is not None:
    torch.cuda.synchronize()
    print(f"Inference time (ms): {start_time.elapsed_time(end_time)}")

# --- BLEU Score ---
bleu_score = corpus_bleu(all_ref_texts, all_pred_texts, smoothing_function=smoother)
print(f"Corpus BLEU: {bleu_score:.4f}")

# --- ROUGE Score ---
# Convert tokens lists to strings for ROUGE
pred_strs = [" ".join(p) for p in all_pred_texts]
ref_strs = [" ".join(r[0]) for r in all_ref_texts]

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge1_scores = []
rougeL_scores = []
for pred, ref in zip(pred_strs, ref_strs):
    scores = scorer.score(ref, pred)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)
avg_rouge1 = np.mean(rouge1_scores)
avg_rougeL = np.mean(rougeL_scores)
print(f"Avg ROUGE-1 F1: {avg_rouge1:.4f}")
print(f"Avg ROUGE-L F1: {avg_rougeL:.4f}")

if args.out:
    with open(args.out, "w") as f:
        f.write(f"{accuracy:.5f},{bleu_score:.4f},{avg_rouge1:.4f},{avg_rougeL:.4f}\n")
