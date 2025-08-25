import sys
import os
import csv
import time
import random
import torch
import numpy as np

# BLEU & ROUGE integration
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from src.model.registry import get_model
from src.model.utils import load_vocab, load_encoded, pad_batch

nltk.download('punkt', quiet=True)  # Ensure NLTK data is available

# --------------- Reproducibility ---------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------- Accuracy, BLEU, ROUGE Evaluation ---------------
def decode_tokens(tokens, inv_vocab):
    return [inv_vocab.get(idx, "<unk>") for idx in tokens if idx > 2]

def evaluate_metrics(model, src_test, tgt_test, batch_size, device, inv_tgt_vocab):
    model.eval()
    total_correct, total_tokens = 0, 0
    all_preds, all_refs = [], []
    with torch.no_grad():
        for i in range(0, len(src_test), batch_size):
            src_batch = pad_batch(src_test[i:i+batch_size])
            tgt_batch = pad_batch(tgt_test[i:i+batch_size])
            src_tensor = torch.tensor(src_batch, dtype=torch.long, device=device)
            tgt_tensor = torch.tensor(tgt_batch, dtype=torch.long, device=device)
            dec_in = tgt_tensor[:, :-1]
            dec_target = tgt_tensor[:, 1:]
            logits = model(src_tensor, dec_in)
            preds = logits.argmax(-1).cpu().tolist()
            refs = dec_target.cpu().tolist()
            total_correct += (torch.tensor(preds) == torch.tensor(refs)).sum().item()
            total_tokens += torch.tensor(refs).numel()
            all_preds.extend([decode_tokens(p, inv_tgt_vocab) for p in preds])
            all_refs.extend([[decode_tokens(r, inv_tgt_vocab)] for r in refs])
    # BLEU
    smoother = SmoothingFunction().method4
    bleu = corpus_bleu(all_refs, all_preds, smoothing_function=smoother)
    # ROUGE
    pred_strs = [" ".join(p) for p in all_preds]
    ref_strs = [" ".join(r[0]) for r in all_refs]
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores, rougeL_scores = [], []
    for pred, ref in zip(pred_strs, ref_strs):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rougeL = np.mean(rougeL_scores)
    accuracy = total_correct / total_tokens if total_tokens else 0.0
    return accuracy, bleu, avg_rouge1, avg_rougeL

# --------------- Benchmark Runner ---------------
def benchmark_model(model_name, config, data_files, test_files, inv_tgt_vocab, seed, results_writer):
    print(f"Running benchmark for {model_name} | config: {config}")

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_vocab = load_vocab(data_files["src_vocab"])
    tgt_vocab = load_vocab(data_files["tgt_vocab"])
    src_train = load_encoded(data_files["src_train"])
    tgt_train = load_encoded(data_files["tgt_train"])

    model = get_model(
        model_name,
        len(src_vocab), len(tgt_vocab),
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_len=config["max_len"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    batch_size = config["batch_size"]

    # Memory & timing init
    torch.cuda.reset_peak_memory_stats(device=device) if torch.cuda.is_available() else None
    start_train = time.time()

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for i in range(0, len(src_train), batch_size):
            src_batch = pad_batch(src_train[i:i+batch_size])
            tgt_batch = pad_batch(tgt_train[i:i+batch_size])
            src_tensor = torch.tensor(src_batch, dtype=torch.long, device=device)
            tgt_tensor = torch.tensor(tgt_batch, dtype=torch.long, device=device)
            dec_in = tgt_tensor[:, :-1]
            dec_target = tgt_tensor[:, 1:]
            optimizer.zero_grad()
            logits = model(src_tensor, dec_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    train_time = time.time() - start_train
    peak_mem = torch.cuda.max_memory_allocated(device=device) if torch.cuda.is_available() else None
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Inference and metrics
    start_inf = time.time()
    model.eval()
    src_test = load_encoded(test_files["src_test"])
    tgt_test = load_encoded(test_files["tgt_test"])
    acc, bleu, rouge1, rougeL = evaluate_metrics(model, src_test, tgt_test, batch_size, device, inv_tgt_vocab)
    inf_time = time.time() - start_inf

    # Log results to CSV
    results_writer.writerow({
        "model": model_name,
        "config": str(config),
        "seed": seed,
        "params": param_count,
        "train_time_sec": round(train_time,2),
        "peak_mem_bytes": peak_mem,
        "inference_latency_sec": round(inf_time,2),
        "accuracy": round(acc, 5),
        "bleu": round(bleu, 4),
        "rouge1": round(rouge1, 4),
        "rougeL": round(rougeL, 4)
    })

def main():
    os.makedirs("benchmarks/results", exist_ok=True)
    configs = [
        dict(model="transformer", d_model=128, d_ff=256, num_heads=4, num_layers=2, max_len=128, batch_size=32, epochs=2, lr=1e-3),
        dict(model="transformer", d_model=128, d_ff=256, num_heads=4, num_layers=2, max_len=256, batch_size=32, epochs=2, lr=1e-3),
        dict(model="transformer", d_model=128, d_ff=256, num_heads=4, num_layers=2, max_len=128, batch_size=64, epochs=2, lr=1e-3),
        # Add configs for reformer, rev_att, etc.
    ]

    data_files = dict(
        src_vocab="data/processed/vocab.src",
        tgt_vocab="data/processed/vocab.tgt",
        src_train="data/processed/train_small.src",
        tgt_train="data/processed/train_small.tgt"
    )

    test_files = dict(
        src_test="data/processed/test.src",
        tgt_test="data/processed/test.tgt"
    )

    # Prepare accuracy translation dict
    tgt_vocab = load_vocab(data_files['tgt_vocab'])
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

    with open("benchmarks/results/benchmark.csv", "w", newline='') as csvfile:
        fieldnames = [
            "model", "config", "seed", "params", "train_time_sec", "peak_mem_bytes",
            "inference_latency_sec", "accuracy", "bleu", "rouge1", "rougeL"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for seed in [42, 123]:
            for config in configs:
                model_name = config["model"]
                config_for_model = {k: v for k, v in config.items() if k != "model"}
                benchmark_model(model_name, config_for_model, data_files, test_files, inv_tgt_vocab, seed, writer)

if __name__ == "__main__":
    main()
