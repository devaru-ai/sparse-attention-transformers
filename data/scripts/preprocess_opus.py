import os
import re
from collections import Counter

raw_dir = 'data/raw'
processed_dir = 'data/processed'
os.makedirs(processed_dir, exist_ok=True)

def clean(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9äöüß .,!?'-]", " ", text)
    text = re.sub(" +", " ", text)
    return text

def tokenize(text):
    return text.split()

def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for line in sentences:
        counter.update(tokenize(line))
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab

def encode(sentence, vocab):
    tokens = tokenize(sentence)
    return [vocab.get(tok, vocab['<pad>']) for tok in tokens] + [vocab['<eos>']]

with open(os.path.join(raw_dir, "train.en"), encoding="utf8") as f:
    src_lines = [clean(line) for line in f]
with open(os.path.join(raw_dir, "train.de"), encoding="utf8") as f:
    tgt_lines = [clean(line) for line in f]

size = len(src_lines)
train_cut, valid_cut = int(0.9 * size), int(0.95 * size)
src_train, src_valid, src_test = src_lines[:train_cut], src_lines[train_cut:valid_cut], src_lines[valid_cut:]
tgt_train, tgt_valid, tgt_test = tgt_lines[:train_cut], tgt_lines[train_cut:valid_cut], tgt_lines[valid_cut:]

src_vocab = build_vocab(src_train)
tgt_vocab = build_vocab(tgt_train)

with open(os.path.join(processed_dir, "vocab.src"), "w", encoding="utf8") as f:
    for k, v in src_vocab.items(): f.write(f"{k}\t{v}\n")
with open(os.path.join(processed_dir, "vocab.tgt"), "w", encoding="utf8") as f:
    for k, v in tgt_vocab.items(): f.write(f"{k}\t{v}\n")

MAX_SEQ_LEN = 128  # or your preferred value

def save_encoded(data, vocab, path):
    with open(path, "w", encoding="utf8") as f:
        for sent in data:
            idxs = encode(sent, vocab)
            idxs = idxs[:MAX_SEQ_LEN]  # Truncate here
            f.write(" ".join(map(str, idxs)) + "\n")


save_encoded(src_train, src_vocab, os.path.join(processed_dir, "train.src"))
save_encoded(tgt_train, tgt_vocab, os.path.join(processed_dir, "train.tgt"))
save_encoded(src_valid, src_vocab, os.path.join(processed_dir, "valid.src"))
save_encoded(tgt_valid, tgt_vocab, os.path.join(processed_dir, "valid.tgt"))
save_encoded(src_test, src_vocab, os.path.join(processed_dir, "test.src"))
save_encoded(tgt_test, tgt_vocab, os.path.join(processed_dir, "test.tgt"))
