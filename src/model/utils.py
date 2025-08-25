# src/model/utils.py

import pickle

def load_vocab(path):
    """Loads vocabulary from a text or pickle file."""
    # Example for a simple text file: one token or word per line
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            token = line.strip()
            vocab[token] = idx
    return vocab

def load_encoded(path):
    """Loads encoded data from a file (expects list of lists)."""
    # Example: each line is a space-separated sequence of integers
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = [int(x) for x in line.strip().split()]
            data.append(tokens)
    return data

def pad_batch(batch, pad_token=0, max_len=None):
    """Pads a batch of sequences to the same length."""
    if max_len is None:
        max_len = max(len(seq) for seq in batch)
    padded = []
    for seq in batch:
        pad_length = max_len - len(seq)
        padded.append(seq + [pad_token] * pad_length)
    return padded
