import os
from datasets import load_dataset

raw_dir = '/content/drive/MyDrive/transformer_project/data/raw'
os.makedirs(raw_dir, exist_ok=True)

dataset = load_dataset("Helsinki-NLP/opus_books", "de-en")
src_texts = [x['translation']['en'] for x in dataset['train']]
tgt_texts = [x['translation']['de'] for x in dataset['train']]

with open(os.path.join(raw_dir, "train.en"), "w", encoding="utf8") as f:
    f.write("\n".join(src_texts))
with open(os.path.join(raw_dir, "train.de"), "w", encoding="utf8") as f:
    f.write("\n".join(tgt_texts))
