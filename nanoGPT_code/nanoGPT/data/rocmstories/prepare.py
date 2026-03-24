"""
Prepare the ROCStories dataset for character-level language modeling.
Reads local train.txt and test.txt, then saves train.bin, val.bin and meta.pkl.
"""

import os
import pickle
import numpy as np

data_dir = os.path.dirname(__file__)

train_file_path = os.path.join(data_dir, 'train.txt')
test_file_path = os.path.join(data_dir, 'test.txt')

with open(train_file_path, 'r', encoding='utf-8') as f:
    train_text = f.read()

with open(test_file_path, 'r', encoding='utf-8') as f:
    test_text = f.read()

print(f"length of train set in characters: {len(train_text):,}")
print(f"length of test set in characters:  {len(test_text):,}")

# 用 train+test 一起建立字符表，避免测试集出现未见字符
data = train_text + test_text

chars = sorted(list(set(data)))
vocab_size = len(chars)

print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# 这里不再自己做 90/10 split
# 直接使用你已经准备好的:
# train.txt -> train.bin
# test.txt  -> val.bin
train_ids = encode(train_text)
val_ids = encode(test_text)

print(f"train has {len(train_ids):,} tokens")
print(f"val has   {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

# save meta information
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Done.")
print(f"Saved to {data_dir}/train.bin, {data_dir}/val.bin, {data_dir}/meta.pkl")