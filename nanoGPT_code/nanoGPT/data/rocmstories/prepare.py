"""Prepare ROCStories with GPT-2 tokenization for nanoGPT.

Reads local train.txt and test.txt (one story per line).
train.txt is split 90/10 into train.bin and val.bin.
test.txt is saved as test.bin.

Outputs:
- train.bin
- val.bin
- test.bin
- meta.pkl
"""

import os
import pickle

import numpy as np
import tiktoken


OUT_DIR = os.path.dirname(__file__)


def read_stories(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def encode_stories(stories, split_name, enc, eot):
    print(f"Encoding {split_name} ({len(stories):,} stories)...")
    ids = []
    for i, story in enumerate(stories):
        story_ids = enc.encode_ordinary(story)
        story_ids.append(eot)
        ids.extend(story_ids)
        if (i + 1) % 10000 == 0:
            print(f"  {split_name}: {i + 1:,} stories processed")
    return np.array(ids, dtype=np.uint16)


def main():
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    # --- train.txt -> train.bin + val.bin (90/10 split) ---
    train_txt = os.path.join(OUT_DIR, "train.txt")
    all_train = read_stories(train_txt)
    split_at = int(len(all_train) * 0.9)
    train_stories = all_train[:split_at]
    val_stories = all_train[split_at:]
    print(f"train.txt: {len(all_train):,} stories -> train {len(train_stories):,} / val {len(val_stories):,}")

    train_ids = encode_stories(train_stories, "train", enc, eot)
    val_ids = encode_stories(val_stories, "val", enc, eot)

    # --- test.txt -> test.bin ---
    test_txt = os.path.join(OUT_DIR, "test.txt")
    test_stories = read_stories(test_txt)
    print(f"test.txt: {len(test_stories):,} stories")
    test_ids = encode_stories(test_stories, "test", enc, eot)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val   has {len(val_ids):,} tokens")
    print(f"test  has {len(test_ids):,} tokens")

    train_ids.tofile(os.path.join(OUT_DIR, "train.bin"))
    val_ids.tofile(os.path.join(OUT_DIR, "val.bin"))
    test_ids.tofile(os.path.join(OUT_DIR, "test.bin"))

    meta = {
        "vocab_size": enc.n_vocab,
        "tokenizer": "gpt2",
        "eot_token": eot,
    }
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("Done. Saved train.bin, val.bin, test.bin, meta.pkl")


if __name__ == "__main__":
    main()