"""
prepare.py — ROCStories-style data preprocessor for nanoGPT
============================================================
Reads train.txt and test.txt from the same directory,
tokenizes with the GPT-2 BPE tokenizer (tiktoken),
and writes train.bin / val.bin / test.bin / meta.pkl.

Story format used (aligns with eval.py's _read_txt_paragraphs):
  - Each story is collapsed to a single line (internal whitespace → single space)
  - Each story is terminated with <|endoftext|>
  - Stories are joined with \n\n so there is one blank line between them

Split:
  - train.txt → 90 % train  /  10 % val   (story-level shuffle, seed 1337)
  - test.txt  → test (no further splitting)

Usage:
  cd data/rocmstories
  python prepare.py
"""

import os
import pickle
import random

import numpy as np
import tiktoken

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_SRC = os.path.join(DATA_DIR, "train.txt")
TEST_SRC  = os.path.join(DATA_DIR, "test.txt")

EOT = "<|endoftext|>"   # GPT-2 end-of-text special token
SPLIT_RATIO = 0.9       # fraction of train.txt stories used for training
RANDOM_SEED = 1337


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_stories(path: str) -> list[str]:
    """
    Read a .txt file and return a list of raw story strings.

    Primary split: paragraphs separated by blank lines (\n\n).
    Fallback: if only one paragraph is found, split on individual non-empty lines.
    Each candidate is stripped; empty results are discarded.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = content.split("\n\n")
    stories = [p.strip() for p in parts if p.strip()]

    # Fallback: single-paragraph file → treat each non-empty line as a story
    if len(stories) <= 1:
        stories = [ln.strip() for ln in content.splitlines() if ln.strip()]

    return stories


def normalize_story(story: str) -> str:
    """
    Collapse all internal whitespace (spaces, tabs, newlines) to a single space
    so that every story becomes exactly one line of text.
    """
    return " ".join(story.split())


def format_stories(stories: list[str]) -> str:
    """
    Turn a list of normalized stories into the final text corpus:
      <story1><|endoftext|>\n\n<story2><|endoftext|>\n\n...
    """
    return "\n\n".join(normalize_story(s) + EOT for s in stories)


# ---------------------------------------------------------------------------
# Encoding & writing
# ---------------------------------------------------------------------------

def encode_stories(stories: list[str], enc: tiktoken.Encoding) -> np.ndarray:
    """
    Encode a list of stories to a flat uint16 numpy array of token ids.
    The full corpus text is built first so that the \n\n separators between
    stories are tokenized in context (consistent with how eval.py reads the
    data at inference time).
    """
    corpus = format_stories(stories)
    ids = enc.encode(corpus, allowed_special={EOT})
    return np.array(ids, dtype=np.uint16)


def write_bin(ids: np.ndarray, path: str) -> None:
    """Write a uint16 numpy array to a binary file."""
    ids.tofile(path)
    print(f"  → wrote {len(ids):,} tokens to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- tokenizer -----------------------------------------------------------
    print("Loading GPT-2 tokenizer …")
    enc = tiktoken.get_encoding("gpt2")

    # --- read raw stories ----------------------------------------------------
    print(f"\nReading train source : {TRAIN_SRC}")
    all_train_stories = read_stories(TRAIN_SRC)
    print(f"  Found {len(all_train_stories):,} stories in train.txt")

    print(f"\nReading test source  : {TEST_SRC}")
    test_stories = read_stories(TEST_SRC)
    print(f"  Found {len(test_stories):,} stories in test.txt")

    # --- split train → train / val -------------------------------------------
    rng = random.Random(RANDOM_SEED)
    shuffled = list(all_train_stories)
    rng.shuffle(shuffled)

    split_idx   = int(len(shuffled) * SPLIT_RATIO)
    train_stories = shuffled[:split_idx]
    val_stories   = shuffled[split_idx:]

    print(f"\nSplit (seed={RANDOM_SEED}, ratio={SPLIT_RATIO}):")
    print(f"  train : {len(train_stories):,} stories")
    print(f"  val   : {len(val_stories):,} stories")
    print(f"  test  : {len(test_stories):,} stories")

    # --- encode --------------------------------------------------------------
    print("\nEncoding …")
    train_ids = encode_stories(train_stories, enc)
    val_ids   = encode_stories(val_stories,   enc)
    test_ids  = encode_stories(test_stories,  enc)

    print(f"  train tokens : {len(train_ids):,}")
    print(f"  val   tokens : {len(val_ids):,}")
    print(f"  test  tokens : {len(test_ids):,}")

    # --- write .bin files ----------------------------------------------------
    print("\nWriting binary files …")
    write_bin(train_ids, os.path.join(DATA_DIR, "train.bin"))
    write_bin(val_ids,   os.path.join(DATA_DIR, "val.bin"))
    write_bin(test_ids,  os.path.join(DATA_DIR, "test.bin"))

    # --- write meta.pkl ------------------------------------------------------
    meta = {
        "vocab_size" : enc.n_vocab,
        "tokenizer"  : "gpt2",
        "eot_token"  : EOT,
        "eot_token_id": enc.eot_token,
        "train_stories": len(train_stories),
        "val_stories"  : len(val_stories),
        "test_stories" : len(test_stories),
    }
    meta_path = os.path.join(DATA_DIR, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  → wrote meta.pkl  (vocab_size={meta['vocab_size']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
