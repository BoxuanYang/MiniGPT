"""
prepare.py — preprocess mixed_train.txt for nanoGPT
===================================================

Input:
  - mixed_train.txt
    Each story is separated by a blank line (\n\n)

Processing:
  - Read stories by paragraph
  - Normalize each story into a single line
  - Encode with GPT-2 tokenizer
  - For each story, append:
        <normalized story tokens> + <|endoftext|> + "\n\n"

Output:
  - train.bin
  - val.bin

Usage:
  python prepare.py
"""

import os
import random
import numpy as np
import tiktoken

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DATA_DIR, "mixed_train.txt")

TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN   = os.path.join(DATA_DIR, "val.bin")

SPLIT_RATIO = 0.9
RANDOM_SEED = 1337


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_stories(path: str) -> list[str]:
    """
    Read stories from a txt file.

    Primary behavior:
      split by blank lines (\n\n)

    Fallback:
      if only one paragraph is found, split by non-empty lines
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = content.split("\n\n")
    stories = [p.strip() for p in parts if p.strip()]

    # fallback: if the file was not really paragraph-separated
    if len(stories) <= 1:
        stories = [line.strip() for line in content.splitlines() if line.strip()]

    return stories


def normalize_story(story: str) -> str:
    """
    Collapse all internal whitespace into a single space,
    so each story becomes one clean line.
    """
    return " ".join(story.split())


def encode_stories(stories: list[str], enc: tiktoken.Encoding) -> np.ndarray:
    """
    Encode stories into one flat uint16 array.

    For each story:
      story_tokens + <|endoftext|> + "\n\n"
    """
    eot_id = enc.eot_token
    newline_ids = enc.encode("\n\n")

    ids: list[int] = []

    for story in stories:
        clean_story = normalize_story(story)
        if not clean_story:
            continue

        ids.extend(enc.encode(clean_story))
        ids.append(eot_id)
        ids.extend(newline_ids)

    return np.array(ids, dtype=np.uint16)


def write_bin(ids: np.ndarray, path: str) -> None:
    ids.tofile(path)
    print(f"  -> wrote {len(ids):,} tokens to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not os.path.exists(INPUT_FILE):
        print(f"Error: input file not found: {INPUT_FILE}")
        return

    print("Loading GPT-2 tokenizer...")
    enc = tiktoken.get_encoding("gpt2")

    print(f"\nReading stories from: {INPUT_FILE}")
    all_stories = read_stories(INPUT_FILE)
    print(f"Found {len(all_stories):,} stories")

    # shuffle + split
    rng = random.Random(RANDOM_SEED)
    shuffled = list(all_stories)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * SPLIT_RATIO)
    train_stories = shuffled[:split_idx]
    val_stories   = shuffled[split_idx:]

    print(f"\nSplit (seed={RANDOM_SEED}, ratio={SPLIT_RATIO}):")
    print(f"  train: {len(train_stories):,} stories")
    print(f"  val  : {len(val_stories):,} stories")

    # encode
    print("\nEncoding...")
    train_ids = encode_stories(train_stories, enc)
    val_ids   = encode_stories(val_stories, enc)

    print(f"  train tokens: {len(train_ids):,}")
    print(f"  val tokens  : {len(val_ids):,}")

    # write
    print("\nWriting .bin files...")
    write_bin(train_ids, TRAIN_BIN)
    write_bin(val_ids, VAL_BIN)

    print("\nDone.")


if __name__ == "__main__":
    main()