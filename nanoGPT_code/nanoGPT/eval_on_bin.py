"""
Evaluate model checkpoint on a .bin file and report average loss and perplexity.

The .bin file should contain pre-tokenized token IDs (numpy uint16 array or similar).
"""

import math
import os
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import tiktoken

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# model/load config (same pattern as eval.py)
init_from = 'resume'  # 'resume' or a GPT-2 variant (e.g. 'gpt2-medium')
out_dir = 'out'  # used when init_from == 'resume'
device = 'cuda'  # 'cpu', 'cuda', 'cuda:0', ...
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
seed = 1337

# data/eval config
input_file = 'data/rocstories/eval_stories.bin'  # path to .bin file
max_paragraphs = -1  # -1 means all
print_first_n = 3  # preview first N sections of loaded tokens

exec(open('configurator.py').read())  # allows overrides from CLI / config file
# -----------------------------------------------------------------------------


def load_bin_tokens(path):
    """Load pre-tokenized token IDs from a .bin file (raw binary uint16 format)."""
    # Load as raw binary uint16 array (produced by prepare.py with np.memmap)
    data = np.fromfile(path, dtype=np.uint16)
    return data.tolist()


torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(f"Unsupported init_from: {init_from}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# tokenizer (same behavior as eval.py)
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    encode = lambda s: [stoi[c] for c in s]
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

# Load tokens from bin file
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input .bin file not found: {input_file}")

token_ids = load_bin_tokens(input_file)
print(f"Loaded {len(token_ids)} tokens from {input_file}")

# Print first N preview sections
if print_first_n > 0:
    block_size = model.config.block_size
    for i in range(min(print_first_n, (len(token_ids) - 1) // block_size)):
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, len(token_ids))
        preview_tokens = token_ids[start_idx:end_idx]
        print(f"[preview {i}] {preview_tokens[:20]}... (length: {len(preview_tokens)})")

if len(token_ids) < 2:
    raise ValueError(f"Not enough tokens in {input_file} to evaluate (need at least 2)")

total_nll = 0.0
total_tokens = 0
block_size = model.config.block_size
pos = 0

with torch.no_grad():
    with ctx:
        while pos < len(token_ids) - 1:
            # Build a contiguous chunk and its shifted targets
            inp = token_ids[pos: pos + block_size]
            tgt = token_ids[pos + 1: pos + 1 + block_size]
            
            if len(tgt) == 0:
                break
            if len(inp) != len(tgt):
                inp = inp[:len(tgt)]

            x = torch.tensor(inp, dtype=torch.long, device=device)[None, :]
            y = torch.tensor(tgt, dtype=torch.long, device=device)[None, :]
            _, loss = model(x, y)  # mean CE over chunk tokens

            n_tok = len(tgt)
            total_nll += loss.item() * n_tok
            total_tokens += n_tok
            pos += n_tok

if total_tokens == 0:
    raise ValueError("No valid tokens to evaluate. Check your input .bin file.")

avg_loss = total_nll / total_tokens
ppl = math.exp(avg_loss)

print("----- Evaluation Results -----")
print(f"model       : {init_from}")
print(f"pred_tokens : {total_tokens}")
print(f"avg_loss    : {avg_loss:.3f}")
print(f"ppl         : {ppl:.2f}")