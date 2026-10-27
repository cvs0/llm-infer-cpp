#!/usr/bin/env python3
"""Dump Hugging Face GPT-2 logits for a prompt.

File layout, little-endian:
  int32 n_tokens
  int32 vocab
  int32 token_ids[n_tokens]
  float32 logits[n_tokens, vocab]
"""

import argparse
import pathlib
import struct

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default="The theory of computation begins with")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    tok = GPT2Tokenizer.from_pretrained(args.model, use_fast=False)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.eval()
    ids = tok.encode(args.prompt, add_special_tokens=False)
    if not ids:
        raise SystemExit("prompt produced no tokens")
    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long)).logits[0].float().cpu().numpy()
    logits = np.ascontiguousarray(logits, dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(struct.pack("<ii", len(ids), logits.shape[1]))
        f.write(np.asarray(ids, dtype=np.int32).tobytes())
        f.write(logits.tobytes())
    print("prompt", args.prompt)
    print("ids", ids)
    print("wrote", args.out, "shape", tuple(logits.shape))


if __name__ == "__main__":
    main()
