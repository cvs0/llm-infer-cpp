#!/usr/bin/env python3
"""Download GPT-2 small and write model.bin plus tokenizer.bin.

Weight layout is little-endian float32, magic GPT2WGT1. Matmul weights stay in
Hugging Face Conv1D order [K, N]. The output projection is tied to wte and is
not stored twice. Tokenizer magic is GPT2TOK1.
"""

import argparse
import json
import pathlib
import struct
import urllib.request

import numpy as np

HF = "https://huggingface.co/gpt2/resolve/main"
FILES = ["config.json", "model.safetensors", "vocab.json", "merges.txt"]


def download(url, dest: pathlib.Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"get {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "gpt2infer/1.0"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")


def load_safetensors(path: pathlib.Path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if info["dtype"] != "F32":
                raise SystemExit(f"{name} dtype {info['dtype']} is not F32")
            off0, off1 = info["data_offsets"]
            f.seek(base + off0)
            raw = f.read(off1 - off0)
            arr = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])
            tensors[name] = np.ascontiguousarray(arr)
        return tensors


def write_u32(f, v):
    f.write(struct.pack("<I", v))


def write_str(f, s: str):
    b = s.encode("utf-8")
    write_u32(f, len(b))
    f.write(b)


def convert_tokenizer(vocab_path, merges_path, dest):
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    n = max(vocab.values()) + 1
    id_to_tok = [None] * n
    for token, idx in vocab.items():
        if id_to_tok[idx] is not None:
            raise SystemExit(f"duplicate id {idx}")
        id_to_tok[idx] = token
    if any(t is None for t in id_to_tok):
        raise SystemExit("vocab ids are not dense")
    merges = []
    for line in merges_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SystemExit(f"bad merge line {line!r}")
        merges.append((parts[0], parts[1]))
    with dest.open("wb") as f:
        f.write(b"GPT2TOK1")
        write_u32(f, 1)
        write_u32(f, n)
        write_u32(f, len(merges))
        for token in id_to_tok:
            write_str(f, token)
        for left, right in merges:
            write_str(f, left)
            write_str(f, right)
    print(f"tokenizer vocab={n} merges={len(merges)} -> {dest}")


def convert_model(config_path, tensors, dest):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    act = cfg.get("activation_function", "gelu_new")
    if act not in ("gelu_new", "gelu_pytorch_tanh"):
        raise SystemExit(f"unsupported activation {act}")
    n_embd = int(cfg["n_embd"])
    n_head = int(cfg["n_head"])
    n_layer = int(cfg["n_layer"])
    n_positions = int(cfg["n_positions"])
    vocab = int(cfg["vocab_size"])
    n_inner = int(cfg.get("n_inner") or (4 * n_embd))
    eps = float(cfg.get("layer_norm_epsilon", 1e-5))

    # Official gpt2 safetensors stores GPT2Model keys (wte.weight, h.0....).
    # A GPT2LMHeadModel state dict prefixes those with "transformer.".
    # h.*.attn.bias is the causal mask buffer, not a parameter.
    def resolve(name):
        if name in tensors:
            return name
        if name.startswith("transformer."):
            alt = name[len("transformer.") :]
        else:
            alt = "transformer." + name
        if alt in tensors:
            return alt
        raise SystemExit(f"missing tensor {name}")

    def get(name, shape):
        key = resolve(name)
        arr = tensors[key]
        if tuple(arr.shape) != tuple(shape):
            raise SystemExit(f"{key} shape {arr.shape} != {shape}")
        return arr

    wte = get("wte.weight", (vocab, n_embd))
    wpe = get("wpe.weight", (n_positions, n_embd))
    for head_name in ("lm_head.weight", "transformer.lm_head.weight"):
        if head_name in tensors and not np.array_equal(tensors[head_name], wte):
            raise SystemExit("lm_head.weight is not tied to wte")

    layers = []
    for i in range(n_layer):
        p = f"h.{i}"
        layers.append(
            (
                get(f"{p}.ln_1.weight", (n_embd,)),
                get(f"{p}.ln_1.bias", (n_embd,)),
                get(f"{p}.attn.c_attn.weight", (n_embd, 3 * n_embd)),
                get(f"{p}.attn.c_attn.bias", (3 * n_embd,)),
                get(f"{p}.attn.c_proj.weight", (n_embd, n_embd)),
                get(f"{p}.attn.c_proj.bias", (n_embd,)),
                get(f"{p}.ln_2.weight", (n_embd,)),
                get(f"{p}.ln_2.bias", (n_embd,)),
                get(f"{p}.mlp.c_fc.weight", (n_embd, n_inner)),
                get(f"{p}.mlp.c_fc.bias", (n_inner,)),
                get(f"{p}.mlp.c_proj.weight", (n_inner, n_embd)),
                get(f"{p}.mlp.c_proj.bias", (n_embd,)),
            )
        )
    ln_f_w = get("ln_f.weight", (n_embd,))
    ln_f_b = get("ln_f.bias", (n_embd,))

    params = wte.size + wpe.size + ln_f_w.size + ln_f_b.size
    for layer in layers:
        params += sum(t.size for t in layer)

    with dest.open("wb") as f:
        f.write(b"GPT2WGT1")
        f.write(struct.pack("<IIIIIII", 1, vocab, n_positions, n_embd, n_head, n_layer, n_inner))
        f.write(struct.pack("<f", eps))
        f.write(np.ascontiguousarray(wte).tobytes())
        f.write(np.ascontiguousarray(wpe).tobytes())
        for layer in layers:
            for t in layer:
                f.write(np.ascontiguousarray(t).tobytes())
        f.write(np.ascontiguousarray(ln_f_w).tobytes())
        f.write(np.ascontiguousarray(ln_f_b).tobytes())
    print(f"model parameters={params} bytes={dest.stat().st_size} -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("models"))
    args = parser.parse_args()
    hf_dir = args.out_dir / "hf"
    for name in FILES:
        download(f"{HF}/{name}", hf_dir / name)
    tensors = load_safetensors(hf_dir / "model.safetensors")
    convert_model(hf_dir / "config.json", tensors, args.out_dir / "model.bin")
    convert_tokenizer(hf_dir / "vocab.json", hf_dir / "merges.txt", args.out_dir / "tokenizer.bin")


if __name__ == "__main__":
    main()
