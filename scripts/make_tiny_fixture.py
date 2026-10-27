#!/usr/bin/env python3
"""Write a tiny GPT-2 weight file and reference logits.

The forward pass mirrors the scalar C++ kernels: float32 GEMM with the K
reduction ordered k = 0..K-1, float64 LayerNorm statistics, and the tanh GELU.
"""

import math
import pathlib
import struct

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "fixtures"

VOCAB = 40
N_POS = 16
N_EMBD = 32
N_HEAD = 4
N_LAYER = 2
N_INNER = 64
EPS = np.float32(1e-5)
TOKENS = [3, 7, 1, 8, 2]


def lcg_floats(n, state):
    out = np.empty(n, np.float32)
    for i in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = np.float32((state % 1000) / 1000.0 - 0.5) * np.float32(0.1)
    return out, state


def gemm_nn(A, B):
    M, K = A.shape
    N = B.shape[1]
    C = np.zeros((M, N), np.float32)
    for i in range(M):
        for k in range(K):
            a = A[i, k]
            for j in range(N):
                prod = np.float32(a * B[k, j])
                C[i, j] = np.float32(C[i, j] + prod)
    return C


def gemm_nt(A, B):
    M, K = A.shape
    N = B.shape[0]
    C = np.zeros((M, N), np.float32)
    for i in range(M):
        for j in range(N):
            s = np.float32(0)
            for k in range(K):
                s = np.float32(s + np.float32(A[i, k] * B[j, k]))
            C[i, j] = s
    return C


def layer_norm(x, w, b):
    xd = x.astype(np.float64)
    mean = float(xd.sum() / xd.shape[0])
    var = float(np.square(xd - mean).sum() / xd.shape[0])
    inv = np.float32(1.0 / math.sqrt(var + float(EPS)))
    y = np.empty_like(x)
    for i in range(x.shape[0]):
        d = np.float32(float(x[i]) - mean)
        y[i] = np.float32(np.float32(np.float32(d * inv) * w[i]) + b[i])
    return y


def gelu(x):
    k1 = np.float32(math.sqrt(2.0 / math.pi))
    y = np.empty_like(x)
    flat = x.reshape(-1)
    out = y.reshape(-1)
    for i in range(flat.shape[0]):
        v = flat[i]
        x3 = np.float32(np.float32(v * v) * v)
        u = np.float32(k1 * np.float32(v + np.float32(np.float32(0.044715) * x3)))
        th = np.float32(math.tanh(float(u)))
        out[i] = np.float32(np.float32(np.float32(0.5) * v) * np.float32(np.float32(1) + th))
    return y


def softmax(x):
    m = float(np.max(x))
    y = np.empty_like(x)
    s = np.float32(0)
    for i in range(x.shape[0]):
        y[i] = np.float32(math.exp(float(x[i]) - m))
        s = np.float32(s + y[i])
    inv = np.float32(np.float32(1) / s)
    for i in range(x.shape[0]):
        y[i] = np.float32(y[i] * inv)
    return y


def build_weights():
    c = N_EMBD
    inner = N_INNER
    per_layer = (c + c) + (c * 3 * c + 3 * c) + (c * c + c) + (c + c) + (c * inner + inner) + (
        inner * c + c
    )
    total = VOCAB * c + N_POS * c + per_layer * N_LAYER + c + c
    data, _ = lcg_floats(total, 42)
    return data


def split_weights(data):
    c = N_EMBD
    inner = N_INNER
    p = 0

    def take(n):
        nonlocal p
        sl = data[p : p + n]
        p += n
        return sl.copy()

    wte = take(VOCAB * c).reshape(VOCAB, c)
    wpe = take(N_POS * c).reshape(N_POS, c)
    layers = []
    for _ in range(N_LAYER):
        layer = {
            "ln1_w": take(c),
            "ln1_b": take(c),
            "attn_w": take(c * 3 * c).reshape(c, 3 * c),
            "attn_b": take(3 * c),
            "attn_proj_w": take(c * c).reshape(c, c),
            "attn_proj_b": take(c),
            "ln2_w": take(c),
            "ln2_b": take(c),
            "fc_w": take(c * inner).reshape(c, inner),
            "fc_b": take(inner),
            "fc_proj_w": take(inner * c).reshape(inner, c),
            "fc_proj_b": take(c),
        }
        layers.append(layer)
    ln_f_w = take(c)
    ln_f_b = take(c)
    if p != data.shape[0]:
        raise SystemExit(f"offset {p} != {data.shape[0]}")
    return wte, wpe, layers, ln_f_w, ln_f_b


def forward(tokens, wte, wpe, layers, ln_f_w, ln_f_b):
    n = len(tokens)
    c = N_EMBD
    h = N_HEAD
    d = c // h
    scale = np.float32(1.0 / math.sqrt(d))
    x = np.zeros((n, c), np.float32)
    for t, tok in enumerate(tokens):
        x[t] = wte[tok] + wpe[t]
    for layer in layers:
        hid = np.stack([layer_norm(x[t], layer["ln1_w"], layer["ln1_b"]) for t in range(n)])
        qkv = gemm_nn(hid, layer["attn_w"]) + layer["attn_b"]
        attn = np.zeros((n, c), np.float32)
        for t in range(n):
            for head in range(h):
                q = qkv[t, head * d : (head + 1) * d]
                scores = np.empty(t + 1, np.float32)
                for j in range(t + 1):
                    k = qkv[j, c + head * d : c + (head + 1) * d]
                    s = np.float32(0)
                    for u in range(d):
                        s = np.float32(s + np.float32(q[u] * k[u]))
                    scores[j] = np.float32(s * scale)
                probs = softmax(scores)
                o = np.zeros(d, np.float32)
                for j in range(t + 1):
                    v = qkv[j, 2 * c + head * d : 2 * c + (head + 1) * d]
                    for u in range(d):
                        o[u] = np.float32(o[u] + np.float32(probs[j] * v[u]))
                attn[t, head * d : (head + 1) * d] = o
        proj = gemm_nn(attn, layer["attn_proj_w"]) + layer["attn_proj_b"]
        x = x + proj
        hid = np.stack([layer_norm(x[t], layer["ln2_w"], layer["ln2_b"]) for t in range(n)])
        mlp = gemm_nn(hid, layer["fc_w"]) + layer["fc_b"]
        mlp = gelu(mlp)
        proj = gemm_nn(mlp, layer["fc_proj_w"]) + layer["fc_proj_b"]
        x = x + proj
    hid = np.stack([layer_norm(x[t], ln_f_w, ln_f_b) for t in range(n)])
    return gemm_nt(hid, wte)


def main():
    data = build_weights()
    wte, wpe, layers, ln_f_w, ln_f_b = split_weights(data)
    logits = forward(TOKENS, wte, wpe, layers, ln_f_w, ln_f_b)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "tiny_model.bin"
    ref_path = OUT_DIR / "tiny_logits.bin"
    with model_path.open("wb") as f:
        f.write(b"GPT2WGT1")
        f.write(struct.pack("<IIIIIII", 1, VOCAB, N_POS, N_EMBD, N_HEAD, N_LAYER, N_INNER))
        f.write(struct.pack("<f", float(EPS)))
        f.write(data.tobytes())
    with ref_path.open("wb") as f:
        f.write(struct.pack("<ii", len(TOKENS), VOCAB))
        f.write(np.asarray(TOKENS, np.int32).tobytes())
        f.write(np.ascontiguousarray(logits, np.float32).tobytes())
    print(f"wrote {model_path} ({model_path.stat().st_size} bytes)")
    print(f"wrote {ref_path} logits {logits.shape} max {float(np.max(logits)):.6f}")


if __name__ == "__main__":
    main()
