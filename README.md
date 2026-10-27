# gpt2infer

GPT-2 small inference on CPU. C++20. 124,439,808 parameters. No framework runtime.

## Architecture

Pre-norm transformer. `n_layer` blocks, then a final LayerNorm, then a tied output projection.

Each block:

1. `h = LayerNorm(x)`
2. `qkv = h @ W_attn + b_attn` with `W_attn` shaped `[n_embd, 3 * n_embd]`
3. Split the last axis into Q, K, V, each `[n_embd]`. Head `h` is the contiguous slice `[h * head_dim, (h + 1) * head_dim)`.
4. Causal attention. Scores are `dot(q, k) / sqrt(head_dim)` over keys `0 .. qpos` inclusive. Softmax, then the weighted sum of V.
5. `x = x + (attn @ W_proj + b_proj)`
6. `h = LayerNorm(x)`
7. `h = GELU(h @ W_fc + b_fc)`, then `x = x + (h @ W_fc_proj + b_fc_proj)`

`C = A @ W + b` is the Hugging Face Conv1D layout: `W` is `[K, N]`, row-major. The output projection is `h @ W_token^T`. `W_token` is `wte` and is not stored a second time.

GPT-2 small, read from the checkpoint config:

| field | value |
| --- | --- |
| vocab | 50257 |
| n_positions | 1024 |
| n_embd | 768 |
| n_head | 12 |
| head_dim | 64 |
| n_layer | 12 |
| n_inner | 3072 |
| ln_eps | 1e-5 |
| activation | tanh GELU (`gelu_new`) |

One `Model` owns the weight blob, the activation workspace, and a persistent thread pool. One forward at a time. The loader requires a little-endian host.

## Data layout

`models/model.bin`, little-endian. `scripts/convert_hf_gpt2.py` writes it. `h.*.attn.bias` in the safetensors file is the causal mask buffer and is not stored.

| offset | type | contents |
| --- | --- | --- |
| 0 | 8 bytes | magic `GPT2WGT1` |
| 8 | u32 | version `1` |
| 12 | i32 x 6 | vocab, n_positions, n_embd, n_head, n_layer, n_inner |
| 36 | f32 | ln_eps |
| 40 | f32 | `wte[vocab, n_embd]`, `wpe[n_positions, n_embd]`, then per layer the 12 tensors below, then `ln_f` weight and bias |

Per layer, in order: `ln1` weight, `ln1` bias, `attn` weight `[n_embd, 3 * n_embd]`, `attn` bias `[3 * n_embd]`, `attn` proj weight `[n_embd, n_embd]`, `attn` proj bias, `ln2` weight, `ln2` bias, `fc` weight `[n_embd, n_inner]`, `fc` bias, `fc` proj weight `[n_inner, n_embd]`, `fc` proj bias. All row-major.

Parameter count is the number of f32 values in that blob. `lm_head` is not added again. GPT-2 small is 124,439,808 f32 values, 497,759,232 bytes, file size 497,759,272 bytes.

`models/tokenizer.bin`:

| field | type |
| --- | --- |
| magic | 8 bytes `GPT2TOK1` |
| version | u32 `1` |
| n_vocab, n_merges | u32, u32 |
| tokens | `n_vocab` strings in id order, each `u32` byte length plus UTF-8 |
| merges | `n_merges` pairs in rank order, each pair two strings. Rank 0 is the first merge. |

Logit dumps (`--dump-logits` and `scripts/dump_reference_logits.py`): `i32 n_tokens`, `i32 vocab`, `i32 ids[n_tokens]`, `f32 logits[n_tokens, vocab]`.

KV cache, separate K and V, each row-major `[n_layer, n_head, max_seq, head_dim]`. Element `(layer, head, pos, d)` is at

```
((layer * n_head + head) * max_seq + pos) * head_dim + d
```

`max_seq` is `n_positions`. `cache->len()` is the next absolute position.

## Algorithms

LayerNorm: mean and population variance (divide by `n`) accumulated in float64. `y = (x - mean) * rsqrt(var + eps) * w + b` with `eps = 1e-5`.

GELU, in place:

```
0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

`sqrt(2/pi)` is computed in float64 and stored as float32.

Softmax: subtract the max, `exp`, normalize. Attention applies it only to the causal prefix, which matches a `-inf` mask on future keys.

Sampling: `temperature <= 0` is argmax, lowest index on ties. `top_k <= 0` keeps the full distribution. Otherwise the `k` largest logits are kept, then `softmax(logit / temperature)`, drawn with `mt19937`.

Generation: prefill the prompt, sample, then forward one new token at a time. The last sampled id is not forwarded. `prompt + new` must be `<= n_positions`. Empty text is rejected. `--bos` starts from the `<|endoftext|>` id. `--no-kv` recomputes K and V for the whole sequence at every step. Within one forward, K and V are still written once.

Tokenizer: GPT-2 byte-level BPE.

1. If `<|endoftext|>` is in the vocab, exact occurrences are emitted as that id and are not split. In the GPT-2 vocab that id is 50256.
2. Remaining spans are split by the GPT-2 regex, in priority order:

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

` ?` is one ASCII space. `\p{L}`, `\p{N}`, and `\s` are the `regex` module classes (tables generated with regex 2.5.123, unicodedata 15.0.0). Invalid UTF-8 throws.

3. Each byte maps to a Unicode codepoint (`bytes_to_unicode`). Printable bytes in the original set are identity. Byte 32 maps to 288. Other bytes map to 256, 257, ... in ascending byte order.
4. BPE: repeatedly merge the adjacent pair with the lowest rank. Every non-overlapping occurrence of that pair is merged in one left-to-right pass, then ranks are recomputed. Pairs absent from the merge list are never merged.

`decode` inverts the byte map and checks UTF-8.

Per decode token at context length `t` (the new token occupies position `t - 1`), multiply-adds:

| term | count |
| --- | --- |
| QKV, per layer | `3 * n_embd^2` |
| attention proj, per layer | `n_embd^2` |
| MLP, per layer | `2 * n_embd * n_inner` |
| attention dots and axpy, per layer | `2 * n_embd * t` |
| output projection | `vocab * n_embd` |

GPT-2 small, GEMM only: `12 * (4 * 768^2 + 2 * 768 * 3072) + 50257 * 768 = 123,532,032` multiply-adds. Attention adds `18,432 * t` multiply-adds.

With a KV cache, a prompt of `P` tokens plus `T` new tokens evaluates `P` rows in the prefill and `T - 1` rows of length 1. GEMM work is `O(P + T)`. Attention work is `O(P^2 + T * (P + T))`.

Without a KV cache the forwarded lengths are `P, P+1, ..., P+T-1`. GEMM work is `O((P+T)^2)`. Attention work is `O((P+T)^3)`.

GEMM runs on more than one thread when `M * N * K >= 1,000,000`. At `M = 1` the attention projection is `768 * 768 = 589,824` multiply-adds, which is below that threshold. QKV (`1,769,472`), both MLP products (`2,359,296` each), and the output projection (`38,597,376`) are above it.

## Memory

| region | bytes |
| --- | --- |
| weight blob | 497,759,232 |
| KV cache at `n_positions` | `2 * 12 * 12 * 1024 * 64 * 4 = 75,497,472` |
| activation workspace | `n_positions * (6 * n_embd + n_inner + 1) * 4 = 31,461,376` |

Workspace buffers, each 64-byte aligned: residual `x` and LayerNorm output `h` are `[n_positions, n_embd]`, QKV is `[n_positions, 3 * n_embd]`, attention output is `[n_positions, n_embd]`, MLP is `[n_positions, n_inner]`, scores are `[n_positions]`.

A decode step reads every parameter except the unused rows of `wpe`. One position row of `wpe` is 3,072 bytes, so 1023 rows stay unread: 494,616,576 bytes of weights move per decode token. The output projection is `50257 * 768 * 4 = 154,389,504` of those bytes.

## Kernels

`gemm_nn`: `C[m, n] = sum_k A[m, k] * B[k, n]`. `gemm_nt`: `C[m, n] = sum_k A[m, k] * B[n, k]`. Leading dimensions `lda`, `ldb`, `ldc`. `C` is overwritten.

Global compile flag `-ffp-contract=off`, so the scalar kernel is a multiply followed by an add.

Scalar `gemm_nn` is `i-k-j`: zero the row, then for each `k` add `A[i, k] * B[k, :]`. `src/gemm_scalar.cpp` is built with `-fno-tree-vectorize -fno-tree-slp-vectorize` (GCC) or `-fno-vectorize -fno-slp-vectorize` (Clang), plus a Clang loop pragma that disables vectorization. Scalar `gemm_nt` is a sequential dot per output.

AVX2 is compiled only into `src/gemm_avx2.cpp` (`-mavx2 -mfma`). It is selected at runtime when `__builtin_cpu_supports("avx2")` and `__builtin_cpu_supports("fma")` are true and the caller passes `simd`. `--no-simd` forces the scalar kernels. AVX-512 is not used.

AVX2 `gemm_nn`: `j` is the outer loop so a B panel is reused across rows. Panels are 32 floats, then 8, then a scalar tail. Each panel is four or one `vfmadd` chains of broadcast `A[i, k]` against unaligned `B`.

AVX2 `gemm_nt`: four outputs at a time, `k` step 8, horizontal sum, scalar tail. This is the output-projection kernel.

Attention dots use two `__m256` accumulators, `k` step 16. `head_dim` is 64, so the tail is empty. `axpy` is 8-wide FMA.

Threads split columns. AVX2 splits are multiples of 32 floats, so the panel grouping matches the single-thread kernel. Each output element is produced by one thread. One thread and `N` threads are bit-identical. The pool is not re-entrant.

## Numerical tolerances

Compared with `transformers` `GPT2LMHeadModel` fp32, prompt `The theory of computation begins with`, token ids `[464, 4583, 286, 29964, 6140, 351]`, `forward_all`, 1 thread:

| kernel | max abs logit error | argmax mismatches |
| --- | --- | --- |
| scalar | 3.12805e-4 | 0 / 6 |
| AVX2 | 2.44141e-4 | 0 / 6 |

`tests/test_hf_logits.cpp` requires max abs `< 1e-3` and zero argmax mismatches. The test runs only when `GPT2_MODEL` and `GPT2_LOGITS` are set. Optional `GPT2_TOKENIZER` and `GPT2_PROMPT` check `encode`.

`tests/fixtures/tiny_model.bin` is an independent float32 reference (vocab 40, context 16, `n_embd` 32, 4 heads, 2 layers, `n_inner` 64, tokens `[3, 7, 1, 8, 2]`, LCG seed 42). The scalar forward must be within `2e-4`. AVX2 must be within `2e-3`. LayerNorm on a 4-vector matches a float64 evaluation exactly (`EXPECT_FLOAT_EQ`). GELU(1.5) is within `2e-6` of the float64 tanh formula. GELU(0) is 0.

AVX2 uses FMA, so it is not bit-identical to the scalar kernel or to PyTorch. Column-parallel AVX2 is bit-identical to single-thread AVX2.

## Build

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

C++20. `-Wall -Wextra -Wpedantic -Wshadow -Werror` on the library and `gpt2_infer`. GoogleTest 1.15.2 is fetched at configure time. `GPT2_SANITIZERS=ON` adds AddressSanitizer and UndefinedBehaviorSanitizer.

CI (`.github/workflows/ci.yml`): GCC and Clang, `Release` plus `ctest`; GCC and Clang, `RelWithDebInfo` with ASan and UBSan. The checkpoint is not downloaded in CI, so `HfLogits.MatchesTransformers` is skipped there.

```
python3 -m pip install -r scripts/requirements.txt
python3 scripts/convert_hf_gpt2.py --out-dir models
python3 scripts/dump_reference_logits.py \
  --model models/hf \
  --prompt "The theory of computation begins with" \
  --out models/ref_logits.bin
```

`dump_reference_logits.py` needs PyTorch and `transformers` (`GPT2Tokenizer`, `use_fast=False`). `models/` and `*.safetensors` are gitignored.

```
GPT2_MODEL=models/model.bin \
GPT2_LOGITS=models/ref_logits.bin \
GPT2_TOKENIZER=models/tokenizer.bin \
GPT2_PROMPT="The theory of computation begins with" \
  ctest --test-dir build -R HfLogits --output-on-failure
```

## Run

```
./build/gpt2_infer \
  --model models/model.bin \
  --tokenizer models/tokenizer.bin \
  --prompt "The theory of computation begins with" \
  --max-tokens 32 --temperature 0 --threads 4
```

`--temperature 0` is greedy. `--top-k 0` disables the filter. Text goes to stdout. Timing goes to stderr and includes the prefill. `tok/s = new_tokens / seconds`. Encode and decode are outside the timer.

Measured greedy continuation, same command, g++ 13.3.0 `Release`:

```
The theory of computation begins with the idea that the number of possible values of a given value is the sum of all possible values of that value. The number of possible values of a given value
```

```
prompt_tokens=6 new_tokens=32 seconds=0.80941 tok_s=39.5349 simd=1 kv=1 threads=4
```

`--bench` runs the four `kv` by `simd` configurations. Each configuration has one untimed warmup, then `--bench-runs` timed runs (default 3). The reported median is the middle `tok/s` of those runs. Temperature is forced to 0.

## Benchmarks

Host: KVM guest. `lscpu` model name `Intel(R) Xeon(R) Processor`, family 6, model 207, stepping 2. 4 cores, 1 thread per core, `cpu MHz` 2400.000. L1d 192 KiB (4 instances), L1i 128 KiB (4), L2 8 MiB (4), L3 320 MiB (1). Flags include AVX2, FMA, and AVX-512. The kernels use AVX2 and FMA only.

Compiler: `g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`. `CMAKE_BUILD_TYPE=Release` (`-O3 -DNDEBUG`), plus `-ffp-contract=off`, and `-mavx2 -mfma` on `src/gemm_avx2.cpp`.

Prompt `The theory of computation begins with` (6 tokens). 32 new tokens. 4 threads. Temperature 0. One untimed warmup per cell, then 3 timed runs. `tok/s = 32 / seconds`, prefill included. Seconds and tok/s are in run order. The median is the middle tok/s after sorting.

| kv | simd | seconds | tok/s | median tok/s |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0.579627, 0.547939, 0.600004 | 55.2079, 58.4006, 53.333 | 55.2079 |
| 1 | 0 | 1.71983, 1.66411, 1.87367 | 18.6065, 19.2295, 17.0788 | 18.6065 |
| 0 | 1 | 3.52365, 3.5109, 3.54395 | 9.08149, 9.11447, 9.02946 | 9.08149 |
| 0 | 0 | 10.3869, 10.046, 10.4642 | 3.08079, 3.18535, 3.05804 | 3.08079 |
