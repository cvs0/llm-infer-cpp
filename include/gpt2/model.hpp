#pragma once

#include "gpt2/aligned.hpp"
#include "gpt2/thread_pool.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace gpt2 {

struct ModelConfig {
  int vocab_size = 0;
  int n_positions = 0;
  int n_embd = 0;
  int n_head = 0;
  int n_layer = 0;
  int n_inner = 0;
  float ln_eps = 1e-5f;

  int head_dim() const { return n_embd / n_head; }
};

struct ForwardConfig {
  bool simd = true;
};

// K and V for every layer. Layout of each tensor, row-major:
//   [n_layer, n_head, max_seq, head_dim]
// Position p of head h in layer L starts at
//   ((L * n_head + h) * max_seq + p) * head_dim.
class KVCache {
public:
  KVCache() = default;
  explicit KVCache(const ModelConfig& cfg);

  void clear() noexcept { len_ = 0; }
  int len() const noexcept { return len_; }
  int max_seq() const noexcept { return max_seq_; }

  float* k_at(int layer, int head, int pos) noexcept { return k_.data() + index(layer, head, pos); }
  float* v_at(int layer, int head, int pos) noexcept { return v_.data() + index(layer, head, pos); }
  const float* k_at(int layer, int head, int pos) const noexcept {
    return k_.data() + index(layer, head, pos);
  }
  const float* v_at(int layer, int head, int pos) const noexcept {
    return v_.data() + index(layer, head, pos);
  }

private:
  friend class Model;

  std::size_t index(int layer, int head, int pos) const noexcept {
    return (((static_cast<std::size_t>(layer) * static_cast<std::size_t>(n_head_) +
              static_cast<std::size_t>(head)) *
             static_cast<std::size_t>(max_seq_)) +
            static_cast<std::size_t>(pos)) *
           static_cast<std::size_t>(head_dim_);
  }

  void set_len(int n) noexcept { len_ = n; }

  int n_head_ = 0;
  int head_dim_ = 0;
  int max_seq_ = 0;
  int len_ = 0;
  AlignedFloats k_;
  AlignedFloats v_;
};

// One forward at a time. Workspace and the thread pool are internal mutable state.
class Model {
public:
  Model();
  ~Model();
  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  static Model load(const std::string& path);

  const ModelConfig& config() const noexcept { return cfg_; }
  std::int64_t parameter_count() const noexcept { return n_params_; }
  int threads() const noexcept;

  void set_threads(int nthreads);

  // Logits for the token that follows the last id in `tokens`.
  // `logits` holds vocab_size floats.
  // cache == nullptr: K/V are discarded after the call (positions restart at 0).
  // cache != nullptr: `tokens` are appended at cache->len(), which is the
  // absolute position of tokens[0]. cache->len() increases by n_tokens.
  void forward(const std::int32_t* tokens, int n_tokens, KVCache* cache, float* logits,
               const ForwardConfig& cfg);

  // Logits for every position. Shape [n_tokens, vocab_size], row-major.
  // Does not read or write `cache`. Positions are 0 .. n_tokens-1.
  void forward_all(const std::int32_t* tokens, int n_tokens, float* logits,
                   const ForwardConfig& cfg);

private:
  struct Block {
    const float* ln1_w = nullptr;
    const float* ln1_b = nullptr;
    const float* attn_w = nullptr;
    const float* attn_b = nullptr;
    const float* attn_proj_w = nullptr;
    const float* attn_proj_b = nullptr;
    const float* ln2_w = nullptr;
    const float* ln2_b = nullptr;
    const float* fc_w = nullptr;
    const float* fc_b = nullptr;
    const float* fc_proj_w = nullptr;
    const float* fc_proj_b = nullptr;
  };

  template <bool Simd>
  void forward_t(const std::int32_t* tokens, int n_tokens, KVCache* kv, int start, float* logits,
                 bool all_logits);

  ModelConfig cfg_{};
  std::int64_t n_params_ = 0;
  AlignedFloats storage_;
  const float* wte_ = nullptr;
  const float* wpe_ = nullptr;
  const float* ln_f_w_ = nullptr;
  const float* ln_f_b_ = nullptr;
  std::unique_ptr<Block[]> blocks_;
  AlignedFloats x_;
  AlignedFloats h_;
  AlignedFloats qkv_;
  AlignedFloats attn_;
  AlignedFloats mlp_;
  AlignedFloats scores_;
  KVCache scratch_;
  std::unique_ptr<ThreadPool> pool_;
};

} // namespace gpt2
