#pragma once

#include "gpt2/model.hpp"

#include <cstdint>
#include <vector>

namespace gpt2 {

struct GenerateConfig {
  int max_new_tokens = 32;
  float temperature = 0.f;
  int top_k = 0;
  std::uint32_t seed = 0;
  bool simd = true;
  bool use_kv_cache = true;
  int threads = 0; // 0 keeps the model's current pool size
};

struct GenerateOutput {
  std::vector<std::int32_t> tokens;
  int prompt_tokens = 0;
  int new_tokens = 0;
  double seconds = 0.0;
};

// Appends up to max_new_tokens ids. The timed region is the forward/sample loop,
// including the prompt prefill and excluding tokenizer encode/decode.
// The final sampled id is not forwarded. Sequence length is prompt + new tokens
// and must be <= n_positions.
GenerateOutput generate(Model& model, const std::vector<std::int32_t>& prompt,
                        const GenerateConfig& cfg);

} // namespace gpt2
