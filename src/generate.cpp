#include "gpt2/generate.hpp"

#include "gpt2/sampler.hpp"

#include <chrono>
#include <random>
#include <stdexcept>
#include <utility>

namespace gpt2 {

GenerateOutput generate(Model& model, const std::vector<std::int32_t>& prompt,
                        const GenerateConfig& cfg) {
  if (prompt.empty())
    throw std::runtime_error("empty prompt");
  if (cfg.max_new_tokens < 0)
    throw std::runtime_error("max_new_tokens < 0");
  const int n_positions = model.config().n_positions;
  if (static_cast<int>(prompt.size()) > n_positions) {
    throw std::runtime_error("prompt exceeds context length");
  }
  if (static_cast<std::int64_t>(prompt.size()) + cfg.max_new_tokens > n_positions) {
    throw std::runtime_error("generation exceeds context length");
  }
  if (cfg.threads > 0)
    model.set_threads(cfg.threads);

  GenerateOutput out;
  out.prompt_tokens = static_cast<int>(prompt.size());
  out.tokens = prompt;
  out.new_tokens = cfg.max_new_tokens;

  std::mt19937 rng(cfg.seed);
  ForwardConfig fwd;
  fwd.simd = cfg.simd;
  std::vector<float> logits(static_cast<std::size_t>(model.config().vocab_size));

  const auto t0 = std::chrono::steady_clock::now();
  if (cfg.use_kv_cache) {
    KVCache cache(model.config());
    model.forward(out.tokens.data(), out.prompt_tokens, &cache, logits.data(), fwd);
    for (int i = 0; i < cfg.max_new_tokens; ++i) {
      const std::int32_t next = static_cast<std::int32_t>(
          sample_token(logits.data(), model.config().vocab_size, cfg.temperature, cfg.top_k, rng));
      out.tokens.push_back(next);
      if (i + 1 == cfg.max_new_tokens)
        break;
      model.forward(&out.tokens.back(), 1, &cache, logits.data(), fwd);
    }
  } else {
    model.forward(out.tokens.data(), static_cast<int>(out.tokens.size()), nullptr, logits.data(),
                  fwd);
    for (int i = 0; i < cfg.max_new_tokens; ++i) {
      const std::int32_t next = static_cast<std::int32_t>(
          sample_token(logits.data(), model.config().vocab_size, cfg.temperature, cfg.top_k, rng));
      out.tokens.push_back(next);
      if (i + 1 == cfg.max_new_tokens)
        break;
      model.forward(out.tokens.data(), static_cast<int>(out.tokens.size()), nullptr, logits.data(),
                    fwd);
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  out.seconds = std::chrono::duration<double>(t1 - t0).count();
  return out;
}

} // namespace gpt2
