#include "gpt2/gemm.hpp"
#include "gpt2/model.hpp"
#include "gpt2/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

std::int32_t read_i32(std::istream& in) {
  unsigned char b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  if (in.gcount() != 4)
    throw std::runtime_error("truncated reference");
  return static_cast<std::int32_t>(
      static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
      (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24));
}

const char* env_or_null(const char* key) {
  const char* v = std::getenv(key);
  if (v == nullptr || v[0] == '\0')
    return nullptr;
  return v;
}

} // namespace

// Compares forward_all logits to a file written by scripts/dump_reference_logits.py.
// Set GPT2_MODEL and GPT2_LOGITS. Optional GPT2_TOKENIZER and GPT2_PROMPT check encode().
TEST(HfLogits, MatchesTransformers) {
  const char* model_path = env_or_null("GPT2_MODEL");
  const char* logits_path = env_or_null("GPT2_LOGITS");
  if (model_path == nullptr || logits_path == nullptr) {
    GTEST_SKIP() << "set GPT2_MODEL and GPT2_LOGITS";
  }

  gpt2::Model model = gpt2::Model::load(model_path);
  std::ifstream in(logits_path, std::ios::binary);
  ASSERT_TRUE(in) << logits_path;
  const std::int32_t n = read_i32(in);
  const std::int32_t vocab = read_i32(in);
  ASSERT_EQ(vocab, model.config().vocab_size);
  ASSERT_GT(n, 0);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    ids[static_cast<std::size_t>(i)] = read_i32(in);

  const char* tok_path = env_or_null("GPT2_TOKENIZER");
  const char* prompt = env_or_null("GPT2_PROMPT");
  if (tok_path != nullptr && prompt != nullptr) {
    gpt2::Tokenizer tok = gpt2::Tokenizer::load(tok_path);
    const std::vector<std::int32_t> enc = tok.encode(prompt);
    EXPECT_EQ(enc, ids);
  }

  std::vector<float> ref(static_cast<std::size_t>(n) * static_cast<std::size_t>(vocab));
  in.read(reinterpret_cast<char*>(ref.data()),
          static_cast<std::streamsize>(ref.size() * sizeof(float)));
  ASSERT_TRUE(in);

  auto score = [&](bool simd) {
    std::vector<float> got(ref.size());
    gpt2::ForwardConfig fwd;
    fwd.simd = simd;
    model.set_threads(1);
    model.forward_all(ids.data(), n, got.data(), fwd);
    float max_abs = 0.f;
    int argmax_miss = 0;
    for (int t = 0; t < n; ++t) {
      int ref_arg = 0;
      int got_arg = 0;
      const std::size_t base = static_cast<std::size_t>(t) * static_cast<std::size_t>(vocab);
      for (int v = 0; v < vocab; ++v) {
        const float d = std::fabs(ref[base + static_cast<std::size_t>(v)] -
                                  got[base + static_cast<std::size_t>(v)]);
        max_abs = std::max(max_abs, d);
        if (ref[base + static_cast<std::size_t>(v)] > ref[base + static_cast<std::size_t>(ref_arg)])
          ref_arg = v;
        if (got[base + static_cast<std::size_t>(v)] > got[base + static_cast<std::size_t>(got_arg)])
          got_arg = v;
      }
      if (ref_arg != got_arg)
        ++argmax_miss;
    }
    std::cout << "hf_max_abs simd=" << simd << " " << max_abs << " argmax_miss=" << argmax_miss
              << "\n";
    // Prompt "The theory of computation begins with", 6 tokens, threads=1,
    // transformers GPT2LMHeadModel fp32. Measured max abs: scalar 3.12805e-4,
    // AVX2 2.44141e-4. Argmax matched at every position.
    EXPECT_LT(max_abs, 1e-3f) << max_abs;
    EXPECT_EQ(argmax_miss, 0);
  };

  score(false);
  if (gpt2::cpu_supports_avx2())
    score(true);
}
