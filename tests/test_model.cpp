#include "gpt2/gemm.hpp"
#include "gpt2/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

void write_u32(std::ostream& out, std::uint32_t v) {
  unsigned char b[4] = {
      static_cast<unsigned char>(v & 0xFF), static_cast<unsigned char>((v >> 8) & 0xFF),
      static_cast<unsigned char>((v >> 16) & 0xFF), static_cast<unsigned char>((v >> 24) & 0xFF)};
  out.write(reinterpret_cast<const char*>(b), 4);
}

void write_i32(std::ostream& out, std::int32_t v) {
  write_u32(out, static_cast<std::uint32_t>(v));
}

std::uint32_t lcg(std::uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  return state;
}

gpt2::Model write_and_load(const std::string& path, gpt2::ModelConfig cfg) {
  std::ofstream out(path, std::ios::binary);
  out.write("GPT2WGT1", 8);
  write_u32(out, 1);
  write_i32(out, cfg.vocab_size);
  write_i32(out, cfg.n_positions);
  write_i32(out, cfg.n_embd);
  write_i32(out, cfg.n_head);
  write_i32(out, cfg.n_layer);
  write_i32(out, cfg.n_inner);
  out.write(reinterpret_cast<const char*>(&cfg.ln_eps), sizeof(float));
  const std::size_t c = static_cast<std::size_t>(cfg.n_embd);
  const std::size_t inner = static_cast<std::size_t>(cfg.n_inner);
  const std::size_t per_layer =
      (c + c) + (c * 3 * c + 3 * c) + (c * c + c) + (c + c) + (c * inner + inner) + (inner * c + c);
  const std::size_t total = static_cast<std::size_t>(cfg.vocab_size) * c +
                            static_cast<std::size_t>(cfg.n_positions) * c +
                            per_layer * static_cast<std::size_t>(cfg.n_layer) + c + c;
  std::uint32_t state = 42;
  for (std::size_t i = 0; i < total; ++i) {
    const float v = (static_cast<float>(lcg(state) % 1000) / 1000.f - 0.5f) * 0.1f;
    out.write(reinterpret_cast<const char*>(&v), sizeof(float));
  }
  out.close();
  return gpt2::Model::load(path);
}

void expect_equal(const std::vector<float>& a, const std::vector<float>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], b[i]) << i;
    if (a[i] != b[i])
      return;
  }
}

} // namespace

TEST(Model, KvCacheMatchesFullForward) {
  gpt2::ModelConfig cfg;
  cfg.vocab_size = 32;
  cfg.n_positions = 16;
  cfg.n_embd = 32;
  cfg.n_head = 4;
  cfg.n_layer = 2;
  cfg.n_inner = 64;
  cfg.ln_eps = 1e-5f;
  const std::string path = "/tmp/gpt2infer_tiny_selfcheck.bin";
  gpt2::Model model = write_and_load(path, cfg);
  EXPECT_GT(model.parameter_count(), 0);

  const std::int32_t tokens[5] = {3, 7, 1, 8, 2};
  const int n = 5;
  for (bool simd : {false, true}) {
    if (simd && !gpt2::cpu_supports_avx2())
      continue;
    gpt2::ForwardConfig fwd;
    fwd.simd = simd;
    std::vector<float> all(static_cast<std::size_t>(n * cfg.vocab_size));
    model.forward_all(tokens, n, all.data(), fwd);

    gpt2::KVCache cache(model.config());
    std::vector<float> step(static_cast<std::size_t>(cfg.vocab_size));
    model.forward(tokens, n - 1, &cache, step.data(), fwd);
    EXPECT_EQ(cache.len(), n - 1);
    model.forward(tokens + (n - 1), 1, &cache, step.data(), fwd);
    EXPECT_EQ(cache.len(), n);

    std::vector<float> last(all.end() - cfg.vocab_size, all.end());
    expect_equal(step, last);
  }
}

TEST(Model, RejectsBadMagic) {
  const std::string path = "/tmp/gpt2infer_bad_magic.bin";
  std::ofstream out(path, std::ios::binary);
  out.write("NOTAMODEL", 9);
  out.close();
  EXPECT_THROW(gpt2::Model::load(path), std::runtime_error);
}

TEST(Model, ReferenceFixture) {
  const std::string model_path = std::string(GPT2_SOURCE_DIR) + "/tests/fixtures/tiny_model.bin";
  const std::string ref_path = std::string(GPT2_SOURCE_DIR) + "/tests/fixtures/tiny_logits.bin";
  std::ifstream model_in(model_path, std::ios::binary);
  std::ifstream ref_in(ref_path, std::ios::binary);
  if (!model_in || !ref_in)
    GTEST_SKIP() << "tiny fixture not generated";

  gpt2::Model model = gpt2::Model::load(model_path);
  auto read_i32 = [](std::istream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    return static_cast<std::int32_t>(
        static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
        (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24));
  };
  const std::int32_t n = read_i32(ref_in);
  const std::int32_t vocab = read_i32(ref_in);
  ASSERT_EQ(vocab, model.config().vocab_size);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    ids[static_cast<std::size_t>(i)] = read_i32(ref_in);
  std::vector<float> ref(static_cast<std::size_t>(n) * static_cast<std::size_t>(vocab));
  ref_in.read(reinterpret_cast<char*>(ref.data()),
              static_cast<std::streamsize>(ref.size() * sizeof(float)));
  ASSERT_TRUE(ref_in);

  std::vector<float> got(ref.size());
  gpt2::ForwardConfig fwd;
  fwd.simd = false;
  model.forward_all(ids.data(), n, got.data(), fwd);
  float max_abs = 0.f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(ref[i] - got[i]));
  }
  EXPECT_LT(max_abs, 2e-4f) << max_abs;

  if (gpt2::cpu_supports_avx2()) {
    std::vector<float> avx(ref.size());
    fwd.simd = true;
    model.forward_all(ids.data(), n, avx.data(), fwd);
    float avx_abs = 0.f;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      avx_abs = std::max(avx_abs, std::fabs(ref[i] - avx[i]));
    }
    EXPECT_LT(avx_abs, 2e-3f) << avx_abs;
  }
}
