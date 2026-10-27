#include "gpt2/model.hpp"

#include "gpt2/gemm.hpp"
#include "gpt2/ops.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace gpt2 {

namespace {

bool little_endian() {
  const std::uint16_t x = 1;
  std::uint8_t b = 0;
  std::memcpy(&b, &x, 1);
  return b == 1;
}

void read_exact(std::istream& in, void* dst, std::size_t n) {
  in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
  if (static_cast<std::size_t>(in.gcount()) != n)
    throw std::runtime_error("truncated model file");
}

std::uint32_t read_u32(std::istream& in) {
  unsigned char b[4];
  read_exact(in, b, 4);
  return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::int32_t read_i32(std::istream& in) {
  return static_cast<std::int32_t>(read_u32(in));
}

float read_f32(std::istream& in) {
  float v = 0.f;
  read_exact(in, &v, sizeof(v));
  return v;
}

void require(bool cond, const char* msg) {
  if (!cond)
    throw std::runtime_error(msg);
}

} // namespace

KVCache::KVCache(const ModelConfig& cfg)
    : n_head_(cfg.n_head), head_dim_(cfg.head_dim()), max_seq_(cfg.n_positions), len_(0),
      k_(static_cast<std::size_t>(cfg.n_layer) * static_cast<std::size_t>(cfg.n_head) *
         static_cast<std::size_t>(cfg.n_positions) * static_cast<std::size_t>(cfg.head_dim())),
      v_(static_cast<std::size_t>(cfg.n_layer) * static_cast<std::size_t>(cfg.n_head) *
         static_cast<std::size_t>(cfg.n_positions) * static_cast<std::size_t>(cfg.head_dim())) {}

Model::Model() = default;
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

int Model::threads() const noexcept {
  return pool_ ? pool_->size() : 1;
}

void Model::set_threads(int nthreads) {
  if (nthreads < 1)
    nthreads = 1;
  if (pool_ && pool_->size() == nthreads)
    return;
  pool_ = std::make_unique<ThreadPool>(nthreads);
}

Model Model::load(const std::string& path) {
  if (!little_endian())
    throw std::runtime_error("model loader requires a little-endian host");
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("failed to open model file");

  char magic[8];
  read_exact(in, magic, 8);
  require(std::memcmp(magic, "GPT2WGT1", 8) == 0, "bad model magic");
  require(read_u32(in) == 1, "unsupported model version");

  ModelConfig cfg;
  cfg.vocab_size = read_i32(in);
  cfg.n_positions = read_i32(in);
  cfg.n_embd = read_i32(in);
  cfg.n_head = read_i32(in);
  cfg.n_layer = read_i32(in);
  cfg.n_inner = read_i32(in);
  cfg.ln_eps = read_f32(in);

  require(cfg.vocab_size > 0 && cfg.vocab_size <= 1000000, "vocab_size out of range");
  require(cfg.n_positions > 0 && cfg.n_positions <= 65536, "n_positions out of range");
  require(cfg.n_embd > 0 && cfg.n_embd <= 8192, "n_embd out of range");
  require(cfg.n_head > 0 && cfg.n_embd % cfg.n_head == 0, "n_head does not divide n_embd");
  require(cfg.n_layer > 0 && cfg.n_layer <= 128, "n_layer out of range");
  require(cfg.n_inner > 0 && cfg.n_inner <= 16384, "n_inner out of range");
  require(cfg.ln_eps > 0.f, "ln_eps must be positive");

  const std::size_t c = static_cast<std::size_t>(cfg.n_embd);
  const std::size_t inner = static_cast<std::size_t>(cfg.n_inner);
  const std::size_t per_layer =
      (c + c) + (c * 3 * c + 3 * c) + (c * c + c) + (c + c) + (c * inner + inner) + (inner * c + c);
  const std::size_t total = static_cast<std::size_t>(cfg.vocab_size) * c +
                            static_cast<std::size_t>(cfg.n_positions) * c +
                            per_layer * static_cast<std::size_t>(cfg.n_layer) + c + c;

  Model model;
  model.cfg_ = cfg;
  model.n_params_ = static_cast<std::int64_t>(total);
  model.storage_ = AlignedFloats(total);
  read_exact(in, model.storage_.data(), total * sizeof(float));
  char extra = 0;
  in.read(&extra, 1);
  require(in.gcount() == 0, "trailing bytes in model file");

  float* slot = model.storage_.data();
  auto take = [&](std::size_t n) {
    float* p = slot;
    slot += n;
    return p;
  };
  model.wte_ = take(static_cast<std::size_t>(cfg.vocab_size) * c);
  model.wpe_ = take(static_cast<std::size_t>(cfg.n_positions) * c);
  model.blocks_ = std::make_unique<Block[]>(static_cast<std::size_t>(cfg.n_layer));
  for (int layer = 0; layer < cfg.n_layer; ++layer) {
    Block& b = model.blocks_[static_cast<std::size_t>(layer)];
    b.ln1_w = take(c);
    b.ln1_b = take(c);
    b.attn_w = take(c * 3 * c);
    b.attn_b = take(3 * c);
    b.attn_proj_w = take(c * c);
    b.attn_proj_b = take(c);
    b.ln2_w = take(c);
    b.ln2_b = take(c);
    b.fc_w = take(c * inner);
    b.fc_b = take(inner);
    b.fc_proj_w = take(inner * c);
    b.fc_proj_b = take(c);
  }
  model.ln_f_w_ = take(c);
  model.ln_f_b_ = take(c);
  require(slot == model.storage_.data() + total, "weight span mismatch");

  const std::size_t s = static_cast<std::size_t>(cfg.n_positions);
  model.x_ = AlignedFloats(s * c);
  model.h_ = AlignedFloats(s * c);
  model.qkv_ = AlignedFloats(s * 3 * c);
  model.attn_ = AlignedFloats(s * c);
  model.mlp_ = AlignedFloats(s * inner);
  model.scores_ = AlignedFloats(s);
  model.scratch_ = KVCache(cfg);
  model.pool_ = std::make_unique<ThreadPool>(1);
  return model;
}

void Model::forward(const std::int32_t* tokens, int n_tokens, KVCache* cache, float* logits,
                    const ForwardConfig& cfg) {
  KVCache* kv = cache;
  if (kv == nullptr) {
    scratch_.clear();
    kv = &scratch_;
  }
  const bool use_simd = cfg.simd && cpu_supports_avx2();
  if (use_simd) {
    forward_t<true>(tokens, n_tokens, kv, kv->len(), logits, false);
  } else {
    forward_t<false>(tokens, n_tokens, kv, kv->len(), logits, false);
  }
}

void Model::forward_all(const std::int32_t* tokens, int n_tokens, float* logits,
                        const ForwardConfig& cfg) {
  scratch_.clear();
  const bool use_simd = cfg.simd && cpu_supports_avx2();
  if (use_simd) {
    forward_t<true>(tokens, n_tokens, &scratch_, 0, logits, true);
  } else {
    forward_t<false>(tokens, n_tokens, &scratch_, 0, logits, true);
  }
}

template <bool Simd>
void Model::forward_t(const std::int32_t* tokens, int n_tokens, KVCache* kv, int start,
                      float* logits, bool all_logits) {
  require(tokens != nullptr && logits != nullptr, "null forward argument");
  require(n_tokens > 0, "empty token span");
  require(start >= 0 && static_cast<std::int64_t>(start) + n_tokens <= cfg_.n_positions,
          "context length exceeded");
  for (int t = 0; t < n_tokens; ++t) {
    require(tokens[t] >= 0 && tokens[t] < cfg_.vocab_size, "token id out of range");
  }

  const int C = cfg_.n_embd;
  const int H = cfg_.n_head;
  const int D = cfg_.head_dim();
  const int I = cfg_.n_inner;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  ThreadPool* pool = pool_.get();
  float* x = x_.data();

  for (int t = 0; t < n_tokens; ++t) {
    const float* te = wte_ + static_cast<std::size_t>(tokens[t]) * static_cast<std::size_t>(C);
    const float* pe = wpe_ + static_cast<std::size_t>(start + t) * static_cast<std::size_t>(C);
    float* row = x + static_cast<std::size_t>(t) * static_cast<std::size_t>(C);
    for (int c = 0; c < C; ++c)
      row[c] = te[c] + pe[c];
  }

  for (int layer = 0; layer < cfg_.n_layer; ++layer) {
    const Block& b = blocks_[static_cast<std::size_t>(layer)];
    float* h = h_.data();
    for (int t = 0; t < n_tokens; ++t) {
      layer_norm(x + static_cast<std::size_t>(t) * static_cast<std::size_t>(C), b.ln1_w, b.ln1_b,
                 h + static_cast<std::size_t>(t) * static_cast<std::size_t>(C), C, cfg_.ln_eps);
    }
    gemm_nn(n_tokens, 3 * C, C, h, C, b.attn_w, 3 * C, qkv_.data(), 3 * C, Simd, pool);
    add_bias_inplace(qkv_.data(), b.attn_b, n_tokens, 3 * C);

    for (int t = 0; t < n_tokens; ++t) {
      const float* row =
          qkv_.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(3 * C);
      const float* krow = row + C;
      const float* vrow = row + 2 * C;
      for (int head = 0; head < H; ++head) {
        std::memcpy(kv->k_at(layer, head, start + t), krow + head * D,
                    sizeof(float) * static_cast<std::size_t>(D));
        std::memcpy(kv->v_at(layer, head, start + t), vrow + head * D,
                    sizeof(float) * static_cast<std::size_t>(D));
      }
    }

    float* attn = attn_.data();
    for (int t = 0; t < n_tokens; ++t) {
      const int qpos = start + t;
      const int nscore = qpos + 1;
      const float* qrow =
          qkv_.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(3 * C);
      float* orow = attn + static_cast<std::size_t>(t) * static_cast<std::size_t>(C);
      for (int head = 0; head < H; ++head) {
        const float* q = qrow + head * D;
        float* scores = scores_.data();
        for (int j = 0; j < nscore; ++j) {
          scores[j] = dot_dispatch<Simd>(q, kv->k_at(layer, head, j), D) * scale;
        }
        softmax_inplace(scores, nscore);
        float* o = orow + head * D;
        for (int d = 0; d < D; ++d)
          o[d] = 0.f;
        for (int j = 0; j < nscore; ++j) {
          axpy_dispatch<Simd>(o, kv->v_at(layer, head, j), scores[j], D);
        }
      }
    }

    gemm_nn(n_tokens, C, C, attn, C, b.attn_proj_w, C, h, C, Simd, pool);
    add_bias_inplace(h, b.attn_proj_b, n_tokens, C);
    add_inplace(x, h, n_tokens * C);

    for (int t = 0; t < n_tokens; ++t) {
      layer_norm(x + static_cast<std::size_t>(t) * static_cast<std::size_t>(C), b.ln2_w, b.ln2_b,
                 h + static_cast<std::size_t>(t) * static_cast<std::size_t>(C), C, cfg_.ln_eps);
    }
    gemm_nn(n_tokens, I, C, h, C, b.fc_w, I, mlp_.data(), I, Simd, pool);
    add_bias_inplace(mlp_.data(), b.fc_b, n_tokens, I);
    gelu_inplace(mlp_.data(), n_tokens * I);
    gemm_nn(n_tokens, C, I, mlp_.data(), I, b.fc_proj_w, C, h, C, Simd, pool);
    add_bias_inplace(h, b.fc_proj_b, n_tokens, C);
    add_inplace(x, h, n_tokens * C);
  }

  float* h = h_.data();
  for (int t = 0; t < n_tokens; ++t) {
    layer_norm(x + static_cast<std::size_t>(t) * static_cast<std::size_t>(C), ln_f_w_, ln_f_b_,
               h + static_cast<std::size_t>(t) * static_cast<std::size_t>(C), C, cfg_.ln_eps);
  }
  if (all_logits) {
    gemm_nt(n_tokens, cfg_.vocab_size, C, h, C, wte_, C, logits, cfg_.vocab_size, Simd, pool);
  } else {
    const float* last = h + static_cast<std::size_t>(n_tokens - 1) * static_cast<std::size_t>(C);
    gemm_nt(1, cfg_.vocab_size, C, last, C, wte_, C, logits, cfg_.vocab_size, Simd, pool);
  }
  kv->set_len(start + n_tokens);
}

} // namespace gpt2
