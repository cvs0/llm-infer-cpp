#include "gpt2/generate.hpp"
#include "gpt2/gemm.hpp"
#include "gpt2/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Args {
  std::string model;
  std::string tokenizer;
  std::string prompt;
  std::string tokens_csv;
  std::string dump_logits;
  int max_tokens = 32;
  float temperature = 0.f;
  int top_k = 0;
  std::uint32_t seed = 0;
  int threads = 0;
  bool simd = true;
  bool kv = true;
  bool bench = false;
  int bench_tokens = 32;
  int bench_runs = 3;
  bool info = false;
  bool bos = false;
};

void usage() {
  std::cerr << "usage: gpt2_infer --model PATH --tokenizer PATH [options]\n"
            << "options:\n"
            << "  --prompt TEXT            prompt text\n"
            << "  --tokens ID,ID           prompt token ids, instead of --prompt\n"
            << "  --bos                    empty prompt starts at <|endoftext|>\n"
            << "  --max-tokens N           new tokens, default 32\n"
            << "  --temperature T          <=0 is greedy argmax, default 0\n"
            << "  --top-k K                0 disables the filter, default 0\n"
            << "  --seed N                 default 0\n"
            << "  --threads N              default: hardware concurrency\n"
            << "  --simd / --no-simd       AVX2 GEMM when the CPU supports it, default on\n"
            << "  --kv / --no-kv           reuse K/V across steps, default on\n"
            << "  --dump-logits PATH       write prompt logits and exit\n"
            << "  --bench                  time kv x simd, median of --bench-runs\n"
            << "  --bench-tokens N         generated tokens per bench run, default 32\n"
            << "  --bench-runs N           default 3, after one warmup per configuration\n"
            << "  --info                   print config and exit\n";
}

std::string need(int& i, int argc, char** argv, const char* flag) {
  if (i + 1 >= argc)
    throw std::runtime_error(std::string("missing value for ") + flag);
  return argv[++i];
}

int parse_int(const std::string& s, const char* flag) {
  std::size_t idx = 0;
  int v = 0;
  try {
    v = std::stoi(s, &idx);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("bad integer for ") + flag);
  }
  if (idx != s.size())
    throw std::runtime_error(std::string("bad integer for ") + flag);
  return v;
}

float parse_float(const std::string& s, const char* flag) {
  std::size_t idx = 0;
  float v = 0.f;
  try {
    v = std::stof(s, &idx);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("bad float for ") + flag);
  }
  if (idx != s.size())
    throw std::runtime_error(std::string("bad float for ") + flag);
  return v;
}

std::vector<std::int32_t> parse_tokens(const std::string& csv) {
  std::vector<std::int32_t> ids;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty())
      throw std::runtime_error("empty token id");
    ids.push_back(parse_int(item, "--tokens"));
  }
  if (ids.empty())
    throw std::runtime_error("no token ids");
  return ids;
}

void write_u32(std::ostream& out, std::uint32_t v) {
  unsigned char b[4] = {
      static_cast<unsigned char>(v & 0xFF), static_cast<unsigned char>((v >> 8) & 0xFF),
      static_cast<unsigned char>((v >> 16) & 0xFF), static_cast<unsigned char>((v >> 24) & 0xFF)};
  out.write(reinterpret_cast<const char*>(b), 4);
}

void write_i32(std::ostream& out, std::int32_t v) {
  write_u32(out, static_cast<std::uint32_t>(v));
}

void dump_logits(const std::string& path, const std::vector<std::int32_t>& ids, int vocab,
                 const float* logits) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("failed to open logits file");
  write_i32(out, static_cast<std::int32_t>(ids.size()));
  write_i32(out, vocab);
  for (std::int32_t id : ids)
    write_i32(out, id);
  out.write(
      reinterpret_cast<const char*>(logits),
      static_cast<std::streamsize>(ids.size() * static_cast<std::size_t>(vocab) * sizeof(float)));
  if (!out)
    throw std::runtime_error("failed to write logits");
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
  try {
    Args args;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--help" || a == "-h") {
        usage();
        return 0;
      } else if (a == "--model") {
        args.model = need(i, argc, argv, "--model");
      } else if (a == "--tokenizer") {
        args.tokenizer = need(i, argc, argv, "--tokenizer");
      } else if (a == "--prompt") {
        args.prompt = need(i, argc, argv, "--prompt");
      } else if (a == "--tokens") {
        args.tokens_csv = need(i, argc, argv, "--tokens");
      } else if (a == "--dump-logits") {
        args.dump_logits = need(i, argc, argv, "--dump-logits");
      } else if (a == "--max-tokens") {
        args.max_tokens = parse_int(need(i, argc, argv, "--max-tokens"), "--max-tokens");
      } else if (a == "--temperature") {
        args.temperature = parse_float(need(i, argc, argv, "--temperature"), "--temperature");
      } else if (a == "--top-k") {
        args.top_k = parse_int(need(i, argc, argv, "--top-k"), "--top-k");
      } else if (a == "--seed") {
        args.seed = static_cast<std::uint32_t>(parse_int(need(i, argc, argv, "--seed"), "--seed"));
      } else if (a == "--threads") {
        args.threads = parse_int(need(i, argc, argv, "--threads"), "--threads");
      } else if (a == "--bench-tokens") {
        args.bench_tokens = parse_int(need(i, argc, argv, "--bench-tokens"), "--bench-tokens");
      } else if (a == "--bench-runs") {
        args.bench_runs = parse_int(need(i, argc, argv, "--bench-runs"), "--bench-runs");
      } else if (a == "--simd") {
        args.simd = true;
      } else if (a == "--no-simd") {
        args.simd = false;
      } else if (a == "--kv") {
        args.kv = true;
      } else if (a == "--no-kv") {
        args.kv = false;
      } else if (a == "--bench") {
        args.bench = true;
      } else if (a == "--info") {
        args.info = true;
      } else if (a == "--bos") {
        args.bos = true;
      } else {
        throw std::runtime_error("unknown argument " + a);
      }
    }

    if (args.model.empty()) {
      usage();
      throw std::runtime_error("--model is required");
    }
    gpt2::Model model = gpt2::Model::load(args.model);
    const int threads = args.threads > 0
                            ? args.threads
                            : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    model.set_threads(threads);

    if (args.info) {
      const gpt2::ModelConfig& c = model.config();
      std::cout << "vocab_size=" << c.vocab_size << "\n"
                << "n_positions=" << c.n_positions << "\n"
                << "n_embd=" << c.n_embd << "\n"
                << "n_head=" << c.n_head << "\n"
                << "n_layer=" << c.n_layer << "\n"
                << "n_inner=" << c.n_inner << "\n"
                << "head_dim=" << c.head_dim() << "\n"
                << "ln_eps=" << c.ln_eps << "\n"
                << "parameters=" << model.parameter_count() << "\n"
                << "weight_bytes=" << (model.parameter_count() * 4) << "\n"
                << "avx2=" << (gpt2::cpu_supports_avx2() ? 1 : 0) << "\n"
                << "threads=" << model.threads() << "\n";
      return 0;
    }

    if (args.tokenizer.empty())
      throw std::runtime_error("--tokenizer is required");
    gpt2::Tokenizer tok = gpt2::Tokenizer::load(args.tokenizer);
    if (tok.vocab_size() != model.config().vocab_size) {
      throw std::runtime_error("tokenizer vocab does not match model vocab");
    }

    std::vector<std::int32_t> prompt_ids;
    if (!args.tokens_csv.empty() && !args.prompt.empty()) {
      throw std::runtime_error("pass only one of --prompt and --tokens");
    }
    if (!args.tokens_csv.empty()) {
      prompt_ids = parse_tokens(args.tokens_csv);
    } else if (!args.prompt.empty()) {
      prompt_ids = tok.encode(args.prompt);
    } else if (args.bos) {
      if (tok.eot_id() < 0)
        throw std::runtime_error("tokenizer has no <|endoftext|> id");
      prompt_ids.push_back(tok.eot_id());
    } else {
      throw std::runtime_error("pass --prompt, --tokens, or --bos");
    }
    if (prompt_ids.empty())
      throw std::runtime_error("prompt produced no tokens");

    if (!args.dump_logits.empty()) {
      std::vector<float> logits(prompt_ids.size() *
                                static_cast<std::size_t>(model.config().vocab_size));
      gpt2::ForwardConfig fwd;
      fwd.simd = args.simd;
      model.forward_all(prompt_ids.data(), static_cast<int>(prompt_ids.size()), logits.data(), fwd);
      dump_logits(args.dump_logits, prompt_ids, model.config().vocab_size, logits.data());
      std::cerr << "wrote " << args.dump_logits << " n=" << prompt_ids.size()
                << " vocab=" << model.config().vocab_size << "\n";
      return 0;
    }

    if (args.bench) {
      if (args.bench_tokens <= 0 || args.bench_runs <= 0) {
        throw std::runtime_error("bench counts must be positive");
      }
      std::cout << "avx2=" << (gpt2::cpu_supports_avx2() ? 1 : 0) << " threads=" << model.threads()
                << " prompt_tokens=" << prompt_ids.size() << " gen_tokens=" << args.bench_tokens
                << " runs=" << args.bench_runs << "\n";
      std::cout << "kv simd run seconds tok_s\n";
      const int kvs[2] = {1, 0};
      const int simds[2] = {1, 0};
      for (int kv : kvs) {
        for (int simd : simds) {
          gpt2::GenerateConfig cfg;
          cfg.max_new_tokens = args.bench_tokens;
          cfg.temperature = 0.f;
          cfg.top_k = 0;
          cfg.seed = args.seed;
          cfg.simd = simd != 0;
          cfg.use_kv_cache = kv != 0;
          cfg.threads = model.threads();
          (void)gpt2::generate(model, prompt_ids, cfg);
          std::vector<double> rates;
          rates.reserve(static_cast<std::size_t>(args.bench_runs));
          for (int run = 0; run < args.bench_runs; ++run) {
            const gpt2::GenerateOutput out = gpt2::generate(model, prompt_ids, cfg);
            const double tok_s =
                out.seconds > 0.0 ? static_cast<double>(out.new_tokens) / out.seconds : 0.0;
            rates.push_back(tok_s);
            std::cout << kv << " " << simd << " " << run << " " << out.seconds << " " << tok_s
                      << "\n";
          }
          std::cout << "median kv=" << kv << " simd=" << simd << " tok_s=" << median(rates) << "\n";
        }
      }
      return 0;
    }

    gpt2::GenerateConfig cfg;
    cfg.max_new_tokens = args.max_tokens;
    cfg.temperature = args.temperature;
    cfg.top_k = args.top_k;
    cfg.seed = args.seed;
    cfg.simd = args.simd;
    cfg.use_kv_cache = args.kv;
    cfg.threads = model.threads();
    const gpt2::GenerateOutput out = gpt2::generate(model, prompt_ids, cfg);
    const std::string text = tok.decode(out.tokens);
    std::cout << text << "\n";
    const double tok_s =
        out.seconds > 0.0 ? static_cast<double>(out.new_tokens) / out.seconds : 0.0;
    std::cerr << "prompt_tokens=" << out.prompt_tokens << " new_tokens=" << out.new_tokens
              << " seconds=" << out.seconds << " tok_s=" << tok_s << " simd=" << (args.simd ? 1 : 0)
              << " kv=" << (args.kv ? 1 : 0) << " threads=" << model.threads() << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
