#pragma once

#include "gpt2/bpe.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gpt2 {

// GPT-2 pretokenizer. Pieces are slices of `text` and cover it without gaps.
// The pattern, in priority order, is the GPT-2 regex:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// ` ?` is a single ASCII space. \p{L}, \p{N}, and \s follow the `regex` module.
// Throws on invalid UTF-8.
std::vector<std::string> pretokenize(std::string_view text);

// Codepoint assigned to a raw byte by the GPT-2 bytes-to-unicode map.
// Printable bytes in the original table map to themselves. Byte 32 maps to 288.
uint32_t byte_to_codepoint(unsigned char byte);

class Tokenizer {
public:
  static Tokenizer load(const std::string& path);

  std::vector<int32_t> encode(std::string_view text) const;
  std::string decode(const int32_t* ids, int n) const;
  std::string decode(const std::vector<int32_t>& ids) const {
    return decode(ids.data(), static_cast<int>(ids.size()));
  }

  int vocab_size() const noexcept { return static_cast<int>(id_to_token_.size()); }
  int eot_id() const noexcept { return eot_id_; }

private:
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  MergeRanks ranks_;
  int eot_id_ = -1;
  std::array<std::string, 256> byte_symbol_{};
};

} // namespace gpt2
