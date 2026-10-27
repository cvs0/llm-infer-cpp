#include "gpt2/tokenizer.hpp"

#include "gpt2/unicode.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace gpt2 {

namespace {

struct Utf8 {
  std::vector<std::uint32_t> cp;
  std::vector<std::size_t> off;
};

void read_exact(std::istream& in, void* dst, std::size_t n) {
  in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
  if (static_cast<std::size_t>(in.gcount()) != n) {
    throw std::runtime_error("truncated tokenizer file");
  }
}

std::uint32_t read_u32(std::istream& in) {
  unsigned char b[4];
  read_exact(in, b, 4);
  return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::string read_string(std::istream& in) {
  const std::uint32_t n = read_u32(in);
  if (n > 10000000u)
    throw std::runtime_error("tokenizer string too long");
  std::string s(n, '\0');
  if (n != 0)
    read_exact(in, s.data(), n);
  return s;
}

std::string utf8_encode(std::uint32_t cp) {
  std::string o;
  if (cp < 0x80) {
    o.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    o.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    o.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    o.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return o;
}

Utf8 decode_utf8(std::string_view text) {
  Utf8 u;
  std::size_t i = 0;
  while (i < text.size()) {
    const auto c0 = static_cast<unsigned char>(text[i]);
    std::size_t len = 0;
    std::uint32_t cp = 0;
    if (c0 < 0x80) {
      len = 1;
      cp = c0;
    } else if ((c0 & 0xE0) == 0xC0) {
      len = 2;
      cp = c0 & 0x1F;
    } else if ((c0 & 0xF0) == 0xE0) {
      len = 3;
      cp = c0 & 0x0F;
    } else if ((c0 & 0xF8) == 0xF0) {
      len = 4;
      cp = c0 & 0x07;
    } else {
      throw std::runtime_error("invalid utf-8");
    }
    if (i + len > text.size())
      throw std::runtime_error("truncated utf-8");
    for (std::size_t k = 1; k < len; ++k) {
      const auto ck = static_cast<unsigned char>(text[i + k]);
      if ((ck & 0xC0) != 0x80)
        throw std::runtime_error("invalid utf-8 continuation");
      cp = (cp << 6) | (ck & 0x3F);
    }
    if (len == 2 && cp < 0x80)
      throw std::runtime_error("overlong utf-8");
    if (len == 3 && cp < 0x800)
      throw std::runtime_error("overlong utf-8");
    if (len == 4 && cp < 0x10000)
      throw std::runtime_error("overlong utf-8");
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
      throw std::runtime_error("invalid unicode scalar");
    }
    u.off.push_back(i);
    u.cp.push_back(cp);
    i += len;
  }
  u.off.push_back(text.size());
  return u;
}

bool starts_with(const std::uint32_t* cp, std::size_t n, std::size_t i, const std::uint32_t* s,
                 std::size_t sn) {
  if (i + sn > n)
    return false;
  for (std::size_t k = 0; k < sn; ++k) {
    if (cp[i + k] != s[k])
      return false;
  }
  return true;
}

bool match_contraction(const std::uint32_t* cp, std::size_t n, std::size_t i, std::size_t* end) {
  static const std::uint32_t s1[] = {'\'', 's'};
  static const std::uint32_t t1[] = {'\'', 't'};
  static const std::uint32_t re[] = {'\'', 'r', 'e'};
  static const std::uint32_t ve[] = {'\'', 'v', 'e'};
  static const std::uint32_t m1[] = {'\'', 'm'};
  static const std::uint32_t ll[] = {'\'', 'l', 'l'};
  static const std::uint32_t d1[] = {'\'', 'd'};
  if (starts_with(cp, n, i, s1, 2) || starts_with(cp, n, i, t1, 2)) {
    *end = i + 2;
    return true;
  }
  if (starts_with(cp, n, i, re, 3) || starts_with(cp, n, i, ve, 3)) {
    *end = i + 3;
    return true;
  }
  if (starts_with(cp, n, i, m1, 2)) {
    *end = i + 2;
    return true;
  }
  if (starts_with(cp, n, i, ll, 3)) {
    *end = i + 3;
    return true;
  }
  if (starts_with(cp, n, i, d1, 2)) {
    *end = i + 2;
    return true;
  }
  return false;
}

bool is_other(std::uint32_t cp) {
  return !is_unicode_space(cp) && !is_unicode_letter(cp) && !is_unicode_number(cp);
}

bool match_class(const std::uint32_t* cp, std::size_t n, std::size_t i, std::size_t* end,
                 bool (*pred)(std::uint32_t)) {
  std::size_t j = i;
  if (j < n && cp[j] == static_cast<std::uint32_t>(' ') && j + 1 < n && pred(cp[j + 1]))
    ++j;
  if (j >= n || !pred(cp[j]))
    return false;
  while (j < n && pred(cp[j]))
    ++j;
  *end = j;
  return true;
}

bool match_ws_boundary(const std::uint32_t* cp, std::size_t n, std::size_t i, std::size_t* end) {
  if (i >= n || !is_unicode_space(cp[i]))
    return false;
  std::size_t j = i;
  while (j < n && is_unicode_space(cp[j]))
    ++j;
  while (j > i) {
    const bool non_space_follows = (j < n) && !is_unicode_space(cp[j]);
    if (!non_space_follows) {
      *end = j;
      return true;
    }
    --j;
  }
  return false;
}

bool match_ws(const std::uint32_t* cp, std::size_t n, std::size_t i, std::size_t* end) {
  if (i >= n || !is_unicode_space(cp[i]))
    return false;
  std::size_t j = i;
  while (j < n && is_unicode_space(cp[j]))
    ++j;
  *end = j;
  return true;
}

const std::array<std::int16_t, 512>& codepoint_to_byte() {
  static const auto table = [] {
    std::array<std::int16_t, 512> t{};
    t.fill(-1);
    for (int b = 0; b < 256; ++b) {
      const std::uint32_t cp = byte_to_codepoint(static_cast<unsigned char>(b));
      t[cp] = static_cast<std::int16_t>(b);
    }
    return t;
  }();
  return table;
}

std::uint32_t byte_codepoint_table(unsigned char byte) {
  // bytes_to_unicode from the GPT-2 encoder: identity on a fixed printable set,
  // then the remaining bytes in ascending order map to 256, 257, ...
  static const auto table = [] {
    std::array<std::uint32_t, 256> cp{};
    bool seen[256] = {};
    std::uint32_t cs[256];
    int nbs = 0;
    auto push = [&](int b) {
      seen[b] = true;
      cs[nbs++] = static_cast<std::uint32_t>(b);
    };
    for (int b = '!'; b <= '~'; ++b)
      push(b);
    for (int b = 0xA1; b <= 0xAC; ++b)
      push(b);
    for (int b = 0xAE; b <= 0xFF; ++b)
      push(b);
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
      if (!seen[b])
        cs[nbs++] = static_cast<std::uint32_t>(256 + extra++);
    }
    int index = 0;
    for (int b = '!'; b <= '~'; ++b)
      cp[static_cast<unsigned>(b)] = cs[index++];
    for (int b = 0xA1; b <= 0xAC; ++b)
      cp[static_cast<unsigned>(b)] = cs[index++];
    for (int b = 0xAE; b <= 0xFF; ++b)
      cp[static_cast<unsigned>(b)] = cs[index++];
    for (int b = 0; b < 256; ++b) {
      if (!seen[b])
        cp[static_cast<unsigned>(b)] = cs[index++];
    }
    return cp;
  }();
  return table[byte];
}

} // namespace

uint32_t byte_to_codepoint(unsigned char byte) {
  return byte_codepoint_table(byte);
}

std::vector<std::string> pretokenize(std::string_view text) {
  const Utf8 u = decode_utf8(text);
  const std::size_t n = u.cp.size();
  std::vector<std::string> pieces;
  std::size_t i = 0;
  while (i < n) {
    std::size_t end = i;
    const std::uint32_t* cp = u.cp.data();
    if (match_contraction(cp, n, i, &end) || match_class(cp, n, i, &end, is_unicode_letter) ||
        match_class(cp, n, i, &end, is_unicode_number) || match_class(cp, n, i, &end, is_other) ||
        match_ws_boundary(cp, n, i, &end) || match_ws(cp, n, i, &end)) {
      const std::size_t b0 = u.off[i];
      const std::size_t b1 = u.off[end];
      pieces.emplace_back(text.substr(b0, b1 - b0));
      i = end;
    } else {
      throw std::runtime_error("pretokenize made no progress");
    }
  }
  return pieces;
}

Tokenizer Tokenizer::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("failed to open tokenizer file");
  char magic[8];
  read_exact(in, magic, 8);
  if (std::memcmp(magic, "GPT2TOK1", 8) != 0)
    throw std::runtime_error("bad tokenizer magic");
  if (read_u32(in) != 1)
    throw std::runtime_error("unsupported tokenizer version");
  const std::uint32_t n_vocab = read_u32(in);
  const std::uint32_t n_merges = read_u32(in);
  if (n_vocab == 0 || n_vocab > 1000000u)
    throw std::runtime_error("bad vocab size");
  if (n_merges > 2000000u)
    throw std::runtime_error("bad merge count");

  Tokenizer tok;
  tok.id_to_token_.resize(n_vocab);
  tok.token_to_id_.reserve(n_vocab);
  for (std::uint32_t i = 0; i < n_vocab; ++i) {
    tok.id_to_token_[i] = read_string(in);
    tok.token_to_id_.emplace(tok.id_to_token_[i], static_cast<std::int32_t>(i));
  }
  tok.ranks_.reserve(n_merges);
  for (std::uint32_t i = 0; i < n_merges; ++i) {
    std::string left = read_string(in);
    std::string right = read_string(in);
    tok.ranks_.emplace(std::make_pair(std::move(left), std::move(right)), static_cast<int>(i));
  }
  char extra = 0;
  in.read(&extra, 1);
  if (in.gcount() != 0)
    throw std::runtime_error("trailing bytes in tokenizer file");

  const auto it = tok.token_to_id_.find("<|endoftext|>");
  if (it != tok.token_to_id_.end())
    tok.eot_id_ = it->second;
  for (int b = 0; b < 256; ++b) {
    tok.byte_symbol_[static_cast<std::size_t>(b)] =
        utf8_encode(byte_to_codepoint(static_cast<unsigned char>(b)));
  }
  return tok;
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text) const {
  std::vector<std::int32_t> ids;
  constexpr std::string_view kEot = "<|endoftext|>";
  std::size_t i = 0;
  while (i < text.size()) {
    if (eot_id_ >= 0 && text.compare(i, kEot.size(), kEot) == 0) {
      ids.push_back(eot_id_);
      i += kEot.size();
      continue;
    }
    std::size_t j = text.size();
    if (eot_id_ >= 0) {
      const auto pos = text.find(kEot, i);
      if (pos != std::string_view::npos)
        j = pos;
    }
    const auto pieces = pretokenize(text.substr(i, j - i));
    for (const std::string& piece : pieces) {
      std::vector<std::string> symbols;
      symbols.reserve(piece.size());
      for (unsigned char b : piece)
        symbols.push_back(byte_symbol_[b]);
      const std::vector<std::string> merged = bpe_merge(std::move(symbols), ranks_);
      for (const std::string& sym : merged) {
        const auto it = token_to_id_.find(sym);
        if (it == token_to_id_.end())
          throw std::runtime_error("bpe piece missing from vocab");
        ids.push_back(it->second);
      }
    }
    i = j;
  }
  return ids;
}

std::string Tokenizer::decode(const std::int32_t* ids, int n) const {
  const auto& map = codepoint_to_byte();
  std::string out;
  for (int i = 0; i < n; ++i) {
    if (ids[i] < 0 || ids[i] >= vocab_size())
      throw std::runtime_error("token id out of range");
    const Utf8 u = decode_utf8(id_to_token_[static_cast<std::size_t>(ids[i])]);
    for (std::uint32_t cp : u.cp) {
      if (cp >= map.size() || map[cp] < 0)
        throw std::runtime_error("token is not byte-level");
      out.push_back(static_cast<char>(map[cp]));
    }
  }
  return out;
}

} // namespace gpt2
