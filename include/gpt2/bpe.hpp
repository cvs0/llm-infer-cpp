#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gpt2 {

struct StringPairHash {
  std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
    std::size_t h1 = std::hash<std::string>{}(p.first);
    std::size_t h2 = std::hash<std::string>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

// Map (left, right) -> rank. Rank 0 is the highest-priority merge.
using MergeRanks = std::unordered_map<std::pair<std::string, std::string>, int, StringPairHash>;

// Merge the adjacent pair with the lowest rank until none remain.
// Every non-overlapping occurrence of that pair is merged in one left-to-right pass
// before ranks are recomputed. Pairs absent from `ranks` are never merged.
std::vector<std::string> bpe_merge(std::vector<std::string> symbols, const MergeRanks& ranks);

} // namespace gpt2
