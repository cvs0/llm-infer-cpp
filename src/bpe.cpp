#include "gpt2/bpe.hpp"

#include <limits>
#include <utility>

namespace gpt2 {

std::vector<std::string> bpe_merge(std::vector<std::string> symbols, const MergeRanks& ranks) {
  while (symbols.size() >= 2) {
    int best_rank = std::numeric_limits<int>::max();
    int best_i = -1;
    for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) {
      const auto it = ranks.find(std::make_pair(symbols[static_cast<std::size_t>(i)],
                                                symbols[static_cast<std::size_t>(i) + 1]));
      if (it != ranks.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_i < 0)
      break;
    const std::string first = symbols[static_cast<std::size_t>(best_i)];
    const std::string second = symbols[static_cast<std::size_t>(best_i) + 1];
    std::vector<std::string> next;
    next.reserve(symbols.size());
    for (std::size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == first && symbols[i + 1] == second) {
        next.push_back(first + second);
        i += 2;
      } else {
        next.push_back(std::move(symbols[i]));
        ++i;
      }
    }
    symbols = std::move(next);
  }
  return symbols;
}

} // namespace gpt2
