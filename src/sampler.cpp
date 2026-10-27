#include "gpt2/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gpt2 {

int sample_token(const float* logits, int n, float temperature, int top_k, std::mt19937& rng) {
  if (n <= 0)
    throw std::runtime_error("empty logits");
  if (!(temperature > 0.f)) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
      if (logits[i] > logits[best])
        best = i;
    }
    return best;
  }

  int k = top_k;
  if (k <= 0 || k > n)
    k = n;
  std::vector<int> ids(static_cast<std::size_t>(n));
  std::iota(ids.begin(), ids.end(), 0);
  if (k < n) {
    std::nth_element(ids.begin(), ids.begin() + (k - 1), ids.end(),
                     [&](int a, int b) { return logits[a] > logits[b]; });
    ids.resize(static_cast<std::size_t>(k));
  }

  float m = logits[ids[0]];
  for (int id : ids)
    m = std::max(m, logits[id]);
  std::vector<float> probs(ids.size());
  float sum = 0.f;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    probs[i] = std::exp((logits[ids[i]] - m) / temperature);
    sum += probs[i];
  }
  if (!(sum > 0.f))
    return ids[0];

  std::uniform_real_distribution<float> dist(0.f, 1.f);
  const float r = dist(rng);
  float cum = 0.f;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    cum += probs[i] / sum;
    if (r < cum)
      return ids[i];
  }
  return ids.back();
}

} // namespace gpt2
