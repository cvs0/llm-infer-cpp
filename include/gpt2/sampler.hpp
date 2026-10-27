#pragma once

#include <cstdint>
#include <random>

namespace gpt2 {

// temperature <= 0 selects argmax (lowest index on ties).
// top_k <= 0 disables the filter. top_k > 0 keeps the k largest logits.
// The filtered distribution is softmax(logit / temperature).
int sample_token(const float* logits, int n, float temperature, int top_k, std::mt19937& rng);

} // namespace gpt2
