#include "gpt2/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace gpt2 {

void layer_norm(const float* x, const float* w, const float* b, float* y, int n, float eps) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += static_cast<double>(x[i]);
  const double mean = sum / static_cast<double>(n);
  double var = 0.0;
  for (int i = 0; i < n; ++i) {
    const double d = static_cast<double>(x[i]) - mean;
    var += d * d;
  }
  var /= static_cast<double>(n);
  const float inv = static_cast<float>(1.0 / std::sqrt(var + static_cast<double>(eps)));
  for (int i = 0; i < n; ++i) {
    const float d = static_cast<float>(static_cast<double>(x[i]) - mean);
    y[i] = d * inv * w[i] + b[i];
  }
}

void gelu_inplace(float* x, int n) {
  // float32(sqrt(2/pi)), matching a double sqrt cast to float32.
  const float k1 = static_cast<float>(std::sqrt(2.0 / 3.14159265358979323846));
  const float k0 = 0.044715f;
  for (int i = 0; i < n; ++i) {
    const float v = x[i];
    const float x3 = v * v * v;
    const float u = k1 * (v + k0 * x3);
    x[i] = 0.5f * v * (1.0f + std::tanh(u));
  }
}

void softmax_inplace(float* x, int n) {
  float m = x[0];
  for (int i = 1; i < n; ++i)
    m = std::max(m, x[i]);
  float sum = 0.f;
  for (int i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - m);
    sum += x[i];
  }
  const float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i)
    x[i] *= inv;
}

void add_bias_inplace(float* x, const float* bias, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float* row = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
    for (int c = 0; c < cols; ++c)
      row[c] += bias[c];
  }
}

void add_inplace(float* x, const float* y, int n) {
  for (int i = 0; i < n; ++i)
    x[i] += y[i];
}

} // namespace gpt2
