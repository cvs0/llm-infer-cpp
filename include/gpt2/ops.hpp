#pragma once

namespace gpt2 {

// y = (x - mean) * rsqrt(var + eps) * w + b
// mean and var are accumulated in float64. var divides by n (LayerNorm, not
// the unbiased sample variance).
void layer_norm(const float* x, const float* w, const float* b, float* y, int n, float eps);

// In-place GPT-2 GELU (the tanh approximation):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu_inplace(float* x, int n);

// In-place softmax over x[0, n). n must be > 0.
void softmax_inplace(float* x, int n);

// x[r, c] += bias[c] for a row-major [rows, cols] matrix.
void add_bias_inplace(float* x, const float* bias, int rows, int cols);

// x[i] += y[i] for i in [0, n).
void add_inplace(float* x, const float* y, int n);

} // namespace gpt2
