#include "gpt2/gemm.hpp"

#include <cstddef>

namespace gpt2 {

#if defined(__clang__)
#define GPT2_NOVEC _Pragma("clang loop vectorize(disable) interleave(disable)")
#else
#define GPT2_NOVEC
#endif

float dot_scalar(const float* a, const float* b, int n) {
  float s = 0.f;
  int k = 0;
  GPT2_NOVEC
  for (; k < n; ++k)
    s += a[k] * b[k];
  return s;
}

void axpy_scalar(float* y, const float* x, float a, int n) {
  int i = 0;
  GPT2_NOVEC
  for (; i < n; ++i)
    y[i] += a * x[i];
}

void gemm_nn_scalar(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
                    int ldc) {
  for (int i = 0; i < M; ++i) {
    float* c = C + static_cast<std::size_t>(i) * static_cast<std::size_t>(ldc);
    for (int j = 0; j < N; ++j)
      c[j] = 0.f;
    const float* a = A + static_cast<std::size_t>(i) * static_cast<std::size_t>(lda);
    for (int k = 0; k < K; ++k) {
      const float av = a[k];
      const float* b = B + static_cast<std::size_t>(k) * static_cast<std::size_t>(ldb);
      int j = 0;
      GPT2_NOVEC
      for (; j < N; ++j)
        c[j] += av * b[j];
    }
  }
}

void gemm_nt_scalar(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
                    int ldc) {
  for (int i = 0; i < M; ++i) {
    const float* a = A + static_cast<std::size_t>(i) * static_cast<std::size_t>(lda);
    float* c = C + static_cast<std::size_t>(i) * static_cast<std::size_t>(ldc);
    for (int j = 0; j < N; ++j) {
      const float* b = B + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb);
      float s = 0.f;
      int k = 0;
      GPT2_NOVEC
      for (; k < K; ++k)
        s += a[k] * b[k];
      c[j] = s;
    }
  }
}

} // namespace gpt2
