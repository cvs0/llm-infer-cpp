#pragma once

#include "gpt2/thread_pool.hpp"

namespace gpt2 {

// C[m, n] = sum_k A[m, k] * B[k, n]
// A is M x K with row stride lda, B is K x N with row stride ldb, C is M x N
// with row stride ldc. All row-major. Overwrites C (beta = 0).
// simd selects the AVX2 kernel when the CPU supports AVX2 and FMA.
// pool may be null. Column tiles are disjoint, so each output element is
// produced by one thread.
void gemm_nn(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
             int ldc, bool simd, ThreadPool* pool);

// C[m, n] = sum_k A[m, k] * B[n, k]
// B is N x K with row stride ldb. Used for the tied output projection, where
// B is the token embedding table [vocab, n_embd].
void gemm_nt(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
             int ldc, bool simd, ThreadPool* pool);

float dot_scalar(const float* a, const float* b, int n);
void axpy_scalar(float* y, const float* x, float a, int n);

bool cpu_supports_avx2();

#if defined(GPT2_COMPILE_AVX2) && GPT2_COMPILE_AVX2
float dot_avx2(const float* a, const float* b, int n);
void axpy_avx2(float* y, const float* x, float a, int n);
#endif

template <bool Simd> inline float dot_dispatch(const float* a, const float* b, int n) {
  if constexpr (Simd) {
#if defined(GPT2_COMPILE_AVX2) && GPT2_COMPILE_AVX2
    return dot_avx2(a, b, n);
#else
    return dot_scalar(a, b, n);
#endif
  } else {
    return dot_scalar(a, b, n);
  }
}

template <bool Simd> inline void axpy_dispatch(float* y, const float* x, float a, int n) {
  if constexpr (Simd) {
#if defined(GPT2_COMPILE_AVX2) && GPT2_COMPILE_AVX2
    axpy_avx2(y, x, a, n);
#else
    axpy_scalar(y, x, a, n);
#endif
  } else {
    axpy_scalar(y, x, a, n);
  }
}

} // namespace gpt2
