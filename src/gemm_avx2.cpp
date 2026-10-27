#include "gpt2/gemm.hpp"

#if defined(GPT2_COMPILE_AVX2) && GPT2_COMPILE_AVX2

#include <cstddef>
#include <immintrin.h>

namespace gpt2 {

namespace {

inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  lo = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, lo);
  lo = _mm_add_ss(lo, shuf);
  return _mm_cvtss_f32(lo);
}

} // namespace

float dot_avx2(const float* a, const float* b, int n) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  int k = 0;
  for (; k + 16 <= n; k += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), acc1);
  }
  float s = hsum256(_mm256_add_ps(acc0, acc1));
  for (; k < n; ++k)
    s += a[k] * b[k];
  return s;
}

void axpy_avx2(float* y, const float* x, float a, int n) {
  const __m256 av = _mm256_set1_ps(a);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 yv = _mm256_loadu_ps(y + i);
    yv = _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i), yv);
    _mm256_storeu_ps(y + i, yv);
  }
  for (; i < n; ++i)
    y[i] += a * x[i];
}

void gemm_nn_avx2(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
                  int ldc) {
  int j = 0;
  for (; j + 32 <= N; j += 32) {
    for (int i = 0; i < M; ++i) {
      const float* a = A + static_cast<std::size_t>(i) * static_cast<std::size_t>(lda);
      __m256 c0 = _mm256_setzero_ps();
      __m256 c1 = _mm256_setzero_ps();
      __m256 c2 = _mm256_setzero_ps();
      __m256 c3 = _mm256_setzero_ps();
      for (int k = 0; k < K; ++k) {
        const __m256 av = _mm256_set1_ps(a[k]);
        const float* b = B + static_cast<std::size_t>(k) * static_cast<std::size_t>(ldb) + j;
        c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b), c0);
        c1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b + 8), c1);
        c2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b + 16), c2);
        c3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b + 24), c3);
      }
      float* c = C + static_cast<std::size_t>(i) * static_cast<std::size_t>(ldc) + j;
      _mm256_storeu_ps(c, c0);
      _mm256_storeu_ps(c + 8, c1);
      _mm256_storeu_ps(c + 16, c2);
      _mm256_storeu_ps(c + 24, c3);
    }
  }
  for (; j + 8 <= N; j += 8) {
    for (int i = 0; i < M; ++i) {
      const float* a = A + static_cast<std::size_t>(i) * static_cast<std::size_t>(lda);
      __m256 acc = _mm256_setzero_ps();
      for (int k = 0; k < K; ++k) {
        const __m256 av = _mm256_set1_ps(a[k]);
        const float* b = B + static_cast<std::size_t>(k) * static_cast<std::size_t>(ldb) + j;
        acc = _mm256_fmadd_ps(av, _mm256_loadu_ps(b), acc);
      }
      float* c = C + static_cast<std::size_t>(i) * static_cast<std::size_t>(ldc) + j;
      _mm256_storeu_ps(c, acc);
    }
  }
  for (; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      const float* a = A + static_cast<std::size_t>(i) * static_cast<std::size_t>(lda);
      float s = 0.f;
      for (int k = 0; k < K; ++k) {
        s += a[k] * B[static_cast<std::size_t>(k) * static_cast<std::size_t>(ldb) + j];
      }
      C[static_cast<std::size_t>(i) * static_cast<std::size_t>(ldc) + j] = s;
    }
  }
}

void gemm_nt_avx2(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
                  int ldc) {
  for (int i = 0; i < M; ++i) {
    const float* a = A + static_cast<std::size_t>(i) * static_cast<std::size_t>(lda);
    float* c = C + static_cast<std::size_t>(i) * static_cast<std::size_t>(ldc);
    int j = 0;
    for (; j + 4 <= N; j += 4) {
      __m256 acc0 = _mm256_setzero_ps();
      __m256 acc1 = _mm256_setzero_ps();
      __m256 acc2 = _mm256_setzero_ps();
      __m256 acc3 = _mm256_setzero_ps();
      int k = 0;
      for (; k + 8 <= K; k += 8) {
        const __m256 av = _mm256_loadu_ps(a + k);
        acc0 =
            _mm256_fmadd_ps(av, _mm256_loadu_ps(B + static_cast<std::size_t>(j) * ldb + k), acc0);
        acc1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(B + static_cast<std::size_t>(j + 1) * ldb + k),
                               acc1);
        acc2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(B + static_cast<std::size_t>(j + 2) * ldb + k),
                               acc2);
        acc3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(B + static_cast<std::size_t>(j + 3) * ldb + k),
                               acc3);
      }
      float s0 = hsum256(acc0);
      float s1 = hsum256(acc1);
      float s2 = hsum256(acc2);
      float s3 = hsum256(acc3);
      for (; k < K; ++k) {
        const float av = a[k];
        s0 += av * B[static_cast<std::size_t>(j) * ldb + k];
        s1 += av * B[static_cast<std::size_t>(j + 1) * ldb + k];
        s2 += av * B[static_cast<std::size_t>(j + 2) * ldb + k];
        s3 += av * B[static_cast<std::size_t>(j + 3) * ldb + k];
      }
      c[j] = s0;
      c[j + 1] = s1;
      c[j + 2] = s2;
      c[j + 3] = s3;
    }
    for (; j < N; ++j) {
      const float* b = B + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb);
      float s = 0.f;
      for (int k = 0; k < K; ++k)
        s += a[k] * b[k];
      c[j] = s;
    }
  }
}

} // namespace gpt2

#endif
