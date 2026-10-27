#include "gpt2/gemm.hpp"

#include <algorithm>
#include <cstddef>

namespace gpt2 {

void gemm_nn_scalar(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
                    int ldc);
void gemm_nt_scalar(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
                    int ldc);

#if defined(GPT2_COMPILE_AVX2) && GPT2_COMPILE_AVX2
void gemm_nn_avx2(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
                  int ldc);
void gemm_nt_avx2(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
                  int ldc);
#endif

namespace {

constexpr std::size_t kMinParallelWork = 1000000;

using NnKernel = void (*)(int, int, int, const float*, int, const float*, int, float*, int);

void split_range(int n, int parts, int part, int* begin, int* end) {
  const int base = n / parts;
  const int rem = n % parts;
  if (part < rem) {
    *begin = part * (base + 1);
    *end = *begin + base + 1;
  } else {
    *begin = rem * (base + 1) + (part - rem) * base;
    *end = *begin + base;
  }
}

struct ColCtx {
  int M;
  int N;
  int K;
  int lda;
  int ldb;
  int ldc;
  int align;
  int nblocks;
  int parts;
  const float* A;
  const float* B;
  float* C;
  NnKernel kernel;
  bool nt;
};

void col_job(int begin, int end, void* opaque) {
  auto* ctx = static_cast<ColCtx*>(opaque);
  for (int part = begin; part < end; ++part) {
    int b0 = 0;
    int b1 = 0;
    split_range(ctx->nblocks, ctx->parts, part, &b0, &b1);
    const int j0 = b0 * ctx->align;
    const int j1 = std::min(ctx->N, b1 * ctx->align);
    if (j0 >= j1)
      continue;
    if (ctx->nt) {
      ctx->kernel(ctx->M, j1 - j0, ctx->K, ctx->A, ctx->lda,
                  ctx->B + static_cast<std::size_t>(j0) * static_cast<std::size_t>(ctx->ldb),
                  ctx->ldb, ctx->C + j0, ctx->ldc);
    } else {
      ctx->kernel(ctx->M, j1 - j0, ctx->K, ctx->A, ctx->lda, ctx->B + j0, ctx->ldb, ctx->C + j0,
                  ctx->ldc);
    }
  }
}

void launch(bool nt, int M, int N, int K, const float* A, int lda, const float* B, int ldb,
            float* C, int ldc, bool simd, ThreadPool* pool, NnKernel scalar_kernel,
            NnKernel avx_kernel) {
  if (M <= 0 || N <= 0)
    return;
  if (K <= 0) {
    for (int i = 0; i < M; ++i) {
      float* row = C + static_cast<std::size_t>(i) * static_cast<std::size_t>(ldc);
      for (int j = 0; j < N; ++j)
        row[j] = 0.f;
    }
    return;
  }

  bool use_avx = false;
#if defined(GPT2_COMPILE_AVX2) && GPT2_COMPILE_AVX2
  use_avx = simd && avx_kernel != nullptr && cpu_supports_avx2();
#else
  (void)simd;
  (void)avx_kernel;
#endif
  NnKernel kernel = use_avx ? avx_kernel : scalar_kernel;
  const int align = use_avx ? 32 : 1;
  const std::size_t work =
      static_cast<std::size_t>(M) * static_cast<std::size_t>(N) * static_cast<std::size_t>(K);
  int parts = 1;
  if (pool != nullptr && work >= kMinParallelWork)
    parts = pool->size();
  int nblocks = (N + align - 1) / align;
  if (parts > nblocks)
    parts = nblocks;
  if (parts <= 1 || pool == nullptr) {
    kernel(M, N, K, A, lda, B, ldb, C, ldc);
    return;
  }

  ColCtx ctx;
  ctx.M = M;
  ctx.N = N;
  ctx.K = K;
  ctx.lda = lda;
  ctx.ldb = ldb;
  ctx.ldc = ldc;
  ctx.align = align;
  ctx.nblocks = nblocks;
  ctx.parts = parts;
  ctx.A = A;
  ctx.B = B;
  ctx.C = C;
  ctx.kernel = kernel;
  ctx.nt = nt;
  pool->parallel_range(parts, &col_job, &ctx);
}

} // namespace

bool cpu_supports_avx2() {
#if defined(__x86_64__)
#if defined(__GNUC__) && !defined(__clang__)
  __builtin_cpu_init();
#endif
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

void gemm_nn(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
             int ldc, bool simd, ThreadPool* pool) {
  NnKernel avx = nullptr;
#if defined(GPT2_COMPILE_AVX2) && GPT2_COMPILE_AVX2
  avx = &gemm_nn_avx2;
#endif
  launch(false, M, N, K, A, lda, B, ldb, C, ldc, simd, pool, &gemm_nn_scalar, avx);
}

void gemm_nt(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
             int ldc, bool simd, ThreadPool* pool) {
  NnKernel avx = nullptr;
#if defined(GPT2_COMPILE_AVX2) && GPT2_COMPILE_AVX2
  avx = &gemm_nt_avx2;
#endif
  launch(true, M, N, K, A, lda, B, ldb, C, ldc, simd, pool, &gemm_nt_scalar, avx);
}

} // namespace gpt2
