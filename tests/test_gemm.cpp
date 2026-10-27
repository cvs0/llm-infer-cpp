#include "gpt2/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace {

void expect_near(const std::vector<float>& a, const std::vector<float>& b, float atol) {
  ASSERT_EQ(a.size(), b.size());
  float max_abs = 0.f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(a[i] - b[i]));
  }
  EXPECT_LT(max_abs, atol) << max_abs;
}

} // namespace

TEST(Gemm, IntegerExactNnAndNt) {
  const float A[2 * 3] = {1, 2, 3, 4, 5, 6};
  const float B[3 * 2] = {1, 0, 0, 1, 1, 1};
  float C[4] = {};
  gpt2::gemm_nn(2, 2, 3, A, 3, B, 2, C, 2, false, nullptr);
  EXPECT_FLOAT_EQ(C[0], 4.f);
  EXPECT_FLOAT_EQ(C[1], 5.f);
  EXPECT_FLOAT_EQ(C[2], 10.f);
  EXPECT_FLOAT_EQ(C[3], 11.f);

  float C2[4] = {};
  gpt2::gemm_nn(2, 2, 3, A, 3, B, 2, C2, 2, true, nullptr);
  EXPECT_FLOAT_EQ(C2[0], 4.f);
  EXPECT_FLOAT_EQ(C2[1], 5.f);
  EXPECT_FLOAT_EQ(C2[2], 10.f);
  EXPECT_FLOAT_EQ(C2[3], 11.f);

  // B rows are the N dimension for NT: Bnt[n, k] = Bnn[k, n], so Bnt is 2 x 3.
  const float Bnt[2 * 3] = {1, 0, 1, 0, 1, 1};
  float Cnt[4] = {};
  gpt2::gemm_nt(2, 2, 3, A, 3, Bnt, 3, Cnt, 2, false, nullptr);
  EXPECT_FLOAT_EQ(Cnt[0], 4.f);
  EXPECT_FLOAT_EQ(Cnt[1], 5.f);
  EXPECT_FLOAT_EQ(Cnt[2], 10.f);
  EXPECT_FLOAT_EQ(Cnt[3], 11.f);
}

TEST(Gemm, StrideAndTail) {
  const int M = 3;
  const int N = 7;
  const int K = 5;
  const int lda = 9;
  std::vector<float> A(static_cast<std::size_t>(M * lda), 0.25f);
  std::vector<float> B(static_cast<std::size_t>(K * N));
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k)
      A[static_cast<std::size_t>(i * lda + k)] = 0.1f * (i + 1) * (k + 1);
  }
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < N; ++j)
      B[static_cast<std::size_t>(k * N + j)] = 0.05f * (k - j);
  }
  std::vector<float> C(static_cast<std::size_t>(M * N));
  std::vector<float> Cref(static_cast<std::size_t>(M * N), 0.f);
  gpt2::gemm_nn(M, N, K, A.data(), lda, B.data(), N, C.data(), N, false, nullptr);
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        Cref[static_cast<std::size_t>(i * N + j)] +=
            A[static_cast<std::size_t>(i * lda + k)] * B[static_cast<std::size_t>(k * N + j)];
      }
    }
  }
  expect_near(C, Cref, 1e-5f);
}

TEST(Gemm, Avx2MatchesScalar) {
  if (!gpt2::cpu_supports_avx2())
    GTEST_SKIP() << "no avx2";
  const int M = 5;
  const int N = 70;
  const int K = 48;
  std::vector<float> A(static_cast<std::size_t>(M * K));
  std::vector<float> B(static_cast<std::size_t>(K * N));
  std::vector<float> Bnt(static_cast<std::size_t>(N * K));
  for (std::size_t i = 0; i < A.size(); ++i) {
    A[i] = 0.01f * static_cast<float>(static_cast<int>(i * 17 % 50) - 25);
  }
  for (std::size_t i = 0; i < B.size(); ++i) {
    B[i] = 0.02f * static_cast<float>(static_cast<int>(i * 13 % 40) - 20);
  }
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k)
      Bnt[static_cast<std::size_t>(n * K + k)] = B[static_cast<std::size_t>(k * N + n)];
  }
  std::vector<float> Cs(static_cast<std::size_t>(M * N));
  std::vector<float> Cv(static_cast<std::size_t>(M * N));
  gpt2::gemm_nn(M, N, K, A.data(), K, B.data(), N, Cs.data(), N, false, nullptr);
  gpt2::gemm_nn(M, N, K, A.data(), K, B.data(), N, Cv.data(), N, true, nullptr);
  expect_near(Cs, Cv, 2e-3f);

  std::vector<float> Ns(static_cast<std::size_t>(M * N));
  std::vector<float> Nv(static_cast<std::size_t>(M * N));
  gpt2::gemm_nt(M, N, K, A.data(), K, Bnt.data(), K, Ns.data(), N, false, nullptr);
  gpt2::gemm_nt(M, N, K, A.data(), K, Bnt.data(), K, Nv.data(), N, true, nullptr);
  expect_near(Ns, Nv, 2e-3f);
}

TEST(Gemm, ThreadsMatchSingle) {
  const int M = 32;
  const int N = 100;
  const int K = 128;
  std::vector<float> A(static_cast<std::size_t>(M * K));
  std::vector<float> B(static_cast<std::size_t>(K * N));
  for (std::size_t i = 0; i < A.size(); ++i)
    A[i] = 0.001f * static_cast<float>(static_cast<int>(i % 97) - 40);
  for (std::size_t i = 0; i < B.size(); ++i)
    B[i] = 0.002f * static_cast<float>(static_cast<int>(i % 53) - 20);
  gpt2::ThreadPool pool(4);
  for (bool simd : {false, true}) {
    if (simd && !gpt2::cpu_supports_avx2())
      continue;
    std::vector<float> one(static_cast<std::size_t>(M * N));
    std::vector<float> many(static_cast<std::size_t>(M * N));
    gpt2::gemm_nn(M, N, K, A.data(), K, B.data(), N, one.data(), N, simd, nullptr);
    gpt2::gemm_nn(M, N, K, A.data(), K, B.data(), N, many.data(), N, simd, &pool);
    ASSERT_EQ(one.size(), many.size());
    for (std::size_t i = 0; i < one.size(); ++i) {
      EXPECT_EQ(one[i], many[i]) << "simd=" << simd << " i=" << i;
      if (one[i] != many[i])
        break;
    }
  }
}

TEST(Gemm, ZeroKClearsC) {
  float A[4] = {1, 2, 3, 4};
  float B[4] = {1, 2, 3, 4};
  float C[4] = {9, 9, 9, 9};
  gpt2::gemm_nn(2, 2, 0, A, 2, B, 2, C, 2, false, nullptr);
  EXPECT_FLOAT_EQ(C[0], 0.f);
  EXPECT_FLOAT_EQ(C[3], 0.f);
}
