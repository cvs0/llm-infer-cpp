#include "gpt2/ops.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

TEST(Ops, LayerNormMatchesDouble) {
  const float x[4] = {1.f, 2.f, 3.f, 4.f};
  const float w[4] = {1.f, -1.f, 0.5f, 2.f};
  const float b[4] = {0.f, 1.f, -1.f, 0.25f};
  const float eps = 1e-5f;
  float y[4];
  gpt2::layer_norm(x, w, b, y, 4, eps);

  double sum = 0.0;
  for (float v : x)
    sum += v;
  const double mean = sum / 4.0;
  double var = 0.0;
  for (float v : x) {
    const double d = static_cast<double>(v) - mean;
    var += d * d;
  }
  var /= 4.0;
  const float inv = static_cast<float>(1.0 / std::sqrt(var + static_cast<double>(eps)));
  for (int i = 0; i < 4; ++i) {
    const float d = static_cast<float>(static_cast<double>(x[i]) - mean);
    const float expect = d * inv * w[i] + b[i];
    EXPECT_FLOAT_EQ(y[i], expect);
  }
}

TEST(Ops, GeluZeroAndFormula) {
  float z = 0.f;
  gpt2::gelu_inplace(&z, 1);
  EXPECT_FLOAT_EQ(z, 0.f);

  float x = 1.5f;
  gpt2::gelu_inplace(&x, 1);
  const double xd = 1.5;
  const double k1 = std::sqrt(2.0 / 3.14159265358979323846);
  const double u = k1 * (xd + 0.044715 * xd * xd * xd);
  const double g = 0.5 * xd * (1.0 + std::tanh(u));
  EXPECT_NEAR(x, static_cast<float>(g), 2e-6f);
}

TEST(Ops, SoftmaxSumsToOne) {
  float x[3] = {1.f, 2.f, 3.f};
  gpt2::softmax_inplace(x, 3);
  EXPECT_NEAR(x[0] + x[1] + x[2], 1.f, 1e-6f);
  EXPECT_GT(x[2], x[1]);
  EXPECT_GT(x[1], x[0]);
}

TEST(Ops, BiasAndResidual) {
  float x[6] = {1, 2, 3, 4, 5, 6};
  const float bias[3] = {10, 20, 30};
  gpt2::add_bias_inplace(x, bias, 2, 3);
  EXPECT_FLOAT_EQ(x[0], 11.f);
  EXPECT_FLOAT_EQ(x[4], 25.f);
  const float y[6] = {1, 1, 1, 1, 1, 1};
  gpt2::add_inplace(x, y, 6);
  EXPECT_FLOAT_EQ(x[0], 12.f);
}
