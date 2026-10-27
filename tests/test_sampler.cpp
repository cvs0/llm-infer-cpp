#include "gpt2/sampler.hpp"

#include <random>
#include <vector>

#include <gtest/gtest.h>

TEST(Sampler, GreedyIsArgmax) {
  const float logits[5] = {0.1f, 3.f, 3.f, -1.f, 2.f};
  std::mt19937 rng(1);
  EXPECT_EQ(gpt2::sample_token(logits, 5, 0.f, 0, rng), 1);
  EXPECT_EQ(gpt2::sample_token(logits, 5, -1.f, 2, rng), 1);
}

TEST(Sampler, TopKStaysInsideSet) {
  const float logits[6] = {0.f, 5.f, 4.f, 1.f, 3.f, -2.f};
  std::mt19937 rng(7);
  for (int i = 0; i < 200; ++i) {
    const int id = gpt2::sample_token(logits, 6, 1.f, 3, rng);
    EXPECT_TRUE(id == 1 || id == 2 || id == 4) << id;
  }
}

TEST(Sampler, SeedRepeats) {
  const float logits[4] = {0.2f, 0.1f, 0.4f, 0.3f};
  std::mt19937 a(99);
  std::mt19937 b(99);
  std::vector<int> left;
  std::vector<int> right;
  for (int i = 0; i < 16; ++i) {
    left.push_back(gpt2::sample_token(logits, 4, 0.8f, 0, a));
    right.push_back(gpt2::sample_token(logits, 4, 0.8f, 0, b));
  }
  EXPECT_EQ(left, right);
}

TEST(Sampler, Top1IsArgmax) {
  const float logits[4] = {1.f, 4.f, 2.f, 3.f};
  std::mt19937 rng(3);
  EXPECT_EQ(gpt2::sample_token(logits, 4, 1.5f, 1, rng), 1);
}
