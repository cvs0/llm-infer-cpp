#include "gpt2/bpe.hpp"
#include "gpt2/tokenizer.hpp"
#include "gpt2/unicode.hpp"

#include <string>
#include <vector>

#include <gtest/gtest.h>

TEST(Unicode, AsciiAndLatin1) {
  EXPECT_TRUE(gpt2::is_unicode_letter('A'));
  EXPECT_TRUE(gpt2::is_unicode_letter('z'));
  EXPECT_FALSE(gpt2::is_unicode_letter('0'));
  EXPECT_FALSE(gpt2::is_unicode_letter('!'));
  EXPECT_TRUE(gpt2::is_unicode_number('0'));
  EXPECT_TRUE(gpt2::is_unicode_number(0xFF11)); // fullwidth digit one
  EXPECT_TRUE(gpt2::is_unicode_letter(0x00E9)); // e acute
  EXPECT_TRUE(gpt2::is_unicode_space(' '));
  EXPECT_TRUE(gpt2::is_unicode_space('\n'));
  EXPECT_TRUE(gpt2::is_unicode_space('\t'));
  EXPECT_TRUE(gpt2::is_unicode_space(0x00A0));
  EXPECT_FALSE(gpt2::is_unicode_space('!'));
}

TEST(ByteLevel, PublishedMapping) {
  EXPECT_EQ(gpt2::byte_to_codepoint(static_cast<unsigned char>('A')), 65u);
  EXPECT_EQ(gpt2::byte_to_codepoint(0), 256u);
  EXPECT_EQ(gpt2::byte_to_codepoint(32), 288u);
  EXPECT_EQ(gpt2::byte_to_codepoint(127), 289u);
  EXPECT_EQ(gpt2::byte_to_codepoint(33), 33u);
}

TEST(Pretokenize, MatchesRegexModule) {
  auto eq = [](const std::string& text, std::vector<std::string> expect) {
    EXPECT_EQ(gpt2::pretokenize(text), expect) << text;
  };
  eq("Hello, world!", {"Hello", ",", " world", "!"});
  eq("don't stop", {"don", "'t", " stop"});
  eq("I'm", {"I", "'m"});
  eq("we'll see", {"we", "'ll", " see"});
  eq("123 456", {"123", " 456"});
  eq(" Hello", {" Hello"});
  eq("a  b", {"a", " ", " b"});
  eq("end  ", {"end", "  "});
  eq("a\nb", {"a", "\n", "b"});
  eq("a\tb", {"a", "\t", "b"});
  eq("wait...", {"wait", "..."});
  eq(std::string("caf\xC3\xA9"), {std::string("caf\xC3\xA9")});
  eq("", {});
  eq("GPT-2", {"GPT", "-", "2"});
  eq("  ", {"  "});
  eq("\n\n", {"\n\n"});
  eq("foo\n", {"foo", "\n"});
  eq("'s", {"'s"});
  eq("123abc", {"123", "abc"});
  eq("abc123", {"abc", "123"});
  eq("it's", {"it", "'s"});
  eq("  x", {" ", " x"});
  eq("x \n y", {"x", " \n", " y"});
  eq("<|endoftext|>", {"<|", "endoftext", "|>"});
  eq(std::string("\xC2\xA0nbsp"), {std::string("\xC2\xA0"), "nbsp"});
  eq("price $5", {"price", " $", "5"});
  eq("foo\r\nbar", {"foo", "\r", "\n", "bar"});
}

TEST(Bpe, MergesLowestRankFirst) {
  gpt2::MergeRanks ranks;
  ranks[{"a", "b"}] = 0;
  ranks[{"ab", "c"}] = 1;
  ranks[{"b", "c"}] = 5;
  auto out = gpt2::bpe_merge({"a", "b", "c", "a", "b"}, ranks);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], "abc");
  EXPECT_EQ(out[1], "ab");
}

TEST(Bpe, StopsWhenNoPairIsListed) {
  gpt2::MergeRanks ranks;
  auto out = gpt2::bpe_merge({"x", "y"}, ranks);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], "x");
  EXPECT_EQ(out[1], "y");
}
