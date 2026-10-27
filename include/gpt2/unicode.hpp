#pragma once

#include <cstdint>

namespace gpt2 {

struct CpRange {
  uint32_t begin;
  uint32_t end;
};

bool is_unicode_letter(uint32_t cp);
bool is_unicode_number(uint32_t cp);
bool is_unicode_space(uint32_t cp);

} // namespace gpt2
