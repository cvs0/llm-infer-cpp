#include "gpt2/unicode.hpp"

namespace gpt2 {

namespace {

#include "unicode_tables.inc"

bool in_ranges(const CpRange* ranges, int n, uint32_t cp) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (cp < ranges[mid].begin) {
      hi = mid;
    } else if (cp >= ranges[mid].end) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

} // namespace

bool is_unicode_letter(uint32_t cp) {
  return in_ranges(kLetterRanges, kLetterRangesCount, cp);
}

bool is_unicode_number(uint32_t cp) {
  return in_ranges(kNumberRanges, kNumberRangesCount, cp);
}

bool is_unicode_space(uint32_t cp) {
  return in_ranges(kSpaceRanges, kSpaceRangesCount, cp);
}

} // namespace gpt2
