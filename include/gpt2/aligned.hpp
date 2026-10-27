#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>

namespace gpt2 {

struct FreeDelete {
  void operator()(void* p) const noexcept { std::free(p); }
};

// 64-byte aligned float buffer. Empty buffers have a null data pointer.
class AlignedFloats {
public:
  AlignedFloats() = default;
  explicit AlignedFloats(std::size_t n) : n_(n) {
    if (n_ == 0)
      return;
    void* mem = nullptr;
    if (posix_memalign(&mem, 64, n_ * sizeof(float)) != 0)
      throw std::bad_alloc();
    ptr_.reset(static_cast<float*>(mem));
  }

  AlignedFloats(AlignedFloats&&) noexcept = default;
  AlignedFloats& operator=(AlignedFloats&&) noexcept = default;
  AlignedFloats(const AlignedFloats&) = delete;
  AlignedFloats& operator=(const AlignedFloats&) = delete;

  float* data() noexcept { return ptr_.get(); }
  const float* data() const noexcept { return ptr_.get(); }
  std::size_t size() const noexcept { return n_; }

private:
  std::size_t n_ = 0;
  std::unique_ptr<float, FreeDelete> ptr_;
};

} // namespace gpt2
