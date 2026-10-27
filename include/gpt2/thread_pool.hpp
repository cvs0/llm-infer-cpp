#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

namespace gpt2 {

// Fixed-size pool. parallel_range is not re-entrant and not safe to call
// from more than one thread at a time.
class ThreadPool {
public:
  using RangeFn = void (*)(int begin, int end, void* ctx);

  explicit ThreadPool(int nthreads);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const noexcept { return nthreads_; }

  // Invoke fn on contiguous subranges that cover [0, n). With one thread, fn
  // runs inline on the caller. With more threads, every worker participates,
  // including workers whose subrange is empty.
  void parallel_range(int n, RangeFn fn, void* ctx);

private:
  void worker(int id);

  int nthreads_ = 1;
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  RangeFn fn_ = nullptr;
  void* ctx_ = nullptr;
  int n_ = 0;
  int finished_ = 0;
  int epoch_ = 0;
  bool stop_ = false;
};

} // namespace gpt2
