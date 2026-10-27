#include "gpt2/thread_pool.hpp"

#include <algorithm>

namespace gpt2 {

ThreadPool::ThreadPool(int nthreads) : nthreads_(std::max(1, nthreads)) {
  if (nthreads_ == 1)
    return;
  threads_.reserve(static_cast<std::size_t>(nthreads_));
  for (int i = 0; i < nthreads_; ++i) {
    threads_.emplace_back([this, i] { worker(i); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : threads_)
    t.join();
}

void ThreadPool::parallel_range(int n, RangeFn fn, void* ctx) {
  if (n <= 0)
    return;
  if (nthreads_ == 1 || threads_.empty()) {
    fn(0, n, ctx);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    fn_ = fn;
    ctx_ = ctx;
    n_ = n;
    finished_ = 0;
    ++epoch_;
  }
  cv_.notify_all();
  std::unique_lock<std::mutex> lock(mu_);
  done_cv_.wait(lock, [&] { return finished_ == nthreads_; });
  fn_ = nullptr;
  ctx_ = nullptr;
}

void ThreadPool::worker(int id) {
  int seen = 0;
  for (;;) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return stop_ || epoch_ != seen; });
    if (stop_)
      return;
    seen = epoch_;
    const int n = n_;
    const int chunk = (n + nthreads_ - 1) / nthreads_;
    const int begin = id * chunk;
    const int end = std::min(n, begin + chunk);
    RangeFn fn = fn_;
    void* ctx = ctx_;
    lock.unlock();
    if (begin < end && fn != nullptr)
      fn(begin, end, ctx);
    lock.lock();
    if (++finished_ == nthreads_)
      done_cv_.notify_one();
  }
}

} // namespace gpt2
