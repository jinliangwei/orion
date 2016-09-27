#pragma once

#include <memory>
#include <orion/bosen/executor.hpp>
#include <orion/bosen/worker_runtime.hpp>
#include <orion/helper.hpp>
#include <cstddef>

namespace orion {
namespace bosen {

class Worker {
 private:
  const size_t kCommBuffCapacity_;
  const uint64_t kListenIp_;
  const int32_t kListenPort_;
  const size_t kNumExecutorsPerWorker_;
  const int32_t kMyId_;

  std::unique_ptr<uint8_t[]> mem_;
  Executor* executor_;
  static const int32_t kExecutorPortSpan_ = 100;
  const size_t kNumWorkers_;
  const HostInfo* kHosts_;
 public:
  Worker(size_t comm_buff_capacity,
         uint64_t listen_ip,
         int32_t listen_port,
         size_t num_executors_per_worker,
         int32_t worker_id,
         size_t num_workers,
         const HostInfo* hosts):
      kCommBuffCapacity_(comm_buff_capacity),
      kListenIp_(listen_ip),
      kListenPort_(listen_port),
      kNumExecutorsPerWorker_(num_executors_per_worker),
      kMyId_(worker_id),
      mem_(std::make_unique<uint8_t[]>(
          num_executors_per_worker
          * sizeof(Executor))),
      kNumWorkers_(num_workers),
      kHosts_(hosts) { }
  ~Worker() { }

  DISALLOW_COPY(Worker);

  int operator() () {
    for (auto i = 0; i < kNumExecutorsPerWorker_; ++i) {
      new (mem_.get() + sizeof(Executor) * i) Executor(
          kCommBuffCapacity_,
          kListenIp_,
          kListenPort_ + kExecutorPortSpan_ * i,
          kMyId_ * kNumExecutorsPerWorker_ + i,
          kNumExecutorsPerWorker_,
          kNumExecutorsPerWorker_ * kNumWorkers_,
          kHosts_);
    }

    executor_ = reinterpret_cast<Executor*>(mem_.get());

    for (auto i = 0; i < kNumExecutorsPerWorker_; ++i) {
      executor_[i]();
    }
    return 0;
  }

  void join_all() {
    for (auto i = 0; i < kNumExecutorsPerWorker_; ++i) {
      executor_[i].join();
      executor_[i].~Executor();
    }
  }
};
}
}
