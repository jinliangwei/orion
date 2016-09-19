#pragma once

#include <memory>
#include <orion/bosen/executor.hpp>
#include <orion/bosen/worker_runtime.hpp>
#include <orion/helper.hpp>

namespace orion {
namespace bosen {

DECLARE_uint64(comm_buff_capacity);
DECLARE_string(worker_driver_ip);
DECLARE_int32(worker_driver_port);
DECLARE_string(worker_ip);
DECLARE_int32(worker_port);
DECLARE_int32(worker_num_executors_per_worker);
DECLARE_int32(worker_id);

class Worker {
 private:
  std::unique_ptr<uint8_t[]> mem_;
  Executor* executor_;
  const size_t num_executors_;
  static const int32_t kExecutorPortSpan = 100;
  const int32_t worker_id_ {0};

 public:
  Worker():
      mem_(std::make_unique<uint8_t[]>(
          FLAGS_worker_num_executors_per_worker
          * sizeof(Executor))),
      num_executors_(FLAGS_worker_num_executors_per_worker),
      worker_id_(FLAGS_worker_id) { }
  ~Worker() { }

  DISALLOW_COPY(Worker);

  int operator() () {
    for (auto i = 0; i < num_executors_; ++i) {
      new (mem_.get() + sizeof(Executor) * i) Executor(
          FLAGS_comm_buff_capacity,
          FLAGS_worker_driver_ip,
          FLAGS_worker_driver_port,
          FLAGS_worker_ip,
          FLAGS_worker_port + kExecutorPortSpan * i,
          worker_id_ * num_executors_ + i,
          FLAGS_worker_num_executors_per_worker);
    }

    executor_ = reinterpret_cast<Executor*>(mem_.get());

    for (auto i = 0; i < num_executors_; ++i) {
      executor_[i]();
    }

    for (auto i = 0; i < num_executors_; ++i) {
      executor_[i].join();
      executor_[i].~Executor();
    }
    return 0;
  }
};
}
}
