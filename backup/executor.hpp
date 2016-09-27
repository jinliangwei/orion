#pragma once

#include <orion/bosen/master.hpp>
#include <orion/bosen/worker_runtime.hpp>
#include <orion/noncopyable.hpp>
#include <thread>
#include <memory>

namespace orion {
namespace bosen {

class Executor {
 private:
  Master master_;
  std::unique_ptr<std::thread> master_thread_;
  const int32_t kMyId_;

 public:
  Executor(
      size_t comm_buff_capacity,
      uint64_t master_listen_ip,
      int32_t master_listen_port,
      int32_t executor_id,
      size_t num_local_executors,
      size_t num_total_executors,
      const HostInfo* hosts):
      master_(comm_buff_capacity,
              master_listen_ip,
              master_listen_port,
              executor_id,
              num_local_executors,
              num_total_executors,
              hosts),
    kMyId_(executor_id) { }
  ~Executor() { }
  DISALLOW_COPY(Executor);

  int operator () () {
    master_thread_ = std::make_unique<std::thread>(
        &Master::operator(),
        &master_);
    master_.WaitUntilEvent(Master::Event::kExecutorReadyAck);
    return 0;
  }

  int join() {
    master_thread_->join();
    return 0;
  }
};

}
}
