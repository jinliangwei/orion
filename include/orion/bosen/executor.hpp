#pragma once

#include <orion/bosen/master.hpp>
#include <orion/bosen/worker_runtime.hpp>
#include <orion/bosen/configs.hpp>
#include <orion/noncopyable.hpp>
#include <thread>
#include <memory>

namespace orion {
namespace bosen {

class Executor {
 private:
  Master master_;
  std::unique_ptr<std::thread> master_thread_;
  const int32_t executor_id_ {0};

 public:
  Executor(
      size_t comm_buff_capacity,
      const std::string &driver_listen_ip,
      int32_t driver_listen_port,
      const std::string &master_listen_ip,
      int32_t master_listen_port,
      int32_t executor_id,
      size_t num_local_executors):
      master_(comm_buff_capacity,
              driver_listen_ip,
              driver_listen_port,
              master_listen_ip,
              master_listen_port,
              executor_id,
              num_local_executors),
    executor_id_(executor_id) { }
  ~Executor() { }
  DISALLOW_COPY(Executor);

  int operator () () {
    master_thread_ = std::make_unique<std::thread>(
        &Master::operator(),
        &master_);
    return 0;
  }

  int join() {
    master_thread_->join();
    return 0;
  }
};

}
}
