#pragma once

#include <orion/bosen/driver_thread.hpp>
#include <orion/bosen/driver_runtime.hpp>
#include <thread>

namespace orion {
namespace bosen {
class DriverTask;

class Driver {
 private:
  DriverThread runner_;
  std::unique_ptr<std::thread> runner_thread_;
 public:
  Driver(const DriverRuntimeConfig &runtime_config):
      runner_(runtime_config){ }
  ~Driver() { }

  int operator () () {
    runner_thread_ = std::make_unique<std::thread>(
        &bosen::DriverThread::operator(),
        &runner_);
    return 0;
  }

  void ScheduleTask(DriverTask *task) {
    runner_.ScheduleTask(task);
  }

  DriverTask* GetCompletedTask() {
    return runner_.GetCompletedTask();
  }

  void WaitUntilExit() {
    runner_thread_->join();
  }
};

}
}
