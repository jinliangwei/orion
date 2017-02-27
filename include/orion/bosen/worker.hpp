#pragma once

#include <memory>
#include <vector>
#include <thread>
#include <glog/logging.h>
#include <orion/bosen/config.hpp>
#include <orion/bosen/executor_thread.hpp>

namespace orion {
namespace bosen {
class Worker {
 private:
  const size_t kNumLocalExecutors;
  std::vector<std::unique_ptr<ExecutorThread>> executor_thread_;
  std::vector<std::thread> runner_;
  const int32_t kWorkerId {0};

 public:
  Worker(const Config &config):
      kNumLocalExecutors(config.kNumExecutorsPerWorker),
      executor_thread_(kNumLocalExecutors),
      runner_(kNumLocalExecutors) {
    for (int i = 0; i < kNumLocalExecutors; i++) {
      executor_thread_[i] = std::make_unique<ExecutorThread>(config, i);
    }
  }
  ~Worker() { }

  void Run() {
    for (int i = 0; i < kNumLocalExecutors; i++) {
      LOG(INFO) << "running " << i;
      runner_[i] = std::thread(
          &ExecutorThread::operator(),
          executor_thread_[i].get());
    }
  }

  void WaitUntilExit() {
    for (int i = 0; i < kNumLocalExecutors; i++) {
      runner_[i].join();
    }
  }

};
}
}
