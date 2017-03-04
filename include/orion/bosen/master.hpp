#pragma once

#include <glog/logging.h>
#include <thread>
#include <orion/bosen/master_thread.hpp>
#include <orion/bosen/config.hpp>

namespace orion {
namespace bosen {

class Master {
 private:
  MasterThread master_thread_;
  std::thread runner_;
 public:
  Master(const Config &config):
      master_thread_(config) { }
  ~Master() { }

  void Run() {
    LOG(INFO) << "here 4";
    runner_ = std::thread(
        &MasterThread::operator(),
        &master_thread_);
  }

  void WaitUntilExit() {
    runner_.join();
  }
};

}
}
