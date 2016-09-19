#pragma once

#include <mutex>
#include <condition_variable>
#include <queue>
#include <glog/logging.h>

namespace orion {
namespace bosen {

template<typename Element>
class BlockQueue {
 private:
  std::queue<Element> queue_;
  std::mutex mtx_;
  std::condition_variable cond_var_;
 public:
  BlockQueue() = default;
  void Push(const Element &element) {
    LOG(INFO) << "push!";
    std::unique_lock<std::mutex> lock(mtx_);
    queue_.push(element);
    cond_var_.notify_one();
  }

  Element Pop() {
    LOG(INFO) << "pop!";
    std::unique_lock<std::mutex> lock(mtx_);
    cond_var_.wait(lock, [this]{ return queue_.size() > 0; });
    auto ret = queue_.front();
    queue_.pop();
    return ret;
  }
};

}
}
