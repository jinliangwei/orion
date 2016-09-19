#pragma once

#include <atomic>
#include <memory>
#include <orion/noncopyable.hpp>
#include <orion/bosen/spsc_queue.hpp>
#include <glog/logging.h>
#include <iostream>
#include <mutex>

namespace orion {
namespace bosen {

template<typename Element>
class SPMCQueue {
 private:
  std::mutex consumer_mtx_;
  SPSCQueue<Element> queue_;
 public:
  SPMCQueue(int capacity):
      queue_(capacity) { }

  ~SPMCQueue() { }

  DISALLOW_COPY(SPMCQueue);

  int GetSize() const {
    return queue_.GetSize();
  }

  bool Push(const Element &element) {
    return queue_.Push(element);
  }

  bool Pull(Element *to_return) {
    std::lock_guard<std::mutex> consumer_lock(consumer_mtx_);
    return queue_.Pull(to_return);
  }
};

}
}
