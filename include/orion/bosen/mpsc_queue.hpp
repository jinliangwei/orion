#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <orion/noncopyable.hpp>
#include <glog/logging.h>
#include <iostream>

namespace orion {
namespace bosen {

template<typename Element>
class MPSCQueue {
 private:
  std::atomic_int begin_;
  std::atomic_int end_;
  std::unique_ptr<Element[]> elements_;
  std::unique_ptr<std::atomic_bool[]> flags_;
  std::mutex prod_mtx_;
  const int capacity_;
 public:
  MPSCQueue(int capacity):
      begin_(0),
      end_(0),
      elements_(new Element[capacity]),
      flags_(new std::atomic_bool[capacity]),
      capacity_(capacity) {
    for (int i = 0; i < capacity_; ++i) {
      flags_[i] = false;
    }
    CHECK (begin_.is_lock_free());
  }

  ~MPSCQueue() { }

  DISALLOW_COPY(MPSCQueue);

  int GetSize() {
    int begin = begin_.load(std::memory_order_relaxed);
    int end = end_.load(std::memory_order_relaxed);
    int size = 0;
    if (begin <= end) {
      size = end - begin;
    } else {
      size = end + capacity_ - begin;
    }

    return size;
  }

  bool Push(const Element &element) {
    int begin = begin_.load(std::memory_order_acquire);
    int end = end_.load();
    int end_after_push = (end + 1) % capacity_;
    if (end_after_push == begin) {
      return false;
    }

    while (!end_.compare_exchange_weak(end, end_after_push)) {
      end = end_.load();
      int end_after_push = (end + 1) % capacity_;
      if (end_after_push == begin) {
        return false;
      }
    }

    elements_[end] = element;
    flags_[end].store(true, std::memory_order_release);
    return true;
  }

  bool Pull(Element *to_return) {
    int begin = begin_.load(std::memory_order_relaxed);
    int end = end_.load(std::memory_order_acquire);
    if (begin == end) {
      LOG(INFO) << "b = " << begin
                << " e = " << end << std::endl;
      return false;
    }
    while (!flags_[begin].load(std::memory_order_acquire)) {
      LOG(INFO) << __func__ << begin << std::endl;
    }

    int begin_after_pull = (begin + 1) % capacity_;
    *to_return = elements_[begin];


    flags_[begin].store(false, std::memory_order_release);
    begin_.store(begin_after_pull, std::memory_order_release);
    return true;
  }
};

}
}
