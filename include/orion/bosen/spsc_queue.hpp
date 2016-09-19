#pragma once

#include <atomic>
#include <memory>
#include <orion/noncopyable.hpp>

namespace orion {
namespace bosen {

template<typename Element>
class SPSCQueue {
 private:
  std::atomic_int begin_;
  std::atomic_int end_;
  std::unique_ptr<Element[]> elements_;
  const int capacity_;
 public:
  SPSCQueue(int capacity):
      begin_(0),
      end_(0),
      elements_(new Element[capacity]),
      capacity_(capacity) { }

  ~SPSCQueue() { }

  DISALLOW_COPY(SPSCQueue);

  int GetSize() const {
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
    int end = end_.load(std::memory_order_relaxed);
    int end_after_push = (end + 1) % capacity_;
    if (end_after_push == begin) {
      return false;
    }

    elements_[end] = element;
    end_.store(end_after_push, std::memory_order_release);
    return true;
  }

  bool Pull(Element *to_return) {
    int begin = begin_.load(std::memory_order_relaxed);
    int end = end_.load(std::memory_order_acquire);
    if (begin == end) return false;
    int begin_after_pull = (begin + 1) % capacity_;
    *to_return = elements_[begin];

    begin_.store(begin_after_pull, std::memory_order_release);
    return true;
  }
};

}
}
