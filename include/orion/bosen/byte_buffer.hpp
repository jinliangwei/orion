#pragma once
#include <string>
#include <string.h>
#include <glog/logging.h>

#include <orion/bosen/blob.hpp>

namespace orion {
namespace bosen {

class ByteBuffer {
 private:
  Blob buff_;
  size_t size_ {0};
 public:
  ByteBuffer();
  ~ByteBuffer();
  void Reset(size_t capacity);
  uint8_t* GetAvailMem();
  void IncSize(size_t size);
  uint8_t* GetBytes();
  size_t GetSize() const;
};

ByteBuffer::ByteBuffer() { }
ByteBuffer::~ByteBuffer() { }

uint8_t*
ByteBuffer::GetBytes() {
  return buff_.data();
}

void
ByteBuffer::Reset(size_t capacity) {
  size_ = 0;
  buff_.clear();
  buff_.reserve(capacity);
}

uint8_t*
ByteBuffer::GetAvailMem() {
  return buff_.data() + size_;
}

void
ByteBuffer::IncSize(size_t size) {
  size_ += size;
}

size_t
ByteBuffer::GetSize() const {
  return size_;
}

}
}
