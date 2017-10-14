#include <orion/bosen/byte_buffer.hpp>

namespace orion {
namespace bosen {

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
  buff_.resize(capacity);
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

size_t
ByteBuffer::GetCapacity() const {
  return buff_.size();
}


}
}
