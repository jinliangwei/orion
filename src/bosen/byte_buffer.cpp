#include <orion/bosen/byte_buffer.hpp>

namespace orion {
namespace bosen {

ByteBuffer::ByteBuffer() { }
ByteBuffer::~ByteBuffer() { }

ByteBuffer::ByteBuffer(size_t buff_capacity):
    buff_(buff_capacity) { }

ByteBuffer::ByteBuffer(const ByteBuffer & other):
    buff_(other.buff_),
    size_(other.size_) { }

ByteBuffer::ByteBuffer(ByteBuffer && other):
    buff_(std::move(other.buff_)),
    size_(other.size_) { }

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
