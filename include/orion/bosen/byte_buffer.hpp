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
  size_t GetCapacity() const;
};

}
}
