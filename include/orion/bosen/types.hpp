#pragma once

#include <string>
#include <stdint.h>

namespace orion {
namespace bosen {

namespace type {

enum class PrimitiveType {
  kInt8 = 1,
    kUInt8 = 2,
    kInt16 = 3,
    kUInt16 = 4,
    kInt32 = 5,
    kUInt32 = 6,
    kInt64 = 7,
    kUInt64 = 8,
    kString = 9
};

size_t SizeOf(PrimitiveType type) {
  switch (type) {
    case kInt8:
      {
        return sizeof(int8_t);
      }
    case kUInt8:
      {
        return sizeof(uint8_t);
      }
    case kInt16:
      {
        return sizeof(int16_t);
      }
    case kUInt16:
      {
        return sizeof(uint16_t);
      }
    case kInt32:
      {
        return sizeof(int32_t);
      }
    case kUInt32:
      {
        return sizeof(uint32_t);
      }
    case kInt64:
      {
        return sizeof(int64_t);
      }
    case kUInt64:
      {
        return sizeof(uint64_t);
      }
    case kString:
      {
        return 0;
      }
    default:
      return 0;
  }
}

}

}
}
