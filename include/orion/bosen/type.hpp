#pragma once

#include <string>
#include <stdint.h>

namespace orion {
namespace bosen {

namespace type {

enum class PrimitiveType {
  kVoid = 0,
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

int SizeOf(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::kVoid:
      {
        return 0;
      }
    case PrimitiveType::kInt8:
      {
        return sizeof(int8_t);
      }
    case PrimitiveType::kUInt8:
      {
        return sizeof(uint8_t);
      }
    case PrimitiveType::kInt16:
      {
        return sizeof(int16_t);
      }
    case PrimitiveType::kUInt16:
      {
        return sizeof(uint16_t);
      }
    case PrimitiveType::kInt32:
      {
        return sizeof(int32_t);
      }
    case PrimitiveType::kUInt32:
      {
        return sizeof(uint32_t);
      }
    case PrimitiveType::kInt64:
      {
        return sizeof(int64_t);
      }
    case PrimitiveType::kUInt64:
      {
        return sizeof(uint64_t);
      }
    case PrimitiveType::kString:
      {
        return -1;
      }
    default:
      return 0;
  }
}

}

}
}
