#pragma once

#include <string>
#include <stdint.h>
#include <julia.h>

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
    kFloat32 = 9,
    kFloat64 = 10,
    kString = 11
};

int SizeOf(PrimitiveType type);

jl_datatype_t *GetJlDataType(PrimitiveType type);

}

}
}
