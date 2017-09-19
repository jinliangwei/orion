#include <orion/bosen/type.hpp>

namespace orion {
namespace bosen {
namespace type {
int SizeOf(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::kVoid:
      {
        return 0;
      }
    case PrimitiveType::kInt8:
      {
        return 1;
      }
    case PrimitiveType::kUInt8:
      {
        return 1;
      }
    case PrimitiveType::kInt16:
      {
        return 2;
      }
    case PrimitiveType::kUInt16:
      {
        return 2;
      }
    case PrimitiveType::kInt32:
      {
        return 4;
      }
    case PrimitiveType::kUInt32:
      {
        return 4;
      }
    case PrimitiveType::kInt64:
      {
        return 4;
      }
    case PrimitiveType::kUInt64:
      {
        return 4;
      }
    case PrimitiveType::kFloat32:
      {
        return 4;
      }
    case PrimitiveType::kFloat64:
      {
        return 8;
      }
    case PrimitiveType::kString:
      {
        return -1;
      }
    default:
      return 0;
  }
}

jl_datatype_t *GetJlDataType(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::kVoid:
      {
        return jl_void_type;
      }
    case PrimitiveType::kInt8:
      {
        return jl_int8_type;
      }
    case PrimitiveType::kUInt8:
      {
        return jl_uint8_type;
      }
    case PrimitiveType::kInt16:
      {
        return jl_int16_type;
      }
    case PrimitiveType::kUInt16:
      {
        return jl_uint16_type;
      }
    case PrimitiveType::kInt32:
      {
        return jl_int32_type;
      }
    case PrimitiveType::kUInt32:
      {
        return jl_uint32_type;
      }
    case PrimitiveType::kInt64:
      {
        return jl_int64_type;
      }
    case PrimitiveType::kUInt64:
      {
        return jl_uint64_type;
      }
    case PrimitiveType::kFloat32:
      {
        return jl_float32_type;
      }
    case PrimitiveType::kFloat64:
      {
        return jl_float64_type;
      }
    case PrimitiveType::kString:
      {
        return jl_string_type;
      }
    default:
      return 0;
  }
}

}
}
}
