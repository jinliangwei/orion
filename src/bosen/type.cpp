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

}
}
}
