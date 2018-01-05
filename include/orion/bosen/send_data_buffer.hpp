#pragma once
#include <utility>

namespace orion {
namespace bosen {

using SendDataBuffer = std::pair<uint8_t*, size_t>;
using ExecutorSendBufferMap = std::unordered_map<int32_t, SendDataBuffer>;

}
}
