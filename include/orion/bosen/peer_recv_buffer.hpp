#pragma once
#include <unordered_map>

#include <orion/bosen/byte_buffer.hpp>

namespace orion {
namespace bosen {

struct PeerRecvRepartitionDistArrayDataBuffer {
  int32_t dist_array_id {0};
  size_t num_executors_received {0};
  std::unordered_map<int32_t, ByteBuffer> byte_buffs;
};

}
}
