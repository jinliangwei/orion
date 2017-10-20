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

struct PeerRecvDistArrayDataBuffer {
  int32_t dist_array_id;
  int32_t partition_id;
  uint8_t *data;
  size_t expected_size;
  size_t received_size;
};

struct PeerRecvExecForLoopDistArrayDataBuffer {
  bool is_executor_expecting {false};
  std::vector<PeerRecvDistArrayDataBuffer> complete_buffers;
  std::unordered_map<int32_t, PeerRecvDistArrayDataBuffer> incomplete_buffers;
};

}
}
