#pragma once

#include <unordered_map>
#include <orion/bosen/dist_array_buffer_info.hpp>
#include <orion/bosen/dist_array.hpp>

namespace orion {
namespace bosen {

class ServerExecForLoop {
 private:
  const int32_t kServerId;
  const int32_t kNumExecutors;
  std::unordered_map<int32_t, DistArray> *dist_arrays_;
  std::unordered_map<int32_t, DistArray> *dist_array_buffers_;
  std::unordered_map<int32_t, DistArray*> helper_dist_arrays_;
  const std::unordered_map<int32_t, DistArrayBufferInfo> &dist_array_buffer_info_map_;
  size_t completed_executors_ { 0 };
 public:
  ServerExecForLoop(
      int32_t server_id,
      size_t num_executors,
      std::unordered_map<int32_t, DistArray> *dist_arrays,
      std::unordered_map<int32_t, DistArray> *dist_array_buffers,
      const std::unordered_map<int32_t, DistArrayBufferInfo> &dist_array_buffer_info_map,
      const int32_t *global_indexed_dist_array_ids,
      size_t num_global_indexed_dist_arrays,
      const int32_t *dist_array_buffer_ids,
      size_t num_dist_array_buffers);
  ~ServerExecForLoop();

  void PrepareHelperDistArrays();
  void DeserializeAndApplyDistArrayCaches(uint8_t* bytes);
  void DeserializeAndApplyDistArrayBuffers(uint8_t* bytes);
  bool NotifyExecForLoopDone();
};

}
}
