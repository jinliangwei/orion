#pragma once
#include <string>
#include <vector>
#include <cstring>

namespace orion {
namespace bosen {

enum class DistArrayBufferDelayMode {
  kDefault = 1,
    kMaxDelay = 2,
    kAuto = 3
};

struct DistArrayBufferDelayInfo {
  size_t delay;
  const size_t kMaxDelay;
  const DistArrayBufferDelayMode kDelayMode;

  DistArrayBufferDelayInfo(size_t max_delay,
                           DistArrayBufferDelayMode delay_mode):
      delay(0),
      kMaxDelay(max_delay),
      kDelayMode(delay_mode) { }
};

struct DistArrayBufferInfo {
  const int32_t kDistArrayId;
  const std::string kApplyBufferFuncName;
  std::vector<int32_t> kHelperDistArrayIds;
  std::vector<int32_t> kHelperDistArrayBufferIds;
  const DistArrayBufferDelayMode kDelayMode;
  const size_t kMaxDelay;
  DistArrayBufferInfo(int32_t dist_array_id,
                      const std::string &apply_buffer_func_name,
                      const int32_t *helper_dist_array_ids,
                      size_t num_helper_dist_arrays,
                      const int32_t *helper_dist_array_buffer_ids,
                      size_t num_helper_dist_array_buffers,
                      DistArrayBufferDelayMode delay_mode,
                      size_t max_delay):
      kDistArrayId(dist_array_id),
      kApplyBufferFuncName(apply_buffer_func_name),
      kHelperDistArrayIds(num_helper_dist_arrays),
      kHelperDistArrayBufferIds(num_helper_dist_array_buffers),
      kDelayMode(delay_mode),
      kMaxDelay(max_delay) {
    memcpy(kHelperDistArrayIds.data(), helper_dist_array_ids,
           num_helper_dist_arrays * sizeof(int32_t));
    memcpy(kHelperDistArrayBufferIds.data(), helper_dist_array_buffer_ids,
           num_helper_dist_array_buffers * sizeof(int32_t));
  }
};

}
}
