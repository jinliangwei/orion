#pragma once
#include <string>
#include <vector>
#include <cstring>

namespace orion {
namespace bosen {
struct DistArrayBufferInfo {
  const int32_t kDistArrayId;
  const std::string kApplyBufferFuncName;
  std::vector<int32_t> kHelperDistArrayIds;
  std::vector<int32_t> kHelperDistArrayBufferIds;
  DistArrayBufferInfo(int32_t dist_array_id,
                      const std::string &apply_buffer_func_name,
                      const int32_t *helper_dist_array_ids,
                      size_t num_helper_dist_arrays,
                      const int32_t *helper_dist_array_buffer_ids,
                      size_t num_helper_dist_array_buffers):
      kDistArrayId(dist_array_id),
      kApplyBufferFuncName(apply_buffer_func_name),
      kHelperDistArrayIds(num_helper_dist_arrays),
      kHelperDistArrayBufferIds(num_helper_dist_array_buffers) {
    memcpy(kHelperDistArrayIds.data(), helper_dist_array_ids,
           num_helper_dist_arrays * sizeof(int32_t));
    memcpy(kHelperDistArrayBufferIds.data(), helper_dist_array_buffer_ids,
           num_helper_dist_array_buffers * sizeof(int32_t));
  }
};

}
}
