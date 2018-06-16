#pragma once

#include <vector>
#include <utility>
#include <map>
#include <unordered_map>

namespace orion {
namespace bosen {

struct PartitionNumUniquePartialKeys {
  const int32_t kDistArrayId;
  std::vector<size_t> dim_indices;
  std::map<int32_t, size_t> num_unique_partial_keys;
  std::unordered_map<int32_t, std::vector<int32_t>> executor_id_to_partition_ids;
  std::unordered_map<int32_t, std::vector<int32_t>> server_id_to_partition_ids;
  size_t total_num_unique_partial_keys {0};
  PartitionNumUniquePartialKeys(int32_t dist_array_id,
                                const size_t *_dim_indices,
                                size_t _num_dim_indices):
      kDistArrayId(dist_array_id),
      dim_indices(_num_dim_indices) {
    memcpy(dim_indices.data(), _dim_indices, _num_dim_indices * sizeof(size_t));
  }
};

}
}
