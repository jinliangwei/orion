#include <orion/bosen/dist_array_meta.hpp>
#include <glog/logging.h>

namespace orion {
namespace bosen {
DistArrayMeta::DistArrayMeta(
    size_t num_dims,
    DistArrayParentType parent_type,
    DistArrayInitType init_type,
    DistArrayMapType map_type,
    DistArrayPartitionScheme partition_scheme,
    JuliaModule map_func_module,
    const std::string &map_func_name,
    type::PrimitiveType random_init_type,
    bool flatten_results,
    bool is_dense,
    const std::string &symbol,
    const std::string &key_func_name):
    kNumDims(num_dims),
    dims_(num_dims, 0),
    kParentType(parent_type),
    kInitType(init_type),
    kMapType(map_type),
    kMapFuncModule(map_func_module),
    kMapFuncName(map_func_name),
    kFlattenResults(flatten_results),
    kIsDense(is_dense),
    kRandomInitType(random_init_type),
    partition_scheme_(partition_scheme),
    index_type_(DistArrayIndexType::kNone),
    symbol_(symbol),
    kKeyFuncName(key_func_name) { }

void
DistArrayMeta::UpdateDimsMax(
    const std::vector<int64_t> &max_keys) {
  for (int i = 0; i < kNumDims; i++) {
    dims_[i] = std::max(max_keys[i], dims_[i]);
  }
}

const std::vector<int64_t> &
DistArrayMeta::GetDims() const {
  return dims_;
}

void
DistArrayMeta::AssignDims(
    const int64_t* dims) {
  memcpy(dims_.data(), dims, dims_.size() * sizeof(int64_t));
}

bool
DistArrayMeta::IsDense() const {
  return kIsDense;
}

void
DistArrayMeta::SetPartitionScheme(DistArrayPartitionScheme partition_scheme) {
  partition_scheme_ = partition_scheme;
}

void
DistArrayMeta::SetIndexType(DistArrayIndexType index_type) {
  index_type_ = index_type;
}

void
DistArrayMeta::SetMaxPartitionIds(int32_t space_id, int32_t time_id) {
  LOG(INFO) << __func__;
  max_partition_ids_.resize(2);
  max_partition_ids_[0] = space_id;
  max_partition_ids_[1] = time_id;
}

void
DistArrayMeta::SetMaxPartitionIds(int32_t partition_id) {
  LOG(INFO) << __func__;
  max_partition_ids_.resize(1);
  max_partition_ids_[0] = partition_id;
}

void
DistArrayMeta::SetMaxPartitionIds(const int32_t* max_ids, size_t num_dims) {
  LOG(INFO) << __func__ << " num_dims = " << num_dims;
  max_partition_ids_.resize(num_dims);
  for (size_t i = 0; i < num_dims; i++) {
    LOG(INFO) << "i = " << i << " id = " << max_ids[i];
    max_partition_ids_[i] = max_ids[i];
  }
}

void
DistArrayMeta::ResetMaxPartitionIds() {
  max_partition_ids_.clear();
}

void
DistArrayMeta::AccumMaxPartitionIds(const int32_t *max_ids, size_t num_dims) {
  if (num_dims != max_partition_ids_.size()) {
    max_partition_ids_.resize(num_dims);
  }
  for (size_t i = 0; i < num_dims; i++) {
    max_partition_ids_[i] = std::max(max_partition_ids_[i], max_ids[i]);
  }
}

void
DistArrayMeta::SetInitValue(const uint8_t *init_value_bytes, size_t num_bytes) {
  init_value_bytes_.resize(num_bytes);
  memcpy(init_value_bytes_.data(), init_value_bytes, num_bytes);
}

const std::vector<uint8_t>&
DistArrayMeta::GetInitValue() const {
  return init_value_bytes_;
}

void
DistArrayMeta::SetContiguousPartitions(bool is_contiguous) {
  contiguous_partitions_ = is_contiguous;
}

bool
DistArrayMeta::IsContiguousPartitions() const {
  return contiguous_partitions_;
}

}
}
