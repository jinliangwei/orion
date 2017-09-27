#include <orion/bosen/dist_array_meta.hpp>
#include <glog/logging.h>

namespace orion {
namespace bosen {
DistArrayMeta::DistArrayMeta(
    size_t num_dims,
    task::DistArrayParentType parent_type,
    task::DistArrayInitType init_type,
    const DistArrayMeta *parent_dist_array_meta,
    bool is_dense):
    kNumDims(num_dims),
    dims_(num_dims, 0),
    kParentType_(parent_type),
    kInitType_(init_type),
    is_dense_(is_dense) {
  switch (kParentType_) {
    case task::TEXT_FILE:
      {
        partition_scheme_ = DistArrayPartitionScheme::kNaive;
      }
      break;
    case task::DIST_ARRAY:
      {
        CHECK(parent_dist_array_meta != nullptr);
        partition_scheme_ = parent_dist_array_meta->partition_scheme_;
      }
      break;
    case task::INIT:
      {
        if (kInitType_ == task::EMPTY) {
          partition_scheme_ = DistArrayPartitionScheme::kHash;
        } else {
          partition_scheme_ = DistArrayPartitionScheme::kRange;
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown parent type " << static_cast<int>(kParentType_);
  }
}

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
  return is_dense_;
}

DistArrayPartitionScheme
DistArrayMeta::GetPartitionScheme() const {
  return partition_scheme_;
}

void
DistArrayMeta::SetPartitionScheme(DistArrayPartitionScheme partition_scheme) {
  partition_scheme_ = partition_scheme;
}

}
}
