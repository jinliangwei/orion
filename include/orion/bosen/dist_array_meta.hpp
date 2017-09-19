#pragma once

#include <vector>
#include <algorithm>
#include <orion/noncopyable.hpp>
#include <orion/bosen/task.pb.h>

namespace orion {
namespace bosen {

enum class DistArrayPartitionScheme {
  kHash = 0,
    kRange = 1
};

class DistArrayMeta {
 private:
  const size_t kNumDims;
  std::vector<int64_t> dims_;
  task::DistArrayInitType kInitType_;
  DistArrayPartitionScheme partition_scheme_;
 public:
  DistArrayMeta(size_t num_dims,
                task::DistArrayInitType init_type);
  ~DistArrayMeta() { }
  DISALLOW_COPY(DistArrayMeta);

  void UpdateDimsMax(const std::vector<int64_t> &dims);
  const std::vector<int64_t> &GetDims() const;
  void AssignDims(const int64_t* dims);
};

DistArrayMeta::DistArrayMeta(
    size_t num_dims,
    task::DistArrayInitType init_type):
    kNumDims(num_dims),
    dims_(num_dims, 0),
    kInitType_(init_type) {
  if (kInitType_ == task::EMPTY) {
    partition_scheme_ = DistArrayPartitionScheme::kHash;
  } else {
    partition_scheme_ = DistArrayPartitionScheme::kRange;
  }
}

void
DistArrayMeta::UpdateDimsMax(
    const std::vector<int64_t> &max_keys) {
  for (int i = 0; i < kNumDims; i++) {
    dims_[i] = std::max(max_keys[i] + 1, dims_[i]);
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

}
}
