#pragma once

#include <vector>
#include <algorithm>
#include <orion/noncopyable.hpp>
namespace orion {
namespace bosen {

class DistArrayMeta {
 private:
  const size_t kNumDims;
  std::vector<int64_t> dims_;
 public:
  DistArrayMeta(size_t num_dims);
  ~DistArrayMeta() { }
  DISALLOW_COPY(DistArrayMeta);

  void UpdateDimsMax(const std::vector<int64_t> &dims);
  const std::vector<int64_t> &GetDims() const;
};

DistArrayMeta::DistArrayMeta(
    size_t num_dims):
    kNumDims(num_dims),
    dims_(num_dims, 0) { }

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



}
}
