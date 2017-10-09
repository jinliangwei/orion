#pragma once

#include <vector>
#include <algorithm>
#include <orion/noncopyable.hpp>
#include <orion/bosen/task.pb.h>

namespace orion {
namespace bosen {

// A DistArray may be parititioned into many AbstractDistArrayPartitions (which are distributed across the cluster)
// in the following ways:
// 1) naively partitioned: each partition corresponds to a fixed-sized block in the input file;
// 2) space-time partitioned (2D partitioned): the dist array is partitioned into two dimensions (space and time),
// a user-defined hash function is used to compute the partition keys
// 3) 1D partitioned: range partitioned based on one dimension of the key
// 4) range partitioned: range partitioned based on the 64-bit full key
// 5) hash partitioned: hash partitioned based on the 64-bit full key

// DistArrays that don't have keys are always naively partitioned.
// All partition schemes allow building local index but only range partitioned and hash partitioned DistArrays
// allow global index.

enum class DistArrayPartitionScheme {
  kNaive = 0,
    kSpaceTime = 1,
    k1D = 2,
    kHash = 3,
    kRange = 4
};

enum class DistArrayIndexType {
  kNone = 0,
    kGlobal = 1,
    kLocal = 2
};

class DistArrayMeta {
 private:
  const size_t kNumDims;
  std::vector<int64_t> dims_;
  task::DistArrayParentType kParentType_;
  task::DistArrayInitType kInitType_;
  DistArrayPartitionScheme partition_scheme_;
  bool is_dense_;
  DistArrayIndexType index_type_;
 public:
  DistArrayMeta(size_t num_dims,
                task::DistArrayParentType parent_type,
                task::DistArrayInitType init_type,
                const DistArrayMeta *parent_dist_array_meta,
                bool is_dense);
  ~DistArrayMeta() { }
  DISALLOW_COPY(DistArrayMeta);

  void UpdateDimsMax(const std::vector<int64_t> &dims);
  const std::vector<int64_t> &GetDims() const;
  void AssignDims(const int64_t* dims);
  bool IsDense() const;
  DistArrayPartitionScheme GetPartitionScheme() const;
  void SetPartitionScheme(DistArrayPartitionScheme partition_scheme);
  void SetIndexType(DistArrayIndexType index_type);
  DistArrayPartitionScheme GetPartitionScheme() { return partition_scheme_; }
};

}
}
