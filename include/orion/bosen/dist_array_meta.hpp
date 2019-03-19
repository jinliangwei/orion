#pragma once

#include <vector>
#include <algorithm>
#include <orion/noncopyable.hpp>
#include <orion/bosen/julia_module.hpp>
#include <orion/bosen/type.hpp>

namespace orion {
namespace bosen {

// A DistArray may be parititioned into many AbstractDistArrayPartitions (which are distributed across the cluster)
// in the following ways:
// 1) naively partitioned: each partition corresponds to a fixed-sized block in the input file;
// 2) space-time partitioned (2D partitioned): the dist array is partitioned into two dimensions (space and time),
// a user-defined hash function is used to compute the partition keys
// 3) 1D partitioned: range partitioned based on one dimension of the key
// 4) range partitioned: range partitioned based on the 64-bit full key
// 5) modulo partitioned: modulo partitioned based on the 64-bit full key

// DistArrays that don't have keys are always naively partitioned.
// All partition schemes allow building local index but only range partitioned and hash partitioned DistArrays
// allow global index.

enum class DistArrayPartitionScheme {
  kNaive = 0,
    kSpaceTime = 1,
    k1D = 2,
    kModuloServer = 3,
    kModuloExecutor = 4,
    kRange = 5,
    kPartialModuloExecutor = 6,
    kPartialRandomExecutor = 7,
    kHashExecutor = 8,
    k1DOrdered = 9
};

enum class DistArrayIndexType {
  kNone = 0,
    kRange = 1,
};

enum class ForLoopParallelScheme {
  k1D = 0,
    kSpaceTime = 1
};

enum class DistArrayParentType {
  kTextFile = 0,
    kDistArray = 1,
    kInit = 2
};

enum class DistArrayInitType {
  kEmpty = 0,
    kUniformRandom = 1,
    kNormalRandom = 2,
    kFill = 3
};

enum class DistArrayMapType {
  kNoMap = 0,
    kMap = 1,
    kMapFixedKeys = 2,
    kMapValues = 3,
    kMapValuesNewKeys = 4,
    kGroupBy = 5
};

class DistArrayMeta {
 private:
  const size_t kNumDims;
  std::vector<int64_t> dims_;
  const DistArrayParentType kParentType;
  const DistArrayInitType kInitType;
  const DistArrayMapType kMapType;
  const JuliaModule kMapFuncModule { JuliaModule::kNone };
  const std::string kMapFuncName;
  const bool kFlattenResults;
  const bool kIsDense;
  const type::PrimitiveType kRandomInitType;
  DistArrayPartitionScheme partition_scheme_;
  DistArrayIndexType index_type_;
  std::vector<int32_t> max_partition_ids_;
  std::string symbol_;
  std::vector<uint8_t> init_value_bytes_;
  bool contiguous_partitions_ { false };
  const std::string kKeyFuncName;
 public:
  DistArrayMeta(size_t num_dims,
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
                const std::string &key_func_name);
  ~DistArrayMeta() { }
  DISALLOW_COPY(DistArrayMeta);

  void UpdateDimsMax(const std::vector<int64_t> &dims);
  const std::vector<int64_t> &GetDims() const;
  void AssignDims(const int64_t* dims);
  bool IsDense() const;
  size_t GetNumDims() const { return kNumDims; }
  DistArrayPartitionScheme GetPartitionScheme() const { return partition_scheme_; }
  void SetPartitionScheme(DistArrayPartitionScheme partition_scheme);
  void SetIndexType(DistArrayIndexType index_type);
  DistArrayIndexType GetIndexType() { return index_type_; }
  void SetMaxPartitionIds(int32_t space_id, int32_t time_id);
  void SetMaxPartitionIds(int32_t partition_id);
  void SetMaxPartitionIds(const int32_t* max_ids, size_t num_dims);
  void SetMaxPartitionIds(const std::vector<int32_t> &max_partition_ids) { max_partition_ids_ = max_partition_ids; }

  void ResetMaxPartitionIds();
  void AccumMaxPartitionIds(const int32_t *max_ids, size_t num_dims);
  const std::vector<int32_t> &GetMaxPartitionIds() { return max_partition_ids_; }
  const std::string &GetSymbol() const { return symbol_; }
  bool IsFlattenResults() const { return kFlattenResults; }

  DistArrayMapType GetMapType() const { return kMapType; }
  JuliaModule GetMapFuncModule() const { return kMapFuncModule; }
  const std::string &GetMapFuncName() const { return kMapFuncName; }

  void SetInitValue(const uint8_t *init_value_bytes, size_t num_bytes);
  const std::vector<uint8_t>& GetInitValue() const;
  type::PrimitiveType GetRandomInitType() const { return kRandomInitType; }
  DistArrayInitType GetInitType() const { return kInitType; }
  void SetContiguousPartitions(bool is_contiguous);
  bool IsContiguousPartitions() const;

  const std::string &GetKeyFuncName() const { return kKeyFuncName; }
};

}
}
