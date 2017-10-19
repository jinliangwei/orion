#pragma once

#include <map>
#include <unordered_map>
#include <orion/noncopyable.hpp>
#include <orion/bosen/type.hpp>
#include <orion/bosen/config.hpp>
#include <orion/bosen/julia_module.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/task.pb.h>
#include <orion/bosen/dist_array_meta.hpp>
#include <orion/bosen/dist_array_access.hpp>

namespace orion {
namespace bosen {
class AbstractDistArrayPartition;
class JuliaEvaluator;

// A DistArray has different access modes depending on
// 1) whether or not a global or local index is built;
// 1) whether it's dense or sparse

// The supported access modes are:
// 1) Seqential: if not index is built
// 2) LocalDenseIndex: if local index is built and dist array partition is dense
// 3) LocalSparseIndex: if local index is built and dist array partition is sparse
// 4) GlobalIndex: if global index is built

// Sequential:
// 1) int64* GetKeys(Partition)
// 2) ValueType* GetValues(Partition)

// LocalDenseIndex:
// 1) ValueType* GetRange()

// LocalSparseIndex:

// GlobalIndex

class DistArray {
 public:
  using TimePartitionMap = std::map<int32_t, AbstractDistArrayPartition*>;
  using SpaceTimePartitionMap = std::map<int32_t, TimePartitionMap>;
  using PartitionMap = std::map<int32_t, AbstractDistArrayPartition*>;
 private:
  const Config &kConfig;
  const type::PrimitiveType kValueType;
  const size_t kValueSize;
  const int32_t kExecutorId;
  SpaceTimePartitionMap space_time_partitions_;
  PartitionMap partitions_;
  std::vector<int64_t> dims_;
  DistArrayMeta meta_;
  //DistArrayPartitionScheme partition_scheme_;
  DistArrayAccess access_;
 public:
  DistArray(const Config& config,
            type::PrimitiveType value_type,
            int32_t executor_id,
            size_t num_dims,
            task::DistArrayParentType parent_type,
            task::DistArrayInitType init_type,
            const DistArrayMeta *parent_dist_array_meta,
            bool is_dense,
            const std::string &symbol);
  ~DistArray();
  //DistArray(DistArray &&other);
  void LoadPartitionsFromTextFile(
      JuliaEvaluator *julia_eval,
      std::string file_path,
      task::DistArrayMapType map_type,
      bool flatten_results,
      size_t num_dims,
      JuliaModule mapper_func_module,
      std::string mapper_func_name,
      Blob *result_buff);
  void SetDims(const std::vector<int64_t> &dims);
  void SetDims(const int64_t* dims, size_t num_dims);
  std::vector<int64_t> &GetDims();
  DistArrayMeta &GetMeta();
  type::PrimitiveType GetValueType();
  PartitionMap &GetLocalPartitionMap();
  AbstractDistArrayPartition *GetLocalPartition(int32_t partition_id);
  SpaceTimePartitionMap &GetSpaceTimePartitionMap();
  void GetAndClearLocalPartitions(std::vector<AbstractDistArrayPartition*>
                                  *buff);

  AbstractDistArrayPartition *CreatePartition();
  void AddPartition(int32_t partition_id,
                    AbstractDistArrayPartition* partition);
  void AddPartition(int32_t space_id, int32_t time_id,
                    AbstractDistArrayPartition* partition);
  void RepartitionSerializeAndClear(
      std::unordered_map<int32_t, std::pair<uint8_t*, size_t>>* send_buff_ptr);
  void RepartitionDeserialize(
      const uint8_t *mem, size_t mem_size);
  void CheckAndBuildIndex();

  void RandomInit(
      JuliaEvaluator *julia_eval,
      task::DistArrayInitType init_type,
      task::DistArrayMapType map_type,
      JuliaModule mapper_func_module,
      std::string mapper_func_name,
      type::PrimitiveType random_init_type);

  void GetMaxPartitionIds(std::vector<int32_t>* ids);

  DistArrayAccess *GetAccessPtr() { return &access_; }
  void SetAccessPartition(int32_t partition_id);
  void SetAccessPartition(AbstractDistArrayPartition *partition);
  AbstractDistArrayPartition* GetAccessPartition();
  void DeletePartition(int32_t partition_id);
 private:
  DISALLOW_COPY(DistArray);
  void RepartitionSerializeAndClearSpaceTime(
      std::unordered_map<int32_t, std::pair<uint8_t*, size_t>>* send_buff_ptr);
  void RepartitionSerializeAndClear1D(
      std::unordered_map<int32_t, std::pair<uint8_t*, size_t>>* send_buff_ptr);

  void RepartitionDeserializeSpaceTime(
      const uint8_t *mem, size_t mem_size);
  void RepartitionDeserialize1D(
      const uint8_t *mem, size_t mem_size);
  void GetMaxPartitionIdsSpaceTime(std::vector<int32_t>* ids);
  void GetMaxPartitionIds1D(std::vector<int32_t>* ids);
  void BuildPartitionIndices();
};

}
}
