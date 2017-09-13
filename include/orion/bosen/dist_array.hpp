#pragma once

#include <map>
#include <unordered_map>
#include <orion/noncopyable.hpp>
#include <orion/bosen/type.hpp>
#include <orion/bosen/config.hpp>
#include <orion/bosen/julia_module.hpp>
#include <orion/bosen/blob.hpp>

namespace orion {
namespace bosen {
class AbstractDistArrayPartition;
class JuliaEvaluator;

class DistArray {
 public:
  using SpacePartition = std::map<int32_t, AbstractDistArrayPartition*>;
 private:
  const Config &kConfig;
  const type::PrimitiveType kValueType;
  const size_t kValueSize;
  const int32_t kExecutorId;
  std::unordered_map<int32_t, AbstractDistArrayPartition*> partitions_;
  std::unordered_map<int32_t, SpacePartition> space_time_partitions_;
  std::vector<int64_t> dims_;
 public:
  DistArray(const Config& config,
            type::PrimitiveType value_type, int32_t executor_id);
  ~DistArray();
  DistArray(DistArray &&other);
  void LoadPartitionsFromTextFile(
      JuliaEvaluator *julia_eval,
      const std::string &file_path,
      bool map,
      bool flatten_results,
      size_t num_dims,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name,
      Blob *result_buff);
  void SetDims(const std::vector<int64_t> &dims);
  std::vector<int64_t> &GetDims();
  std::unordered_map<int32_t, AbstractDistArrayPartition*>&
  GetLocalPartitions();
  AbstractDistArrayPartition *GetLocalPartition(int32_t partition_id);
  std::unordered_map<int32_t, SpacePartition> &GetSpaceTimePartitions();

  AbstractDistArrayPartition *CreatePartition() const;
  void AddSpaceTimePartition(int32_t space_id, int32_t time_id,
                             AbstractDistArrayPartition* partition);
  void SerializeAndClearSpaceTimePartitions(
      std::unordered_map<int32_t, Blob> *send_buff_ptr);
  void DeserializeSpaceTimePartitions(
      const uint8_t *mem, size_t mem_size);
 private:
  DISALLOW_COPY(DistArray);
};

}
}
