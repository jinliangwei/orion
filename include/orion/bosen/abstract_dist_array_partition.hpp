#pragma once

#include <stdint.h>
#include <string>
#include <hdfs.h>
#include <vector>
#include <orion/bosen/config.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/julia_module.hpp>
#include <orion/bosen/julia_evaluator.hpp>
#include <orion/bosen/julia_thread_requester.hpp>
#include <orion/noncopyable.hpp>
#include <orion/bosen/dist_array_meta.hpp>
#include <orion/bosen/send_data_buffer.hpp>
#include <orion/bosen/key_vec_type.hpp>

namespace orion {
namespace bosen {
class DistArray;
class JuliaEvaluator;

enum class DistArrayPartitionStorageType {
  kKeyValueBuffer = 1, // Default; this may include dense index and no index
    kSparseIndex = 2,
    kAccessor = 3 // this only happens when an for loop is executed
                // 1) on workers, this is for dist array, cache and buffer
                // 2) on servers, this is for dist arrays that serve as helper dist arrays (sparse)
                // and dist array buffers (dense)
                // when executing an apply_buffer_function
};

// state transition:
// 1) kKeyValueBuffer => kSparseIndex via CheckAndBuildIndex() and BuildIndex()
// 2) kSparseIndex => kKeyValueBuffer via BuildKeyValueBuffersFromSparseIndex()
// 3) kKeyValueBuffer => kAccessor via Create**Accessor()
// 4) kAccessor => kKeyValueBuffer via Clear**Accessor()

class AbstractDistArrayPartition {
 protected:
  const Config& kConfig;
  const type::PrimitiveType kValueType;

  DistArray *dist_array_;
  JuliaThreadRequester *julia_requester_;
  std::vector<int64_t> keys_;
  std::vector<char> char_buff_;
  std::vector<int64_t> key_buff_;
  int64_t key_start_ { -1 }; // used (set to nonnegative when a dense index is built)
  bool sorted_ { false };
  DistArrayPartitionStorageType storage_type_ { DistArrayPartitionStorageType::kKeyValueBuffer };

 public:
  AbstractDistArrayPartition(DistArray* dist_array,
                             const Config &config,
                             type::PrimitiveType value_type,
                             JuliaThreadRequester *julia_requester);
  virtual ~AbstractDistArrayPartition() { }

  const std::vector<int64_t>& GetDims() const;
  const std::string &GetDistArraySymbol();
  size_t GetLength() const;
  void ComputeKeyDiffs(const std::vector<int64_t> &target_keys,
                       std::vector<int64_t> *diff_keys) const;

  // loading from text files
  bool LoadTextFile(const std::string &path, int32_t partition_id);
  void ParseText(Blob *max_key, size_t line_num_start);
  void ComputeKeysFromBuffer(const std::vector<int64_t> &dims);
  size_t CountNumLines() const;

  // execute computation
  void Init(int64_t key_begin, size_t num_elements);
  void Map(AbstractDistArrayPartition *child_partition);
  void Execute(const std::string &loop_batch_func_name,
               const std::vector<jl_value_t*> &accessed_dist_arrays,
               const std::vector<jl_value_t*> &accessed_dist_array_buffers,
               const std::vector<jl_value_t*> &global_read_only_var_vals,
               const std::vector<std::string> &accumulator_var_syms,
               size_t offset,
               size_t num_elements);
  void ComputePrefetchIndices(const std::string &prefetch_batch_func_name,
                              const std::vector<int32_t> &dist_array_ids_vec,
                              const std::unordered_map<int32_t, DistArray*> &global_indexed_dist_arrays,
                              const std::vector<jl_value_t*> &global_read_only_var_vals,
                              const std::vector<std::string> &accumulator_var_syms,
                              PointQueryKeyDistArrayMap *point_key_vec_map,
                              size_t offset,
                              size_t num_elements);

  // repartition
  void ComputeHashRepartitionIdsAndRepartition(size_t num_partitions);
  void ComputeRepartitionIdsAndRepartition(
      const std::string &repartition_func_name);

  // storage type state transition
  void BuildIndex();
  void CheckAndBuildIndex();
  void ApplyBufferedUpdates(
      AbstractDistArrayPartition* dist_array_buffer,
      const std::vector<AbstractDistArrayPartition*> &helper_dist_arrays,
      const std::vector<AbstractDistArrayPartition*> &helper_dist_array_buffers,
      const std::string &apply_buffer_func_name);

  // storage type state transition
  virtual void CreateAccessor() = 0;
  virtual void ClearAccessor() = 0;
  virtual void CreateCacheAccessor() = 0;
  virtual void CreateBufferAccessor() = 0;
  virtual void ClearCacheAccessor() = 0;
  virtual void ClearBufferAccessor() = 0;
  virtual void BuildKeyValueBuffersFromSparseIndex() = 0;

  virtual void GetAndSerializeValue(int64_t key, Blob *bytes_buff) = 0;
  virtual void GetAndSerializeValues(int64_t *keys, size_t num_keys,
                                     Blob *bytes_buff) = 0;
  virtual SendDataBuffer Serialize() = 0;
  virtual void HashSerialize(ExecutorDataBufferMap *data_buffer_map) = 0;
  virtual uint8_t* Deserialize(uint8_t *buffer) = 0;
  virtual uint8_t* DeserializeAndAppend(uint8_t *buffer) = 0;

  // apply updates
  virtual uint8_t* DeserializeAndOverwrite(uint8_t *buffer) = 0;
  virtual void Clear() = 0;
  virtual void Sort() = 0;
 protected:
  DISALLOW_COPY(AbstractDistArrayPartition);
  void AppendKeyValue(int64_t key, jl_value_t* value);
  std::vector<int64_t>& GetKeys();
  void Repartition(const int32_t *repartition_ids) ;

  virtual void RepartitionSpaceTime(const int32_t *repartition_ids) = 0;
  virtual void Repartition1D(const int32_t *repartition_ids) = 0;

  virtual void GetJuliaValueArray(jl_value_t **value) = 0;
  virtual void GetJuliaValueArray(std::vector<int64_t> &keys, jl_value_t **value) = 0;
  virtual void SetJuliaValues(std::vector<int64_t> &keys, jl_value_t *value) = 0;

  virtual void AppendJuliaValue(jl_value_t *value) = 0;
  virtual void AppendJuliaValueArray(jl_value_t *value) = 0;

  static void GetBufferBeginAndEnd(int32_t partition_id,
                                   size_t partition_size,
                                   size_t read_size,
                                   std::vector<char> *char_buff,
                                   size_t *begin,
                                   size_t *end);

  static bool LoadFromHDFS(const std::string &hdfs_name_node,
                           const std::string &file_path,
                           int32_t partition_id,
                           size_t num_executors,
                           size_t partition_size,
                           std::vector<char> *char_buff);

  static bool LoadFromPosixFS(const std::string &file_path,
                              int32_t partition_id,
                              size_t num_executors,
                              size_t partition_size,
                              std::vector<char> *char_buff);

  virtual void BuildDenseIndex() = 0;
  virtual void BuildSparseIndex() = 0;

  virtual void ShrinkValueVecToFit() = 0;
};

}
}
