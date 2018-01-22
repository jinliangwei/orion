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

 public:
  AbstractDistArrayPartition(DistArray* dist_array,
                             const Config &config,
                             type::PrimitiveType value_type,
                             JuliaThreadRequester *julia_requester);
  virtual ~AbstractDistArrayPartition() { }

  bool LoadTextFile(const std::string &path, int32_t partition_id);
  void ParseText(Blob *max_key, size_t line_num_start);
  size_t CountNumLines() const;
  std::vector<int64_t>& GetDims();
  void Init(int64_t key_begin, size_t num_elements);
  void Map(AbstractDistArrayPartition *child_partition);
  void ComputeKeysFromBuffer(const std::vector<int64_t> &dims);
  const std::string &GetDistArraySymbol();
  size_t GetNumKeyValues() { return keys_.size(); }
  void ComputeHashRepartitionIdsAndRepartition(size_t num_partitions);
  void ComputeRepartitionIdsAndRepartition(
      const std::string &repartition_func_name);
  void ComputePrefetchIndinces(const std::string &prefetch_batch_func_name,
                               const std::vector<int32_t> &dist_array_ids_vec,
                               PointQueryKeyDistArrayMap *point_key_vec_map,
                               RangeQueryKeyDistArrayMap *range_key_vec_map);
  void Execute(const std::string &loop_batch_func_name);

  virtual void CreateAccessor() = 0;
  virtual void ClearAccessor() = 0;
  virtual void CreateCacheAccessor() = 0;
  virtual void CreateBufferAccessor() = 0;
  virtual void ClearCacheOrBufferAccessor() = 0;
  virtual void BuildKeyValueBuffersFromSparseIndex() = 0;
  virtual void BuildIndex() = 0;

  virtual SendDataBuffer Serialize() = 0;
  virtual const uint8_t* Deserialize(const uint8_t *buffer) = 0;
  virtual const uint8_t* DeserializeAndAppend(const uint8_t *buffer) = 0;
  virtual jl_value_t *GetGcPartition() = 0;
  virtual void Clear() = 0;
 protected:
  DISALLOW_COPY(AbstractDistArrayPartition);

  void AppendKeyValue(int64_t key, jl_value_t* value);
  virtual void Repartition(const int32_t *repartition_ids) = 0;

  virtual void GetJuliaValueArray(jl_value_t **value) = 0;

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
};

}
}
