#pragma once

#include <stdint.h>
#include <string>
#include <hdfs.h>
#include <vector>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/julia_module.hpp>
#include <orion/bosen/julia_evaluator.hpp>
#include <orion/noncopyable.hpp>
#include <orion/bosen/task.pb.h>

namespace orion {
namespace bosen {
class DistArray;

class AbstractDistArrayPartition {
 public:
  AbstractDistArrayPartition() { }
  virtual ~AbstractDistArrayPartition() { }

  virtual bool LoadTextFile(
      JuliaEvaluator *julia_eval,
      const std::string &file_path,
      int32_t partition_id,
      task::DistArrayMapType map_type,
      bool flatten_results,
      size_t num_dims,
      JuliaModule mapper_func_module,
      const std::string &mapper_func,
      Blob *result_buff) = 0;
  virtual void SetDims(const std::vector<int64_t> &dims) = 0;
  virtual std::vector<int64_t> &GetDims() = 0;
  virtual type::PrimitiveType GetValueType() = 0;

  virtual void Insert(int64_t key, const Blob &buff) = 0;
  virtual void Get(int64_t key, Blob *buff) = 0;
  virtual void GetRange(int64_t start, int64_t end, Blob *buff) = 0;
  virtual std::vector<int64_t>& GetKeys() = 0;
  virtual void* GetValues() = 0;
  virtual void AppendKeyValue(int64_t key, const void* value) = 0;
  virtual void Repartition(const int32_t *repartition_ids) = 0;
  virtual size_t GetNumKeyValues() = 0;
  virtual size_t GetValueSize() = 0;
  virtual void CopyValues(void *mem) const = 0;
  virtual void RandomInit(
      JuliaEvaluator *julia_eval,
      const std::vector<int64_t> &dims,
      int64_t key_begin,
      size_t num_elements,
      task::DistArrayInitType init_type,
      task::DistArrayMapType map_type,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name,
      type::PrimitiveType random_init_type) = 0;

  virtual void ReadRange(
      int64_t key_begin,
      size_t num_elements,
      void *mem) = 0;

  virtual void WriteRange(
      int64_t key_begin,
      size_t num_elements,
      void *mem) = 0;

 protected:
  DISALLOW_COPY(AbstractDistArrayPartition);

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
                           std::vector<char> *char_buff, size_t *begin,
                           size_t *end);

  static bool LoadFromPosixFS(const std::string &file_path,
                              int32_t partition_id,
                              size_t num_executors,
                              size_t partition_size,
                              std::vector<char> *char_buff, size_t *begin,
                              size_t *end);
};

}
}
