#pragma once

#include <stdint.h>
#include <vector>
#include <stx/btree_map>
#include <stdio.h>
#include <orion/bosen/julia_evaluator.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/config.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>

#ifdef ORION_USE_HDFS
#include <hdfs.h>
#endif

namespace orion {
namespace bosen {

template<typename ValueType>
class DistArrayPartition : public AbstractDistArrayPartition {
 private:
  std::vector<uint64_t> keys_;
  std::vector<ValueType> values_;
  stx::btree_map<uint64_t, ValueType> index_;
  bool index_exists_ {false};
  JuliaEvaluator *julia_eval_;
  const Config* kConfig;

 public:
  DistArrayPartition(JuliaEvaluator *julia_eval,
                     const Config &config);
  ~DistArrayPartition();

  bool LoadTextFile(const std::string &path, int32_t partition_id,
                    const std::string &parser_func);

  void Insert(uint64_t key, const Blob &buff) { }
  void Get(uint64_t key, Blob *buff) { }
  void GetRange(uint64_t start, uint64_t end, Blob *buff) { }
 private:
  void LoadFromPosixFS(const std::string &path, int32_t partition_id,
                       const std::string &parser_func);
  void LoadFromHDFS(const std::string &path, int32_t partition_id,
                    const std::string &parser_func);
};

template<typename ValueType>
DistArrayPartition<ValueType>::DistArrayPartition(
    JuliaEvaluator *julia_eval,
    const Config &config):
    julia_eval_(julia_eval),
    kConfig(&config) { }

template<typename ValueType>
DistArrayPartition<ValueType>::~DistArrayPartition() { }


template<typename ValueType>
bool
DistArrayPartition<ValueType>::LoadTextFile(
    const std::string &path, int32_t partition_id,
    const std::string &parser_func) {
  size_t offset = path.find_first_of(':');
  std::string prefix = path.substr(0, offset);
  std::string file_path = path.substr(offset + 3, path.length() - offset - 3);
  if (prefix == "hdfs") {
    LoadFromHDFS(file_path, partition_id, parser_func);
  } else if (prefix == "file") {
    LoadFromPosixFS(file_path, partition_id, parser_func);
  } else {
    LOG(FATAL) << "Cannot parse the path specification " << path;
  }
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::LoadFromHDFS(
    const std::string &path, int32_t partition_id,
    const std::string &parser_func) {
#ifdef ORION_USE_HDFS
#else
  LOG(FATAL) << "HDFS is not supported in this build";
#endif
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::LoadFromPosixFS(
    const std::string &path, int32_t partition_id,
    const std::string &parser_func) {
}

}
}
