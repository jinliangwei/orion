#include <orion/bosen/dist_array_partition.hpp>
namespace orion {
namespace bosen {
/*---- template const char* implementation -----*/
DistArrayPartition<const char*>::DistArrayPartition(
    DistArray *dist_array,
    const Config &config,
    type::PrimitiveType value_type):
    dist_array_(dist_array),
    kConfig(config),
    kValueType(value_type),
    key_start_(-1) { }

DistArrayPartition<const char*>::~DistArrayPartition() { }

bool
DistArrayPartition<const char*>::LoadTextFile(
    JuliaEvaluator *julia_eval,
    const std::string &path, int32_t partition_id,
    task::DistArrayMapType map_type,
    bool flatten_results,
    size_t num_dims,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name,
    Blob *max_key) {
  LOG(INFO) << __func__;
  size_t offset = path.find_first_of(':');
  std::string prefix = path.substr(0, offset);
  std::string file_path = path.substr(offset + 3, path.length() - offset - 3);
  std::vector<char> char_buff;
  size_t begin = 0, end = 0;
  bool read = false;
  if (prefix == "hdfs") {
    read = LoadFromHDFS(kConfig.kHdfsNameNode, file_path, partition_id,
                        kConfig.kNumExecutors,
                        kConfig.kPartitionSizeMB * 1024 * 1024,
                        &char_buff, &begin, &end);
  } else if (prefix == "file") {
    read = LoadFromPosixFS(file_path, partition_id,
                           kConfig.kNumExecutors,
                           kConfig.kPartitionSizeMB * 1024 * 1024,
                           &char_buff, &begin, &end);
  } else {
    LOG(FATAL) << "Cannot parse the path specification " << path;
  }
  return read;
}

std::vector<int64_t> &
DistArrayPartition<const char*>::GetDims() {
  return dist_array_->GetDims();
}

type::PrimitiveType
DistArrayPartition<const char*>::GetValueType() {
  return dist_array_->GetValueType();
}

void
DistArrayPartition<const char*>::SetDims(const std::vector<int64_t> &dims) {
}

void
DistArrayPartition<const char*>::AppendKeyValue(int64_t key, const void* value) {

}

void
DistArrayPartition<const char*>::AddToSpaceTimePartitions(
    DistArray *dist_array,
    const std::vector<int32_t> &partition_ids) {

}

size_t
DistArrayPartition<const char*>::GetNumKeyValues() {
  return keys_.size();
}

size_t
DistArrayPartition<const char*>::GetValueSize() {
  return 0;
}

void
DistArrayPartition<const char*>::CopyValues(void *mem) const {
}

void
DistArrayPartition<const char*>::RandomInit(
    JuliaEvaluator* julia_eval,
    const std::vector<int64_t> &dims,
    int64_t key_begin,
    size_t num_elements,
    task::DistArrayInitType init_type,
    task::DistArrayMapType map_type,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name,
    type::PrimitiveType random_init_type) {
  LOG(INFO) << "random init is not supported by element type const char*";
}

void
DistArrayPartition<const char*>::ReadRange(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {

}

void
DistArrayPartition<const char*>::ReadRangeDense(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {

}

void
DistArrayPartition<const char*>::ReadRangeSparse(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {

}

void
DistArrayPartition<const char*>::WriteRange(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense();
  auto partition_scheme = dist_array_meta.GetPartitionScheme();

  if (is_dense
      && (partition_scheme == DistArrayPartitionScheme::k1D
          || partition_scheme == DistArrayPartitionScheme::kRange)) {
    WriteRangeDense(key_begin, num_elements, mem);
  } else {
    WriteRangeSparse(key_begin, num_elements, mem);
  }
}

void
DistArrayPartition<const char*>::WriteRangeDense(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {
}

void
DistArrayPartition<const char*>::WriteRangeSparse(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {

}

}
}
