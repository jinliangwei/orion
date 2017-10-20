#pragma once

#include <stdint.h>
#include <vector>
#include <map>
#include <stx/btree_map>
#include <stdio.h>
#include <algorithm>
#include <glog/logging.h>
#include <orion/bosen/julia_evaluator.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/config.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/key.hpp>

#ifdef ORION_USE_HDFS
#include <hdfs.h>
#endif

namespace orion {
namespace bosen {
namespace {
template<typename ValueType>
void RandomInitAndRunMap(
    JuliaEvaluator* julia_eval,
    const std::vector<int64_t> &dims,
    task::DistArrayInitType init_type,
    task::DistArrayMapType map_type,
    type::PrimitiveType random_init_type,
    size_t num_elements,
    int64_t *keys,
    std::vector<int64_t> *output_keys,
    type::PrimitiveType output_value_type,
    Blob *output_values,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name) {
  LOG(INFO) << __func__
            << " random_init_type = " << static_cast<int>(random_init_type)
            << " outpu_value_type = " << static_cast<int>(output_value_type);
  std::vector<ValueType> init_values(num_elements);
  switch (init_type) {
    case task::NORMAL_RANDOM:
      {
        julia_eval->RandNormal(random_init_type,
                               reinterpret_cast<uint8_t*>(init_values.data()),
                               num_elements);
      }
      break;
    default:
      LOG(FATAL) << "not yet supported " << static_cast<int>(init_type);
  }
  CHECK(map_type != task::NO_MAP);

  julia_eval->RunMapGeneric(
      map_type,
      dims,
      num_elements,
      keys,
      random_init_type,
      reinterpret_cast<uint8_t*>(init_values.data()),
      output_keys,
      output_value_type,
      output_values,
      mapper_func_module,
      mapper_func_name);
}
}

template<typename ValueType>
class DistArrayPartition : public AbstractDistArrayPartition {
 private:
  DistArray *dist_array_;
  std::vector<int64_t> keys_;
  std::vector<ValueType> values_;
  stx::btree_map<int64_t, ValueType> index_;
  bool index_exists_ {false};
  const Config& kConfig;
  const type::PrimitiveType kValueType;
  // temporary to facilitate LoadTextFile
  std::vector<int64_t> key_buff_;
  int64_t key_start_; // used (set to nonnegative when a dense index is built)
 public:
  using KeyValueBuffer = std::pair<std::vector<int64_t>,
                                   std::vector<ValueType>>;

  DistArrayPartition(DistArray *dist_array,
                     const Config &config,
                     type::PrimitiveType value_type);
  ~DistArrayPartition();

  bool LoadTextFile(JuliaEvaluator *julia_eval,
                    const std::string &file_path, int32_t partition_id,
                    task::DistArrayMapType map_type,
                    bool flatten_results,
                    size_t num_dims,
                    JuliaModule mapper_func_module,
                    const std::string &mapper_func_name,
                    Blob *max_key);

  void ComputeKeysFromBuffer(const std::vector<int64_t> &dims);
  std::vector<int64_t> &GetDims();
  type::PrimitiveType GetValueType();

  void Insert(int64_t key, const Blob &buff) { }
  void Get(int64_t key, Blob *buff) { }
  void GetRange(int64_t start, int64_t end, Blob *buff) { }
  std::vector<int64_t>& GetKeys() { return keys_; }
  void *GetValues() { return &values_; }
  void AppendKeyValue(int64_t key, const void* value);
  void Repartition(const int32_t *repartition_ids);

  size_t GetNumKeyValues();
  size_t GetValueSize();
  void CopyValues(void *mem) const;
  void RandomInit(
      JuliaEvaluator* julia_eval,
      const std::vector<int64_t> &dims,
      int64_t key_begin,
      size_t num_elements,
      task::DistArrayInitType init_type,
      task::DistArrayMapType map_type,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name,
      type::PrimitiveType random_init_type);

  void ReadRange(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void ReadRangeDense(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void ReadRangeSparse(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void WriteRange(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void WriteRangeDense(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void WriteRangeSparse(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void BuildIndex();

  std::pair<uint8_t*, size_t> Serialize();
  void Deserialize(const uint8_t *buffer, size_t num_bytes);

 private:
  void RepartitionSpaceTime(
      const int32_t *repartition_ids);

  void Repartition1D(
      const int32_t *repartition_ids);

  void BuildDenseIndex();
  void BuildSparseIndex();
};

/*----- Specialized for String (const char*) ------*/
template<>
class DistArrayPartition<const char*> : public AbstractDistArrayPartition {
 private:
  DistArray *dist_array_;
  std::vector<int64_t> keys_;
  std::vector<char> values_;
  std::vector<size_t> str_offsets_;
  stx::btree_map<int64_t, const char*> index_;
  bool index_exists_ {false};
  const Config& kConfig;
  const type::PrimitiveType kValueType;
  int64_t key_start_; // used (set to nonnegative when a dense index is built)
 public:
  DistArrayPartition(DistArray *dist_array_,
                     const Config &config,
                     type::PrimitiveType value_type);
  ~DistArrayPartition();

  bool LoadTextFile(JuliaEvaluator *julia_eval,
                    const std::string &file_path, int32_t partition_id,
                    task::DistArrayMapType map_type,
                    bool flatten_results,
                    size_t num_dims,
                    JuliaModule mapper_func_module,
                    const std::string &mapper_func_name,
                    Blob *max_key);
  void ComputeKeysFromBuffer(const std::vector<int64_t> &dims);
  std::vector<int64_t> &GetDims();
  type::PrimitiveType GetValueType();
  void Insert(int64_t key, const Blob &buff) { }
  void Get(int64_t key, Blob *buff) { }
  void GetRange(int64_t start, int64_t end, Blob *buff) { }
  std::vector<int64_t>& GetKeys() { return keys_; }
  void *GetValues() { return &values_; }
  void AppendKeyValue(int64_t key, const void* value);
  void Repartition(
      const int32_t *repartition_ids);
  size_t GetNumKeyValues();
  size_t GetValueSize();
  void CopyValues(void *mem) const;
  void RandomInit(
      JuliaEvaluator* julia_eval,
      const std::vector<int64_t> &dims,
      int64_t key_begin,
      size_t num_elements,
      task::DistArrayInitType init_type,
      task::DistArrayMapType map_type,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name,
      type::PrimitiveType random_init_type);

  void ReadRange(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void ReadRangeDense(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void ReadRangeSparse(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void WriteRange(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void WriteRangeDense(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void WriteRangeSparse(
      int64_t key_begin,
      size_t num_elements,
      void *mem);

  void BuildIndex();

  std::pair<uint8_t*, size_t> Serialize();
  void Deserialize(const uint8_t *buffer, size_t num_bytes);
 private:
  void RepartitionSpaceTime(
      const int32_t *repartition_ids);

  void Repartition1D(
      const int32_t *repartition_ids);
};

/*---- template general implementation -----*/
template<typename ValueType>
DistArrayPartition<ValueType>::DistArrayPartition(
    DistArray *dist_array,
    const Config &config,
    type::PrimitiveType value_type):
    dist_array_(dist_array),
    kConfig(config),
    kValueType(value_type),
    key_start_(-1) { }

template<typename ValueType>
DistArrayPartition<ValueType>::~DistArrayPartition() { }

template<typename ValueType>
bool
DistArrayPartition<ValueType>::LoadTextFile(
    JuliaEvaluator *julia_eval,
    const std::string &path, int32_t partition_id,
    task::DistArrayMapType map_type,
    bool flatten_results,
    size_t num_dims,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name,
    Blob *max_key) {
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
  if (!read) return read;
  if (map_type == task::MAP_VALUES_NEW_KEYS) {
    Blob value(type::SizeOf(kValueType));
    auto* parser_func = julia_eval->GetFunction(GetJlModule(mapper_func_module),
                                                mapper_func_name.c_str());
    if (num_dims > 0) {
      key_buff_.clear();
      std::vector<int64_t> key(num_dims);
      char *line = strtok(char_buff.data() + begin, "\n");
      while (line != nullptr) {
        julia_eval->ParseString(line, parser_func, kValueType,
                                &key, &value);
        line = strtok(nullptr, "\n");
        for (auto key_ith : key) {
          key_buff_.push_back(key_ith);
        }
        values_.push_back(*((ValueType*) value.data()));
      }

      if (max_key != nullptr) {
        std::vector<int64_t> max_key_vec(num_dims, 0);
        memcpy(max_key_vec.data(), max_key->data(), sizeof(int64_t) * num_dims);
        for (int i = 0; i < num_dims; i++) {
          for (int j = i; j < key_buff_.size(); j += num_dims) {
            max_key_vec[i] = std::max(max_key_vec[i], key_buff_[j]);
          }
        }
        memcpy(max_key->data(), max_key_vec.data(), sizeof(int64_t) * num_dims);
      }
    } else {
      CHECK(map_type == task::MAP_VALUES);
      char *line = strtok(char_buff.data() + begin, "\n");
      while (line != nullptr) {
        julia_eval->ParseStringValueOnly(line, parser_func, kValueType,
                                         &value);
        line = strtok(nullptr, "\n");
        values_.push_back(*((ValueType*) value.data()));
      }
    }
  }
  return read;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ComputeKeysFromBuffer(const std::vector<int64_t> &dims) {
  size_t num_dims = dims.size();
  CHECK_EQ(key_buff_.size() / num_dims, values_.size());
  keys_.clear();
  for (int i = 0; i < key_buff_.size(); i += num_dims) {
    int64_t key = key::array_to_int64(dims, key_buff_.data() + i);
    keys_.push_back(key);
  }
  return;
}

template<typename ValueType>
std::vector<int64_t> &
DistArrayPartition<ValueType>::GetDims() {
  return dist_array_->GetDims();
}

template<typename ValueType>
type::PrimitiveType
DistArrayPartition<ValueType>::GetValueType() {
  return dist_array_->GetValueType();
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::AppendKeyValue(int64_t key, const void* value) {
  keys_.emplace_back(key);
  values_.emplace_back(*reinterpret_cast<const ValueType*>(value));
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::Repartition(
    const int32_t *repartition_ids) {
  auto &dist_array_meta = dist_array_->GetMeta();
  auto partition_scheme = dist_array_meta.GetPartitionScheme();
  if (partition_scheme == DistArrayPartitionScheme::kSpaceTime) {
    RepartitionSpaceTime(repartition_ids);
  } else {
    Repartition1D(repartition_ids);
  }
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::RepartitionSpaceTime(
    const int32_t *repartition_ids) {
  auto& space_time_partitions = dist_array_->GetSpaceTimePartitionMap();
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    ValueType value = values_[i];
    int32_t space_partition_id = repartition_ids[i * 2];
    int32_t time_partition_id = repartition_ids[i * 2 + 1];
    auto &time_partitions = space_time_partitions[space_partition_id];
    auto dist_array_partition_iter
        = time_partitions.find(time_partition_id);
    if (dist_array_partition_iter == time_partitions.end()) {
      //LOG(INFO) << "created partition, space = " << space_partition_id
      //          << " time = " << time_partition_id;
      auto* partition = dist_array_->CreatePartition();
      auto ret = time_partitions.emplace(
          std::make_pair(time_partition_id,
                         partition));
      CHECK(ret.second);
      dist_array_partition_iter = ret.first;
    }
    auto* partition = dist_array_partition_iter->second;
    partition->AppendKeyValue(key, &value);
  }
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::Repartition1D(
    const int32_t *repartition_ids) {
  auto& partitions = dist_array_->GetLocalPartitionMap();
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    ValueType value = values_[i];
    int32_t repartition_id = repartition_ids[i];

    auto dist_array_partition_iter
        = partitions.find(repartition_id);
    if (dist_array_partition_iter == partitions.end()) {
      auto* partition = dist_array_->CreatePartition();
      auto ret = partitions.emplace(
          std::make_pair(repartition_id,
                         partition));
      CHECK(ret.second);
      dist_array_partition_iter = ret.first;
    }
    auto* partition = dist_array_partition_iter->second;
    partition->AppendKeyValue(key, &value);
  }
}

template<typename ValueType>
size_t DistArrayPartition<ValueType>::GetNumKeyValues() {
  return keys_.size();
}

template<typename ValueType>
size_t DistArrayPartition<ValueType>::GetValueSize() {
  return type::SizeOf(kValueType);
}

template<typename ValueType>
void DistArrayPartition<ValueType>::CopyValues(void *mem) const {
  memcpy(mem, values_.data(), values_.size() * type::SizeOf(kValueType));
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::RandomInit(
    JuliaEvaluator* julia_eval,
    const std::vector<int64_t> &dims,
    int64_t key_begin,
    size_t num_elements,
    task::DistArrayInitType init_type,
    task::DistArrayMapType map_type,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name,
    type::PrimitiveType random_init_type) {
  keys_.resize(num_elements);
  for (size_t i = 0; i < num_elements; i++) {
    keys_[i] = key_begin + i;
  }
  values_.resize(num_elements);

  std::vector<int64_t> output_keys;
  if (map_type == task::MAP || map_type == task::MAP_VALUES_NEW_KEYS) {
    output_keys.resize(num_elements);
  }

  if (map_type == task::NO_MAP) {
    CHECK(kValueType == type::PrimitiveType::kFloat32 || kValueType == type::PrimitiveType::kFloat64);
    switch (init_type) {
      case task::NORMAL_RANDOM:
        {
          julia_eval->RandNormal(kValueType,
                                 reinterpret_cast<uint8_t*>(values_.data()),
                                 num_elements);
        }
        break;
      default:
        LOG(FATAL) << "not yet supported " << static_cast<int>(init_type);
    }
  } else {
    LOG(INFO) << "random_init_type = " << static_cast<int>(random_init_type);
    Blob output_values;
    if (random_init_type == type::PrimitiveType::kFloat64) {
      RandomInitAndRunMap<double>(
          julia_eval,
          dims,
          init_type,
          map_type,
          random_init_type,
          num_elements,
          keys_.data(),
          &output_keys,
          kValueType,
          &output_values,
          mapper_func_module,
          mapper_func_name);
    } else {
      CHECK(random_init_type == type::PrimitiveType::kFloat32) << "random_init_type = " << static_cast<int>(random_init_type);
      RandomInitAndRunMap<float>(
          julia_eval,
          dims,
          init_type,
          map_type,
          random_init_type,
          num_elements,
          keys_.data(),
          &output_keys,
          kValueType,
          &output_values,
          mapper_func_module,
          mapper_func_name);
    }
    size_t num_elements_after_map = output_keys.size();
    values_.resize(num_elements_after_map);
    memcpy(values_.data(), output_values.data(), output_values.size());
  }
  if (map_type == task::MAP || map_type == task::MAP_VALUES_NEW_KEYS) {
    keys_.resize(output_keys.size());
    memcpy(keys_.data(), output_keys.data(), output_keys.size() * sizeof(int64_t));
  }
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ReadRange(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense();

  if (is_dense) {
    ReadRangeDense(key_begin, num_elements, mem);
  } else {
    ReadRangeSparse(key_begin, num_elements, mem);
  }
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ReadRangeDense(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {
  //CHECK_GE(key_start_, 0) << " need to build dense index first";
  //CHECK(key_begin >= key_start_ && num_elements < keys_.size());
  size_t offset = key_begin - key_start_;
  memcpy(mem, values_.data() + offset, num_elements * type::SizeOf(kValueType));
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ReadRangeSparse(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {

}

template<typename ValueType>
void
DistArrayPartition<ValueType>::WriteRange(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense();
  if (is_dense) {
    WriteRangeDense(key_begin, num_elements, mem);
  } else {
    WriteRangeSparse(key_begin, num_elements, mem);
  }
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::WriteRangeDense(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {
  //CHECK_GE(key_start_, 0) << " need to build dense index first";
  //CHECK(key_begin >= key_start_ && num_elements < keys_.size());
  size_t offset = key_begin - key_start_;
  memcpy(values_.data() + offset, mem, values_.size() * type::SizeOf(kValueType));
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::WriteRangeSparse(
    int64_t key_begin,
    size_t num_elements,
    void *mem) {

}

template<typename ValueType>
void
DistArrayPartition<ValueType>::BuildIndex() {
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense();
  if (is_dense) {
    BuildDenseIndex();
  } else {
    BuildSparseIndex();
  }
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::BuildDenseIndex() {
  if (keys_.size() == 0) return;
  int64_t min_key = keys_[0];
  CHECK(values_.size() == keys_.size());
  for (auto key : keys_) {
    min_key = std::min(key, min_key);
  }
  key_start_ = min_key;
  std::vector<size_t> perm(keys_.size());
  std::vector<ValueType> values_temp(values_);

  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&] (const size_t &i, const size_t &j) {
              return keys_[i] < keys_[j];
            });
  std::transform(perm.begin(), perm.end(), values_.begin(),
                 [&](size_t i) { return values_temp[i]; });

  for (size_t i = 0; i < keys_.size(); i++) {
    keys_[i] = min_key + i;
  }
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::BuildSparseIndex() {
  LOG(FATAL) << "unsupported yet!";
}

template<typename ValueType>
std::pair<uint8_t*, size_t>
DistArrayPartition<ValueType>::Serialize() {
  size_t num_bytes = sizeof(size_t)
                     + keys_.size() * (sizeof(int64_t) + sizeof(ValueType));
  uint8_t* buff = new uint8_t[num_bytes];
  uint8_t* cursor = buff;
  *(reinterpret_cast<size_t*>(cursor)) = keys_.size();
  cursor += sizeof(size_t);
  memcpy(cursor, keys_.data(), keys_.size() * sizeof(int64_t));
  cursor += sizeof(int64_t) * keys_.size();
  memcpy(cursor, values_.data(), values_.size());
  return std::make_pair(buff, num_bytes);
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::Deserialize(const uint8_t *buffer,
                                           size_t num_bytes) {

  const uint8_t* cursor = buffer;
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);
  keys_.resize(num_keys);
  values_.resize(num_keys);

  memcpy(keys_.data(), cursor, num_keys * sizeof(int64_t));
  cursor += sizeof(int64_t) * num_keys;
  memcpy(values_.data(), cursor, num_keys * sizeof(ValueType));
}


}
}
