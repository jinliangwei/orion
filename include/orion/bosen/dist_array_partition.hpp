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
#include <julia.h>

#ifdef ORION_USE_HDFS
#include <hdfs.h>
#endif

namespace orion {
namespace bosen {

template<typename ValueType>
class DistArrayPartition : public AbstractDistArrayPartition {
 private:
  std::vector<ValueType> values_;
  stx::btree_map<int64_t, ValueType> sparse_index_;

 public:
  DistArrayPartition(DistArray *dist_array,
                     const Config &config,
                     type::PrimitiveType value_type,
                     JuliaThreadRequester *julia_requester);
  ~DistArrayPartition();

  void CreateAccessor();
  void ClearAccessor();
  void CreateCacheAccessor();
  void CreateBufferAccessor();
  void ClearCacheAccessor();
  void ClearBufferAccessor();
  void BuildKeyValueBuffersFromSparseIndex();
  void GetAndSerializeValue(int64_t key, Blob *bytes_buff);
  void GetAndSerializeValues(int64_t *keys, size_t num_keys,
                             Blob *bytes_buff);

  SendDataBuffer Serialize();
  void HashSerialize(ExecutorDataBufferMap *data_buffer_map);
  uint8_t* Deserialize(uint8_t *buffer);
  uint8_t* DeserializeAndAppend(uint8_t *buffer);

  uint8_t* DeserializeAndOverwrite(uint8_t *buffer);
  void ApplyBufferedUpdates(
      const AbstractDistArrayPartition* dist_array_buffer,
      const std::vector<const AbstractDistArrayPartition*> &helper_dist_array_buffers,
      const std::string &apply_buffer_func_name);

  void Clear();

  void AppendKeyValue(int64_t key, ValueType value);
  void AppendValue(ValueType value);
 private:
  void RepartitionSpaceTime(const int32_t *repartition_ids);
  void Repartition1D(const int32_t *repartition_ids);

  void GetJuliaValueArray(jl_value_t **value);
  void GetJuliaValueArray(std::vector<int64_t> &keys,
                          jl_value_t **value_array);
  void SetJuliaValues(std::vector<int64_t> &keys,
                      jl_value_t *values);
  void AppendJuliaValue(jl_value_t *value);
  void AppendJuliaValueArray(jl_value_t *value);

  void BuildDenseIndex();
  void BuildSparseIndex();
  void Sort();
  void ShrinkValueVecToFit();
};

/*---- template general implementation -----*/
template<typename ValueType>
DistArrayPartition<ValueType>::DistArrayPartition(
    DistArray *dist_array,
    const Config &config,
    type::PrimitiveType value_type,
    JuliaThreadRequester *julia_requester):
    AbstractDistArrayPartition(dist_array, config, value_type, julia_requester) { }

template<typename ValueType>
DistArrayPartition<ValueType>::~DistArrayPartition() { }

template<typename ValueType>
void
DistArrayPartition<ValueType>::CreateAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 6);
  jl_value_t *&value_type_jl = jl_values[0];
  jl_value_t *&values_array_type_jl = jl_values[1];
  jl_value_t *&values_array_jl = jl_values[2];
  jl_value_t *&key_begin_jl = jl_values[3];
  jl_value_t *&keys_array_type_jl = jl_values[4];
  jl_value_t *&keys_array_jl = jl_values[5];

  jl_value_t *dist_array_jl = nullptr;
  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  JuliaEvaluator::GetVarJlValue(symbol, &dist_array_jl);
  bool is_dense = dist_array_meta.IsDense() && dist_array_meta.IsContiguousPartitions();
  value_type_jl = reinterpret_cast<jl_value_t*>(type::GetJlDataType(kValueType));
  values_array_type_jl = jl_apply_array_type(value_type_jl, 1);
  values_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      values_array_type_jl,
      values_.data(), values_.size(), 0));
  auto *create_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "create_dist_array_accessor");
  if (is_dense) {
    Sort();
    key_begin_jl = jl_box_int64(keys_.size() > 0 ? keys_[0] : 0);
    jl_call3(create_accessor_func, dist_array_jl, key_begin_jl,
             values_array_jl);
  } else {
    keys_array_type_jl = jl_apply_array_type(value_type_jl, 1);
    keys_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
        keys_array_type_jl,
        keys_.data(), keys_.size(), 0));
    jl_call3(create_accessor_func, dist_array_jl, keys_array_jl,
             values_array_jl);
  }

  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  storage_type_ = DistArrayPartitionStorageType::kAccessor;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ClearAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kAccessor);
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense() && dist_array_meta.IsContiguousPartitions();
  const std::string &symbol = dist_array_meta.GetSymbol();
  jl_value_t *dist_array_jl = nullptr;
  JuliaEvaluator::GetVarJlValue(symbol, &dist_array_jl);

  if (!is_dense) {
    jl_value_t* tuple_jl = nullptr;
    jl_value_t* keys_array_jl = nullptr;
    jl_value_t* values_array_jl = nullptr;
    JL_GC_PUSH3(&tuple_jl, &keys_array_jl, &values_array_jl);

    auto *get_keys_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_keys_values_vec");
    tuple_jl = jl_call1(get_keys_values_vec_func, dist_array_jl);
    keys_array_jl = jl_get_nth_field(tuple_jl, 0);
    values_array_jl = jl_get_nth_field(tuple_jl, 1);
    auto *keys_vec = reinterpret_cast<int64_t*>(jl_array_data(keys_array_jl));
    size_t num_keys = jl_array_len(keys_array_jl);
    auto *values_vec = reinterpret_cast<ValueType*>(jl_array_data(values_array_jl));
    keys_.resize(num_keys);
    values_.resize(num_keys);
    memcpy(keys_.data(), keys_vec, num_keys * sizeof(int64_t));
    memcpy(values_.data(), values_vec, num_keys * sizeof(ValueType));
    sorted_ = false;
    JL_GC_POP();
  }
  auto *delete_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "delete_dist_array_accessor");
  jl_call1(delete_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::CreateCacheAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 5);
  jl_value_t *&value_type_jl = jl_values[0];
  jl_value_t *&values_array_type_jl = jl_values[1];
  jl_value_t *&values_array_jl = jl_values[2];
  jl_value_t *&keys_array_type_jl = jl_values[3];
  jl_value_t *&keys_array_jl = jl_values[4];

  jl_value_t *dist_array_jl = nullptr;
  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  JuliaEvaluator::GetVarJlValue(symbol, &dist_array_jl);

  value_type_jl = reinterpret_cast<jl_value_t*>(type::GetJlDataType(kValueType));
  values_array_type_jl = jl_apply_array_type(value_type_jl, 1);
  values_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      values_array_type_jl,
      values_.data(), values_.size(), 0));

  keys_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  keys_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl,
      keys_.data(), keys_.size(), 0));
  auto *create_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "create_dist_array_cache_accessor");
  jl_call3(create_accessor_func, dist_array_jl, keys_array_jl,
           values_array_jl);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  storage_type_ = DistArrayPartitionStorageType::kAccessor;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::CreateBufferAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer)
      << " storage_type = " << static_cast<int>(storage_type_);
  jl_value_t *dist_array_jl = nullptr;
  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  JuliaEvaluator::GetVarJlValue(symbol, &dist_array_jl);

  auto *create_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "create_dist_array_buffer_accessor");
  jl_call1(create_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  storage_type_ = DistArrayPartitionStorageType::kAccessor;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ClearCacheAccessor() {
  LOG(INFO) << __func__ << " dist_array_id = " << dist_array_->kId;
  CHECK(storage_type_ == DistArrayPartitionStorageType::kAccessor);
  jl_value_t* tuple_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH3(&tuple_jl, &keys_array_jl, &values_array_jl);

  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  jl_value_t *dist_array_jl = nullptr;
  JuliaEvaluator::GetVarJlValue(symbol, &dist_array_jl);

  auto *get_keys_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
      "dist_array_get_accessor_keys_values_vec");
  tuple_jl = jl_call1(get_keys_values_vec_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  keys_array_jl = jl_get_nth_field(tuple_jl, 0);
  values_array_jl = jl_get_nth_field(tuple_jl, 1);
  auto *keys_vec = reinterpret_cast<int64_t*>(jl_array_data(keys_array_jl));
  size_t num_keys = jl_array_len(keys_array_jl);
  auto *values_vec = reinterpret_cast<ValueType*>(jl_array_data(values_array_jl));
  keys_.resize(num_keys);
  values_.resize(num_keys);

  memcpy(keys_.data(), keys_vec, num_keys * sizeof(int64_t));
  memcpy(values_.data(), values_vec, num_keys * sizeof(ValueType));

  auto *delete_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "delete_dist_array_accessor");
  jl_call1(delete_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  sorted_ = false;
  JL_GC_POP();
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ClearBufferAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kAccessor);
  jl_value_t* tuple_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH3(&tuple_jl, &keys_array_jl, &values_array_jl);

  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  jl_value_t *dist_array_jl = nullptr;
  JuliaEvaluator::GetVarJlValue(symbol, &dist_array_jl);
  bool is_dense = dist_array_meta.IsDense();
  CHECK(dist_array_meta.IsContiguousPartitions());
  if (!is_dense) {
    auto *get_keys_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_keys_values_vec");
    tuple_jl = jl_call1(get_keys_values_vec_func, dist_array_jl);
    JuliaEvaluator::AbortIfException();
    keys_array_jl = jl_get_nth_field(tuple_jl, 0);
    values_array_jl = jl_get_nth_field(tuple_jl, 1);
    auto *keys_vec = reinterpret_cast<int64_t*>(jl_array_data(keys_array_jl));
    size_t num_keys = jl_array_len(keys_array_jl);
    auto *values_vec = reinterpret_cast<ValueType*>(jl_array_data(values_array_jl));
    keys_.resize(num_keys);
    values_.resize(num_keys);

    memcpy(keys_.data(), keys_vec, num_keys * sizeof(int64_t));
    memcpy(values_.data(), values_vec, num_keys * sizeof(ValueType));
  } else {
    auto *get_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_values_vec");
    values_array_jl = jl_call1(get_values_vec_func, dist_array_jl);
    size_t num_values = jl_array_len(values_array_jl);
    auto *values_vec = reinterpret_cast<ValueType*>(jl_array_data(values_array_jl));
    values_.resize(num_values);
    memcpy(values_.data(), values_vec, num_values * sizeof(ValueType));
    keys_.resize(num_values);
    for (size_t i = 0; i < num_values; i++) {
      keys_[i] = i;
    }
  }

  auto *delete_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "delete_dist_array_accessor");
  jl_call1(delete_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  sorted_ = false;
  JL_GC_POP();
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::BuildKeyValueBuffersFromSparseIndex() {
  if (storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer) return;
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  CHECK(keys_.empty());
  keys_.resize(sparse_index_.size());
  values_.resize(sparse_index_.size());
  auto iter = sparse_index_.begin();
  size_t i = 0;
  for (; iter != sparse_index_.end(); iter++, i++) {
    int64_t key = iter->first;
    auto value = iter->second;
    keys_[i] = key;
    values_[i] = value;
  }
  sorted_ = true;
  sparse_index_.clear();
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::BuildDenseIndex() {
  Sort();
  if (keys_.size() > 0) key_start_ = keys_[0];
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::BuildSparseIndex() {
  LOG(INFO) << __func__;
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    auto value = values_[i];
    sparse_index_[key] = value;
  }
  {
    std::vector<int64_t> empty_buff;
    keys_.swap(empty_buff);
  }
  {
    std::vector<ValueType> empty_value_buff;
    values_.swap(empty_value_buff);
  }
  storage_type_ = DistArrayPartitionStorageType::kSparseIndex;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::GetAndSerializeValue(int64_t key,
                                                    Blob *bytes_buff) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  auto iter = sparse_index_.find(key);
  CHECK (iter != sparse_index_.end()) << __func__ << " key = " << key;
  auto value = iter->second;
  LOG(INFO) << __func__ << " key = " << key
            << " value = " << value;
  bytes_buff->resize(sizeof(ValueType));
  *(reinterpret_cast<ValueType*>(bytes_buff->data())) = value;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::GetAndSerializeValues(int64_t *keys,
                                                     size_t num_keys,
                                                     Blob *bytes_buff) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  bytes_buff->resize(sizeof(bool) + sizeof(size_t)
                     + (sizeof(int64_t) + sizeof(ValueType)) * num_keys);

  auto *cursor = bytes_buff->data();
  *reinterpret_cast<bool*>(cursor) = false;
  cursor += sizeof(bool);
  *reinterpret_cast<size_t*>(cursor) = num_keys;
  cursor += sizeof(size_t);
  memcpy(cursor, keys, sizeof(int64_t) * num_keys);
  cursor += sizeof(int64_t) * num_keys;
  for (size_t i = 0; i < num_keys; i++) {
    auto key = keys[i];
    auto iter = sparse_index_.find(key);
    CHECK (iter != sparse_index_.end()) << " i = " << i
                                        << " key = " << key
                                        << " size = " << sparse_index_.size();
    auto value = iter->second;
    LOG(INFO) << __func__ << " key = " << key
              << " value = " << value;
    *reinterpret_cast<ValueType*>(cursor) = value;
    cursor += sizeof(ValueType);
  }
}


template<typename ValueType>
void
DistArrayPartition<ValueType>::Sort() {
  if (sorted_) return;
  if (keys_.size() == 0) return;
  int64_t min_key = keys_[0];
  CHECK(values_.size() == keys_.size());
  for (auto key : keys_) {
    min_key = std::min(key, min_key);
  }
  std::vector<size_t> perm(keys_.size());
  std::vector<ValueType> values_temp(values_);

  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&] (const size_t &i, const size_t &j) {
              return keys_[i] < keys_[j];
            });
  std::transform(perm.begin(), perm.end(), values_.begin(),
                 [&](size_t i) { return values_temp[i]; });

  auto &meta = dist_array_->GetMeta();
  if (meta.IsContiguousPartitions()) {
    for (size_t i = 0; i < keys_.size(); i++) {
      keys_[i] = min_key + i;
    }
  } else {
    std::sort(keys_.begin(), keys_.end());
  }

  sorted_ = true;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ShrinkValueVecToFit() {
  std::vector<ValueType> temp_values;
  values_.swap(temp_values);
  values_.resize(temp_values.size());
  memcpy(values_.data(), temp_values.data(), temp_values.size() * sizeof(ValueType));
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::Clear() {
  CHECK(storage_type_ != DistArrayPartitionStorageType::kAccessor);
  {
    std::vector<int64_t> empty_buff;
    keys_.swap(empty_buff);
  }
  {
    std::vector<ValueType> empty_value_buff;
    values_.swap(empty_value_buff);
  }
  sparse_index_.clear();
  key_start_ = -1;
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::AppendKeyValue(int64_t key,
                                              ValueType value) {
  keys_.push_back(key);
  values_.push_back(value);
  sorted_ = false;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::AppendValue(ValueType value) {
  values_.push_back(value);
  sorted_ = false;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::RepartitionSpaceTime(
    const int32_t *repartition_ids) {
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    ValueType value = values_[i];
    int32_t space_partition_id = repartition_ids[i * 2];
    int32_t time_partition_id = repartition_ids[i * 2 + 1];
    auto new_partition_pair = dist_array_->GetAndCreateLocalPartition(space_partition_id,
                                            time_partition_id);
    auto *partition_to_add = dynamic_cast<DistArrayPartition<ValueType>*>(
        new_partition_pair.first);
    partition_to_add->AppendKeyValue(key, value);
  }
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::Repartition1D(
    const int32_t *repartition_ids) {
   for (size_t i = 0; i < keys_.size(); i++) {
     int64_t key = keys_[i];
     ValueType value = values_[i];
     int32_t repartition_id = repartition_ids[i];
     auto new_partition_pair =
         dist_array_->GetAndCreateLocalPartition(repartition_id);
     auto *partition_to_add = dynamic_cast<DistArrayPartition<ValueType>*>(
         new_partition_pair.first);
     partition_to_add->AppendKeyValue(key, value);
   }
}

template<typename ValueType>
SendDataBuffer
DistArrayPartition<ValueType>::Serialize() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  size_t num_bytes = sizeof(bool) + sizeof(size_t)
                     + keys_.size() * (sizeof(int64_t) + sizeof(ValueType));
  uint8_t* buff = new uint8_t[num_bytes];
  uint8_t* cursor = buff;
  *(reinterpret_cast<bool*>(cursor)) = sorted_;
  cursor += sizeof(bool);
  *(reinterpret_cast<size_t*>(cursor)) = keys_.size();
  cursor += sizeof(size_t);
  memcpy(cursor, keys_.data(), keys_.size() * sizeof(int64_t));
  cursor += sizeof(int64_t) * keys_.size();
  memcpy(cursor, values_.data(), values_.size() * sizeof(ValueType));
  return std::make_pair(buff, num_bytes);
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::HashSerialize(
    ExecutorDataBufferMap *data_buffer_map) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  std::unordered_map<int32_t, size_t> server_accum_size;
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    int32_t server_id = key % kConfig.kNumServers;
    auto iter = server_accum_size.find(server_id);
    if (iter == server_accum_size.end()) {
      server_accum_size[server_id] = 1;
    } else {
      server_accum_size[server_id] += 1;
    }
  }

  std::unordered_map<int32_t, uint8_t*> server_cursor;
  std::unordered_map<int32_t, uint8_t*> server_value_cursor;
  for (auto &accum_size_pair : server_accum_size) {
    size_t num_key_values = accum_size_pair.second;
    accum_size_pair.second *= sizeof(int64_t) + sizeof(ValueType);
    accum_size_pair.second += sizeof(bool) + sizeof(size_t);
    int32_t server_id = accum_size_pair.first;
    auto iter_pair = data_buffer_map->emplace(server_id,
                                              Blob(accum_size_pair.second));
    server_cursor[server_id] = iter_pair.first->second.data();
    *(reinterpret_cast<bool*>(server_cursor[server_id])) = false;
    server_cursor[server_id] += sizeof(bool);
    *(reinterpret_cast<size_t*>(server_cursor[server_id])) = num_key_values;
    server_cursor[server_id] += sizeof(size_t);
    server_value_cursor[server_id] = server_cursor[server_id] \
                                     + num_key_values * sizeof(int64_t);
  }

  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    int32_t server_id = key % kConfig.kNumServers;

    *reinterpret_cast<int64_t*>(server_cursor[server_id]) = key;
    server_cursor[server_id] += sizeof(int64_t);
    *reinterpret_cast<ValueType*>(server_value_cursor[server_id]) = values_[i];
    server_value_cursor[server_id] += sizeof(ValueType);
  }
}

template<typename ValueType>
uint8_t*
DistArrayPartition<ValueType>::Deserialize(uint8_t *buffer) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  uint8_t* cursor = buffer;
  sorted_ = *(reinterpret_cast<const bool*>(cursor));
  cursor += sizeof(bool);
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);
  keys_.resize(num_keys);
  values_.resize(num_keys);

  memcpy(keys_.data(), cursor, num_keys * sizeof(int64_t));
  cursor += sizeof(int64_t) * num_keys;
  memcpy(values_.data(), cursor, num_keys * sizeof(ValueType));
  cursor += num_keys * sizeof(ValueType);

  return cursor;
}

template<typename ValueType>
uint8_t*
DistArrayPartition<ValueType>::DeserializeAndAppend(uint8_t *buffer) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  sorted_ = false;
  uint8_t* cursor = buffer;
  cursor += sizeof(bool);
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);
  size_t orig_num_keys = keys_.size();
  keys_.resize(orig_num_keys + num_keys);
  values_.resize(orig_num_keys + num_keys);

  memcpy(keys_.data() + orig_num_keys, cursor,
         num_keys * sizeof(int64_t));
  cursor += sizeof(int64_t) * num_keys;
  memcpy(values_.data() + orig_num_keys, cursor,
         num_keys * sizeof(ValueType));
  cursor += num_keys * sizeof(ValueType);
  return cursor;
}

template<typename ValueType>
uint8_t*
DistArrayPartition<ValueType>::DeserializeAndOverwrite(
    uint8_t *buffer) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);

  uint8_t* cursor = buffer;
  cursor += sizeof(bool);
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);

  uint8_t* value_cursor = cursor + sizeof(int64_t) * num_keys;
  for (size_t i = 0; i < num_keys; i++) {
    auto key = *(reinterpret_cast<const int64_t*>(cursor));
    cursor += sizeof(int64_t);
    auto value = *(reinterpret_cast<const ValueType*>(value_cursor));
    value_cursor += sizeof(ValueType);
    sparse_index_[key] = value;
  }

  return value_cursor;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::GetJuliaValueArray(jl_value_t **value) {
  jl_value_t* value_array_type = nullptr;
  JL_GC_PUSH1(&value_array_type);

  value_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(
      type::GetJlDataType(kValueType)), 1);
  *value = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(value_array_type, values_.data(), values_.size(), 0));
  JL_GC_POP();
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::GetJuliaValueArray(std::vector<int64_t> &keys,
                                                  jl_value_t **value_array) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t* value_array_type = nullptr;
  JL_GC_PUSH1(&value_array_type);

  value_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(
      type::GetJlDataType(kValueType)), 1);

  *value_array = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(
      value_array_type,
      keys.size()));
  ValueType *value_array_mem = reinterpret_cast<ValueType*>(jl_array_data(*value_array));
  auto *cursor = value_array_mem;
  for (size_t i = 0; i < keys.size(); i++, cursor++) {
    auto key = keys[i];
    auto iter = sparse_index_.find(key);
    CHECK (iter != sparse_index_.end()) << " i = " << i
                                        << " key = " << key
                                        << " size = " << sparse_index_.size();
    auto value = iter->second;
    *cursor = value;
  }
  JL_GC_POP();
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::SetJuliaValues(std::vector<int64_t> &keys,
                                              jl_value_t *values) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  ValueType *values_mem = reinterpret_cast<ValueType*>(jl_array_data(reinterpret_cast<jl_array_t*>(values)));
  auto *cursor = values_mem;
  for (size_t i = 0; i < keys.size(); i++, cursor++) {
    auto key = keys[i];
    sparse_index_[key] = *cursor;
  }
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::AppendJuliaValue(jl_value_t *value) {
  ValueType value_buff;
  JuliaEvaluator::UnboxValue(value, kValueType, reinterpret_cast<uint8_t*>(&value_buff));
  values_.push_back(value_buff);
  sorted_ = false;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::AppendJuliaValueArray(jl_value_t *value) {
  size_t num_elements = jl_array_len(reinterpret_cast<jl_array_t*>(value));
  ValueType* value_array = reinterpret_cast<ValueType*>(jl_array_data(reinterpret_cast<jl_array_t*>(value)));
  for (size_t i = 0; i < num_elements; i++) {
    values_.push_back(value_array[i]);
  }
  sorted_ = false;
}

/*----- Specialized for String (const char*) ------*/
template<>
class DistArrayPartition<std::string> : public AbstractDistArrayPartition {
 private:
  std::vector<std::string> values_;
  stx::btree_map<int64_t, std::string> sparse_index_;

 public:
  DistArrayPartition(DistArray *dist_array,
                     const Config &config,
                     type::PrimitiveType value_type,
                     JuliaThreadRequester *julia_requester);
  ~DistArrayPartition();

  void CreateAccessor();
  void ClearAccessor();
  void CreateCacheAccessor();
  void CreateBufferAccessor();
  void ClearCacheAccessor();
  void ClearBufferAccessor();
  void BuildKeyValueBuffersFromSparseIndex();
  void GetAndSerializeValue(int64_t key, Blob *bytes_buff);
  void GetAndSerializeValues(int64_t *keys, size_t num_keys,
                             Blob *bytes_buff);

  SendDataBuffer Serialize();
  void HashSerialize(ExecutorDataBufferMap *data_buffer_map);
  uint8_t* Deserialize(uint8_t *buffer);
  uint8_t* DeserializeAndAppend(uint8_t *buffer);

  uint8_t* DeserializeAndOverwrite(uint8_t *buffer);
  void ApplyBufferedUpdates(
      const AbstractDistArrayPartition* dist_array_buffer,
      const std::vector<const AbstractDistArrayPartition*> &helper_dist_array_buffers,
      const std::string &apply_buffer_func_name);

  void Clear();

  void AppendKeyValue(int64_t key, const std::string &value);
  void AppendValue(const std::string &value);
 private:
  void GetJuliaValueArray(jl_value_t **value);
  void GetJuliaValueArray(std::vector<int64_t> &keys,
                          jl_value_t **value_array);
  void SetJuliaValues(std::vector<int64_t> &keys,
                      jl_value_t *values);
  void AppendJuliaValue(jl_value_t *value);
  void AppendJuliaValueArray(jl_value_t *value);

  void RepartitionSpaceTime(const int32_t *repartition_ids);
  void Repartition1D(const int32_t *repartition_ids);

  void BuildDenseIndex();
  void BuildSparseIndex();
  void Sort();
  void ShrinkValueVecToFit();
};

/*----- Specialized for all other types ------*/
template<>
class DistArrayPartition<void> : public AbstractDistArrayPartition {
 private:
  std::string ptr_str_;

 public:
  DistArrayPartition(DistArray *dist_array,
                     const Config &config,
                     type::PrimitiveType value_type,
                     JuliaThreadRequester *julia_requester);
  ~DistArrayPartition();
  void CreateAccessor();
  void ClearAccessor();
  void CreateCacheAccessor();
  void CreateBufferAccessor();
  void ClearCacheAccessor();
  void ClearBufferAccessor();
  void BuildKeyValueBuffersFromSparseIndex();
  void GetAndSerializeValue(int64_t key, Blob *bytes_buff);
  void GetAndSerializeValues(int64_t *keys, size_t num_keys,
                             Blob *bytes_buff);

  SendDataBuffer Serialize();
  void HashSerialize(ExecutorDataBufferMap *data_buffer_map);
  uint8_t* Deserialize(uint8_t *buffer);
  uint8_t* DeserializeAndAppend(uint8_t *buffer);

  uint8_t* DeserializeAndOverwrite(uint8_t *buffer);
  void ApplyBufferedUpdates(
      const AbstractDistArrayPartition* dist_array_buffer,
      const std::vector<const AbstractDistArrayPartition*> &helper_dist_array_buffers,
      const std::string &apply_buffer_func_name);

  void Clear();

 private:
  void GetJuliaValueArray(jl_value_t **value);
  void GetJuliaValueArray(std::vector<int64_t> &keys,
                          jl_value_t **value_array);
  void SetJuliaValues(std::vector<int64_t> &keys,
                      jl_value_t *values);
  void AppendJuliaValue(jl_value_t *value);
  void AppendJuliaValueArray(jl_value_t *value);

  void RepartitionSpaceTime(const int32_t *repartition_ids);
  void Repartition1D(const int32_t *repartition_ids);

  void BuildDenseIndex();
  void BuildSparseIndex();
  void Sort();
  void GetJuliaDistArray(jl_value_t** dist_array);
  void ShrinkValueVecToFit();
};

}
}
