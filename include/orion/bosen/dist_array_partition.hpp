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
  bool sparse_index_exists_ {false};

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
  void GetAndSerializeValues(const int64_t *keys, size_t num_keys,
                             Blob *bytes_buff);

  void Repartition(const int32_t *repartition_ids);
  SendDataBuffer Serialize();
  const uint8_t* Deserialize(const uint8_t *buffer);
  const uint8_t* DeserializeAndAppend(const uint8_t *buffer);

  jl_value_t *GetGcPartition() { return nullptr; }
  void Clear();

  void AppendKeyValue(int64_t key, ValueType value);
  void AppendValue(ValueType value);
 private:
  void GetJuliaValueArray(jl_value_t **value);
  void AppendJuliaValue(jl_value_t *value);
  void AppendJuliaValueArray(jl_value_t *value);

  void RepartitionSpaceTime(const int32_t *repartition_ids);
  void Repartition1D(const int32_t *repartition_ids);

  void BuildDenseIndex();
  void BuildSparseIndex();
  void Sort();
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
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl);
  bool is_dense = dist_array_meta.IsDense();
  value_type_jl = reinterpret_cast<jl_value_t*>(type::GetJlDataType(kValueType));
  values_array_type_jl = jl_apply_array_type(
      reinterpret_cast<jl_datatype_t*>(value_type_jl), 1);
  values_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      values_array_type_jl,
      values_.data(), values_.size(), 0));
  auto *create_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "create_dist_array_accessor");
  if (is_dense) {
    Sort();
    if (keys_.size() > 0) key_start_ = keys_[0];
    key_begin_jl = jl_box_int64(key_start_);
    jl_call3(create_accessor_func, dist_array_jl, key_begin_jl,
               values_array_jl);
  } else {
    keys_array_type_jl = jl_apply_array_type(
        reinterpret_cast<jl_datatype_t*>(value_type_jl), 1);
    keys_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
        keys_array_type_jl,
        keys_.data(), keys_.size(), 0));
    jl_call3(create_accessor_func, dist_array_jl, keys_array_jl,
             values_array_jl);
  }

  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ClearAccessor() {
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense();
  const std::string &symbol = dist_array_meta.GetSymbol();
  jl_value_t *dist_array_jl = nullptr;
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl);

  if (!is_dense) {
    jl_value_t* keys_array_jl = nullptr;
    jl_value_t* values_array_jl = nullptr;
    JL_GC_PUSH2(&keys_array_jl, &values_array_jl);

    auto *get_keys_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_keys_vec");
    keys_array_jl = jl_call1(get_keys_vec_func, dist_array_jl);
    auto *keys_vec = reinterpret_cast<int64_t*>(jl_array_data(keys_array_jl));
    size_t num_keys = jl_array_len(keys_array_jl);

    auto *get_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accesor_values_vec");
    values_array_jl = jl_call1(get_values_vec_func, dist_array_jl);
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
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::CreateCacheAccessor() {
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
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl);

  value_type_jl = reinterpret_cast<jl_value_t*>(type::GetJlDataType(kValueType));
  values_array_type_jl = jl_apply_array_type(
      reinterpret_cast<jl_datatype_t*>(value_type_jl), 1);
  values_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      values_array_type_jl,
      values_.data(), values_.size(), 0));
  keys_array_type_jl = jl_apply_array_type(jl_int64_type, 1);
  keys_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl,
      keys_.data(), keys_.size(), 0));
  auto *create_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "create_dist_array_cache_accessor");
  jl_call3(create_accessor_func, dist_array_jl, keys_array_jl,
           values_array_jl);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::CreateBufferAccessor() {
  jl_value_t *dist_array_jl = nullptr;
  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl);

  auto *create_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "create_dist_array_buffer_accessor");
  jl_call1(create_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ClearCacheAccessor() {
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH2(keys_array_jl, &values_array_jl);

  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  jl_value_t *dist_array_jl = nullptr;
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl);

  auto *get_keys_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
      "dist_array_get_accessor_keys_vec");
  keys_array_jl = jl_call1(get_keys_vec_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  LOG(INFO) << "call done";
  auto *keys_vec = reinterpret_cast<int64_t*>(jl_array_data(keys_array_jl));
  size_t num_keys = jl_array_len(keys_array_jl);
  keys_.resize(num_keys);

  auto *get_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
    "dist_array_get_accessor_values_vec");
  values_array_jl = jl_call1(get_values_vec_func, dist_array_jl);
  auto *values_vec = reinterpret_cast<ValueType*>(jl_array_data(values_array_jl));
  values_.resize(num_keys);

  memcpy(keys_.data(), keys_vec, num_keys * sizeof(int64_t));
  memcpy(values_.data(), values_vec, num_keys * sizeof(ValueType));

  auto *delete_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "delete_dist_array_accessor");
  jl_call1(delete_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  sorted_ = false;
  JL_GC_POP();
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ClearBufferAccessor() {
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH2(keys_array_jl, &values_array_jl);

  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  jl_value_t *dist_array_jl = nullptr;
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl);
  bool is_dense = dist_array_meta.IsDense();
  if (!is_dense) {
    auto *get_keys_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_keys_vec");
    keys_array_jl = jl_call1(get_keys_vec_func, dist_array_jl);
    auto *keys_vec = reinterpret_cast<int64_t*>(jl_array_data(keys_array_jl));
    size_t num_keys = jl_array_len(keys_array_jl);
    keys_.resize(num_keys);
    memcpy(keys_.data(), keys_vec, num_keys * sizeof(int64_t));
  } else {
    key_start_ = 0;
  }
  auto *get_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
    "dist_array_get_accessor_values_vec");
  values_array_jl = jl_call1(get_values_vec_func, dist_array_jl);
  size_t num_values = jl_array_len(values_array_jl);
  auto *values_vec = reinterpret_cast<ValueType*>(jl_array_data(values_array_jl));
  values_.resize(num_values);
  memcpy(values_.data(), values_vec, num_values * sizeof(ValueType));

  auto *delete_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "delete_dist_array_accessor");
  jl_call1(delete_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  sorted_ = false;
  JL_GC_POP();
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::BuildKeyValueBuffersFromSparseIndex() {
  if (!sparse_index_exists_) return;
  if (!keys_.empty()) return;
  keys_.resize(sparse_index_.size());
  values_.resize(sparse_index_.size());
  auto iter = sparse_index_.begin();
  size_t i = 0;
  for (; iter != sparse_index_.end(); iter++) {
    int64_t key = iter->first;
    auto value = iter->second;
    keys_[i] = key;
    values_[i] = value;
  }
  sparse_index_.clear();
  sparse_index_exists_ = false;
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
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    auto value = values_[i];
    sparse_index_[key] = value;
  }
  keys_.clear();
  values_.clear();
  sparse_index_exists_ = true;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::GetAndSerializeValue(int64_t key,
                                                    Blob *bytes_buff) {
  auto iter = sparse_index_.find(key);
  CHECK (iter != sparse_index_.end());
  auto value = iter->second;
  bytes_buff->resize(sizeof(ValueType));
  *(reinterpret_cast<ValueType*>(bytes_buff->data())) = value;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::GetAndSerializeValues(const int64_t *keys,
                                                     size_t num_keys,
                                                     Blob *bytes_buff) {
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
    *reinterpret_cast<ValueType*>(cursor) = value;
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

  for (size_t i = 0; i < keys_.size(); i++) {
    keys_[i] = min_key + i;
  }
  sorted_ = true;
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::Clear() {
  keys_.clear();
  values_.clear();
  sparse_index_.clear();
  sparse_index_exists_ = false;
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
    sorted_ = false;
    values_.push_back(value);
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
const uint8_t*
DistArrayPartition<ValueType>::Deserialize(const uint8_t *buffer) {
  const uint8_t* cursor = buffer;
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
const uint8_t*
DistArrayPartition<ValueType>::DeserializeAndAppend(const uint8_t *buffer) {
  sorted_ = false;
  const uint8_t* cursor = buffer;
  cursor += sizeof(bool);
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  LOG(INFO) << __func__ << " num_keys = " << num_keys;
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
void
DistArrayPartition<ValueType>::GetJuliaValueArray(jl_value_t **value) {
  jl_value_t* value_array_type = nullptr;
  JL_GC_PUSH1(&value_array_type);

  value_array_type = jl_apply_array_type(type::GetJlDataType(kValueType), 1);

  *value = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(value_array_type, values_.data(), values_.size(), 0));
  JL_GC_POP();
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
  bool sparse_index_exists_ {false};

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
  void GetAndSerializeValues(const int64_t *keys, size_t num_keys,
                             Blob *bytes_buff);

  void Repartition(const int32_t *repartition_ids);
  SendDataBuffer Serialize();
  const uint8_t* Deserialize(const uint8_t *buffer);
  const uint8_t* DeserializeAndAppend(const uint8_t *buffer);
  jl_value_t *GetGcPartition() { return nullptr; }
  void Clear();

  void AppendKeyValue(int64_t key, const std::string &value);
  void AppendValue(const std::string &value);
 private:
  void GetJuliaValueArray(jl_value_t **value);
  void AppendJuliaValue(jl_value_t *value);
  void AppendJuliaValueArray(jl_value_t *value);

  void RepartitionSpaceTime(const int32_t *repartition_ids);
  void Repartition1D(const int32_t *repartition_ids);

  void BuildDenseIndex();
  void BuildSparseIndex();
  void Sort();
};

/*----- Specialized for all other types ------*/
template<>
class DistArrayPartition<void> : public AbstractDistArrayPartition {
 private:
  jl_module_t *orion_worker_module_;
  jl_value_t *dist_array_jl_;
  jl_value_t *partition_jl_;
  jl_value_t *values_array_jl_;
  stx::btree_map<int64_t, size_t> sparse_index_;
  bool sparse_index_exists_ {false};
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
  void GetAndSerializeValues(const int64_t *keys, size_t num_keys,
                             Blob *bytes_buff);

  void Repartition(const int32_t *repartition_ids);
  SendDataBuffer Serialize();
  const uint8_t* Deserialize(const uint8_t *buffer);
  const uint8_t* DeserializeAndAppend(const uint8_t *buffer);

  jl_value_t *GetGcPartition() { return partition_jl_; }
  void Clear();

 private:
  void GetJuliaValueArray(jl_value_t **value);
  void AppendJuliaValue(jl_value_t *value);
  void AppendJuliaValueArray(jl_value_t *value);

  void RepartitionSpaceTime(const int32_t *repartition_ids);
  void Repartition1D(const int32_t *repartition_ids);

  void BuildDenseIndex();
  void BuildSparseIndex();
  void Sort();
  void GetJuliaDistArray(jl_value_t** dist_array);
};

}
}
