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

  void BuildKeyValueBuffersFromSparseIndex();
  void ReadRangeDense(int64_t key_begin, size_t num_elements, jl_value_t* buff);
  void ReadRangeSparse(int64_t key_begin, size_t num_elements,
                       jl_value_t** key_buff, jl_value_t** value_buff);
  void ReadRangeSparseWithInitValue(
      int64_t key_begin, size_t num_elements,
      jl_value_t* value_buff);

 void ReadRangeSparseWithRequest(
      int64_t key_begin, size_t num_elements,
      jl_value_t** key_buff, jl_value_t** value_buff);

  void WriteRange(int64_t key_begin, size_t num_elements, jl_value_t* buff);
  void BuildIndex();

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
DistArrayPartition<ValueType>::ReadRangeDense(int64_t key_begin, size_t num_elements,
                                              jl_value_t* buff) {
  ValueType *mem = reinterpret_cast<ValueType*>(jl_array_data(buff));
  size_t offset = key_begin - key_start_;
  memcpy(mem, values_.data() + offset, num_elements * sizeof(ValueType));
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ReadRangeSparse(int64_t key_begin,
                                               size_t num_elements,
                                               jl_value_t** key_buff,
                                               jl_value_t** value_buff) {
  jl_value_t* key_array_type = nullptr;
  jl_value_t* value_array_type = nullptr;
  JL_GC_PUSH2(&key_array_type, &value_array_type);

  key_array_type = jl_apply_array_type(jl_int64_type, 1);
  value_array_type = jl_apply_array_type(
      type::GetJlDataType(kValueType), 1);

  int64_t curr_key = key_begin;
  int64_t max_key = key_begin + num_elements;
  int64_t exist_key_begin = curr_key;
  size_t num_values = 0;
  auto iter = sparse_index_.find(curr_key);
  while (iter == sparse_index_.end() &&
         curr_key + 1 < max_key) {
    curr_key++;
    iter = sparse_index_.find(curr_key);
  }

  // either max_key - 1 or (key_begin, max_key - 1) with an existant entry
  exist_key_begin = curr_key;

  while (iter != sparse_index_.end()) {
    curr_key = iter->first;
    if (curr_key >= max_key) break;
    num_values++;
    iter++;
  }

  *key_buff = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(key_array_type, num_values));
  *value_buff = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(value_array_type, num_values));

  int64_t *key_mem = reinterpret_cast<int64_t*>(jl_array_data(*key_buff));
  ValueType *value_mem = reinterpret_cast<ValueType*>(jl_array_data(*value_buff));
  size_t index = 0;
  iter = sparse_index_.find(exist_key_begin);

  while (iter != sparse_index_.end()) {
    curr_key = iter->first;
    if (curr_key >= max_key) break;
    ValueType value = iter->second;

    key_mem[index] = curr_key;
    value_mem[index] = value;
    iter++;
    index++;
  }

  CHECK(index == num_values);
  JL_GC_POP();
}

template<typename ValueType>
void DistArrayPartition<ValueType>::ReadRangeSparseWithInitValue(
    int64_t key_begin, size_t num_elements,
    jl_value_t* value_buff) {
  auto &dist_array_meta = dist_array_->GetMeta();

  ValueType init_value = *reinterpret_cast<const ValueType*>(
      dist_array_meta.GetInitValue().data());

  int64_t curr_key = key_begin;
  int64_t max_key = key_begin + num_elements;

  ValueType *value_mem = reinterpret_cast<ValueType*>(jl_array_data(value_buff));
  size_t index = 0;
  auto iter = sparse_index_.find(curr_key);

  while (iter == sparse_index_.end() &&
         curr_key + 1 < max_key) {
    value_mem[index] = init_value;
    curr_key++;
    index++;
    iter = sparse_index_.find(curr_key);
  }

  if (iter != sparse_index_.end()) {
    while (iter != sparse_index_.end() &&
           iter->first < max_key) {
      while (curr_key <= iter->first) {
        value_mem[index] = iter->second;
        index++;
        curr_key++;
      }
      iter++;
    }
    for (; curr_key < max_key; curr_key++) {
      value_mem[index] = init_value;
      index++;
    }
  } else {
    value_mem[index] = init_value;
    index++;
  }

  CHECK(index == num_elements);
  JL_GC_POP();
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::ReadRangeSparseWithRequest(
    int64_t key_begin, size_t num_elements,
    jl_value_t** key_buff, jl_value_t** value_buff) {
  jl_value_t* key_array_type = nullptr;
  jl_value_t* value_array_type = nullptr;
  jl_value_t* key_jl = nullptr;
  jl_value_t* temp_value = nullptr;
  JL_GC_PUSH4(&key_array_type, &value_array_type,
              &key_jl, &temp_value);

  key_array_type = jl_apply_array_type(jl_int64_type, 1);
  value_array_type = jl_apply_array_type(type::GetJlDataType(kValueType), 1);

  *key_buff = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(key_array_type, 0));
  *value_buff = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(value_array_type, 0));

  int64_t curr_key = key_begin;
  int64_t max_key = key_begin + num_elements;
  while (curr_key < max_key) {
    auto iter = sparse_index_.find(curr_key);
    if (iter == sparse_index_.end()) {
      julia_requester_->RequestDistArrayData(dist_array_->kId, curr_key,
                                        kValueType, &temp_value);

    } else {
      ValueType value = iter->second;
      JuliaEvaluator::BoxValue(kValueType, reinterpret_cast<uint8_t*>(&value),
                               &temp_value);
    }
    if (temp_value != nullptr) {
      key_jl = jl_box_int64(iter->first);
      jl_array_ptr_1d_push(reinterpret_cast<jl_array_t*>(*key_buff), key_jl);
      jl_array_ptr_1d_push(reinterpret_cast<jl_array_t*>(*value_buff), temp_value);
    }
    curr_key++;
  }

  JL_GC_POP();
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::WriteRange(int64_t key_begin, size_t num_elements,
                                          jl_value_t* buff) {
  CHECK(jl_is_array(buff));
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense();
  CHECK_GE(key_start_, 0) << " key_begin = " << key_begin;
  ValueType *value_mem = reinterpret_cast<ValueType*>(jl_array_data(buff));
  if (is_dense) {
    size_t offset = key_begin - key_start_;
    memcpy(values_.data() + offset, value_mem, num_elements * sizeof(ValueType));
  } else {
    for (size_t i = 0; i < num_elements; i++) {
      sparse_index_[key_begin + i] = value_mem[i];
    }
  }
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
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::AppendValue(ValueType value) {
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
  size_t num_bytes = sizeof(size_t)
                     + keys_.size() * (sizeof(int64_t) + sizeof(ValueType));
  uint8_t* buff = new uint8_t[num_bytes];
  uint8_t* cursor = buff;
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
  const uint8_t* cursor = buffer;
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
}

template<typename ValueType>
void
DistArrayPartition<ValueType>::AppendJuliaValueArray(jl_value_t *value) {
  size_t num_elements = jl_array_len(reinterpret_cast<jl_array_t*>(value));
  ValueType* value_array = reinterpret_cast<ValueType*>(jl_array_data(reinterpret_cast<jl_array_t*>(value)));
  for (size_t i = 0; i < num_elements; i++) {
    values_.push_back(value_array[i]);
  }
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
  void BuildKeyValueBuffersFromSparseIndex();
  void ReadRangeDense(int64_t key_begin, size_t num_elements, jl_value_t* buff);
  void ReadRangeSparse(int64_t key_begin, size_t num_elements,
                       jl_value_t** key_buff, jl_value_t** value_buff);
  void ReadRangeSparseWithInitValue(
      int64_t key_begin, size_t num_elements,
      jl_value_t* value_buff);
 void ReadRangeSparseWithRequest(
      int64_t key_begin, size_t num_elements,
      jl_value_t** key_buff, jl_value_t** value_buff);

  void WriteRange(int64_t key_begin, size_t num_elements, jl_value_t* buff);
  void BuildIndex();

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
};

/*----- Specialized for String (const char*) ------*/
template<>
class DistArrayPartition<void> : public AbstractDistArrayPartition {
 private:
  jl_module_t *orion_worker_module_;
  jl_value_t *dist_array_jl_;
  jl_value_t *partition_jl_;
  stx::btree_map<int64_t, size_t> sparse_index_;
  std::vector<size_t> julia_array_index_;
  bool sparse_index_exists_ {false};
 public:
  DistArrayPartition(DistArray *dist_array,
                     const Config &config,
                     type::PrimitiveType value_type,
                     JuliaThreadRequester *julia_requester);
  ~DistArrayPartition();
  void BuildKeyValueBuffersFromSparseIndex();
  void ReadRangeDense(int64_t key_begin, size_t num_elements, jl_value_t* buff);
  void ReadRangeSparse(int64_t key_begin, size_t num_elements,
                       jl_value_t** key_buff, jl_value_t** value_buff);
  void ReadRangeSparseWithInitValue(
      int64_t key_begin, size_t num_elements,
      jl_value_t* value_buff);
 void ReadRangeSparseWithRequest(
      int64_t key_begin, size_t num_elements,
      jl_value_t** key_buff, jl_value_t** value_buff);

  void WriteRange(int64_t key_begin, size_t num_elements, jl_value_t* buff);
  void BuildIndex();

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

  void GetJuliaDistArray(jl_value_t** dist_array);
};

}
}
