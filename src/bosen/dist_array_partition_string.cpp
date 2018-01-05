#include <orion/bosen/dist_array_partition.hpp>
#include <julia.h>
namespace orion {
namespace bosen {
/*---- template std::string implementation -----*/
DistArrayPartition<std::string>::DistArrayPartition(
    DistArray *dist_array,
    const Config &config,
    type::PrimitiveType value_type,
    JuliaThreadRequester *julia_requester):
    AbstractDistArrayPartition(dist_array, config, value_type, julia_requester) { }

DistArrayPartition<std::string>::~DistArrayPartition() { }

void
DistArrayPartition<std::string>::BuildKeyValueBuffersFromSparseIndex() {
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

void
DistArrayPartition<std::string>::ReadRangeDense(int64_t key_begin, size_t num_elements,
                                                jl_value_t* buff) {
  jl_value_t *string_jl = nullptr;
  JL_GC_PUSH1(&string_jl);
  size_t offset = key_begin - key_start_;
  for (size_t i = 0; i < num_elements; i++) {
    string_jl = jl_cstr_to_string(values_[i + offset].c_str());
    jl_arrayset(reinterpret_cast<jl_array_t*>(buff), string_jl, i);
  }
  JL_GC_POP();
}

void
DistArrayPartition<std::string>::ReadRangeSparse(int64_t key_begin,
                                                 size_t num_elements,
                                                 jl_value_t** key_buff,
                                                 jl_value_t** value_buff) {
  jl_value_t* key_array_type = nullptr;
  jl_value_t* value_array_type = nullptr;
  jl_value_t *string_jl = nullptr;
  JL_GC_PUSH3(&key_array_type, &value_array_type, &string_jl);

  key_array_type = jl_apply_array_type(jl_int64_type, 1);
  value_array_type = jl_apply_array_type(jl_string_type, 1);

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

  int64_t *key_mem = reinterpret_cast<int64_t*>(
      jl_array_data(reinterpret_cast<jl_array_t*>(*key_buff)));
  size_t index = 0;
  iter = sparse_index_.find(exist_key_begin);

  while (iter != sparse_index_.end()) {
    curr_key = iter->first;
    if (curr_key >= max_key) break;
    std::string& value = iter->second;

    key_mem[index] = curr_key;
    string_jl = jl_cstr_to_string(value.c_str());
    jl_arrayset(reinterpret_cast<jl_array_t*>(*value_buff), string_jl, index);

    iter++;
    index++;
  }

  CHECK(index == num_values);
  JL_GC_POP();
}

void
DistArrayPartition<std::string>::ReadRangeSparseWithInitValue(
    int64_t key_begin, size_t num_elements,
    jl_value_t* value_buff) {
  auto &dist_array_meta = dist_array_->GetMeta();

  std::string init_value(reinterpret_cast<const char*>(
      dist_array_meta.GetInitValue().data()));

  jl_value_t *init_string_jl = nullptr;
  jl_value_t *string_jl = nullptr;
  JL_GC_PUSH2(&init_string_jl, &string_jl);
  init_string_jl = jl_cstr_to_string(init_value.c_str());

  int64_t curr_key = key_begin;
  int64_t max_key = key_begin + num_elements;

  size_t index = 0;
  auto iter = sparse_index_.find(curr_key);

  while (iter == sparse_index_.end() &&
         curr_key + 1 < max_key) {
    jl_arrayset(reinterpret_cast<jl_array_t*>(value_buff),
                init_string_jl, index);
    curr_key++;
    index++;
    iter = sparse_index_.find(curr_key);
  }

  if (iter != sparse_index_.end()) {
    while (iter != sparse_index_.end() &&
           iter->first < max_key) {
      while (curr_key <= iter->first) {
        string_jl = jl_cstr_to_string(iter->second.c_str());
        jl_arrayset(reinterpret_cast<jl_array_t*>(value_buff), string_jl, index);
        index++;
        curr_key++;
      }
      iter++;
    }
    for (; curr_key < max_key; curr_key++) {
      jl_arrayset(reinterpret_cast<jl_array_t*>(value_buff), init_string_jl, index);
      index++;
    }
  } else {
    jl_arrayset(reinterpret_cast<jl_array_t*>(value_buff), init_string_jl, index);
    index++;
  }

  CHECK(index == num_elements);
  JL_GC_POP();
}

void
DistArrayPartition<std::string>::ReadRangeSparseWithRequest(
    int64_t key_begin, size_t num_elements,
    jl_value_t **key_buff, jl_value_t** value_buff) {
  jl_value_t* key_array_type = nullptr;
  jl_value_t* value_array_type = nullptr;
  jl_value_t* key_jl = nullptr;
  jl_value_t* temp_value = nullptr;
  JL_GC_PUSH4(&key_array_type, &value_array_type,
              &key_jl, &temp_value);

  key_array_type = jl_apply_array_type(jl_int64_type, 1);
  value_array_type = jl_apply_array_type(type::GetJlDataType(kValueType),
                                         1);

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
      temp_value = jl_cstr_to_string(iter->second.c_str());
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

void
DistArrayPartition<std::string>::WriteRange(int64_t key_begin, size_t num_elements,
                                            jl_value_t* buff) {
  jl_value_t *string_jl = nullptr;
  JL_GC_PUSH1(&string_jl);
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense();
  CHECK_LE(key_start_, 0);
  if (is_dense) {
      size_t offset = key_begin - key_start_;
    for (size_t i = 0; i < num_elements; i++) {
      string_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(buff), i);
      const char* c_str = jl_string_ptr(string_jl);
      values_[offset + i] = std::string(c_str);
    }
  } else {
    for (size_t i = 0; i < num_elements; i++) {
      string_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(buff), i);
      const char* c_str = jl_string_ptr(string_jl);
      sparse_index_[key_begin + i] = std::string(c_str);
    }
  }
  JL_GC_POP();
}

void
DistArrayPartition<std::string>::BuildIndex() {
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense();
  if (is_dense) {
    BuildDenseIndex();
  } else {
    BuildSparseIndex();
  }
}

void
DistArrayPartition<std::string>::BuildDenseIndex() {
  if (keys_.size() == 0) return;
  int64_t min_key = keys_[0];
  CHECK(values_.size() == keys_.size());
  for (auto key : keys_) {
    min_key = std::min(key, min_key);
  }
  key_start_ = min_key;
  std::vector<int64_t> perm(keys_.size());
  std::vector<std::string> values_temp(values_);

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

void
DistArrayPartition<std::string>::BuildSparseIndex() {
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    const auto& value = values_[i];
    sparse_index_[key] = value;
  }
  keys_.clear();
  values_.clear();
  sparse_index_exists_ = true;
}

void
DistArrayPartition<std::string>::Clear() {
  keys_.clear();
  values_.clear();
  sparse_index_.clear();
  sparse_index_exists_ = false;
}

void
DistArrayPartition<std::string>::AppendKeyValue(int64_t key,
                                                const std::string &value) {
  keys_.push_back(key);
  values_.push_back(value);
}

void
DistArrayPartition<std::string>::AppendValue(const std::string &value) {
  values_.push_back(value);
}

void
DistArrayPartition<std::string>::Repartition(
    const int32_t *repartition_ids) {
  auto &dist_array_meta = dist_array_->GetMeta();
  auto partition_scheme = dist_array_meta.GetPartitionScheme();
  if (partition_scheme == DistArrayPartitionScheme::kSpaceTime) {
    RepartitionSpaceTime(repartition_ids);
  } else {
    Repartition1D(repartition_ids);
  }
}

void
DistArrayPartition<std::string>::RepartitionSpaceTime(
    const int32_t *repartition_ids) {
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    const std::string& value = values_[i];
    int32_t space_partition_id = repartition_ids[i * 2];
    int32_t time_partition_id = repartition_ids[i * 2 + 1];
    auto new_partition_pair = dist_array_->GetAndCreateLocalPartition(space_partition_id,
                                            time_partition_id);
    auto *partition_to_add = dynamic_cast<DistArrayPartition<std::string>*>(
        new_partition_pair.first);
    partition_to_add->AppendKeyValue(key, value);
  }
}

void
DistArrayPartition<std::string>::Repartition1D(
    const int32_t *repartition_ids) {
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    const std::string& value = values_[i];
    int32_t repartition_id = repartition_ids[i];
    auto new_partition_pair =
        dist_array_->GetAndCreateLocalPartition(repartition_id);
    auto *partition_to_add = dynamic_cast<DistArrayPartition<std::string>*>(
        new_partition_pair.first);
    partition_to_add->AppendKeyValue(key, value);
  }
}

SendDataBuffer
DistArrayPartition<std::string>::Serialize() {
  size_t num_bytes = sizeof(size_t)
                     + keys_.size() * sizeof(int64_t);
  for (const auto &str : values_) {
    num_bytes += str.size() + 1;
  }

  uint8_t* buff = new uint8_t[num_bytes];
  uint8_t* cursor = buff;
  *(reinterpret_cast<size_t*>(cursor)) = keys_.size();
  cursor += sizeof(size_t);
  memcpy(cursor, keys_.data(), keys_.size() * sizeof(int64_t));
  cursor += sizeof(int64_t) * keys_.size();
  for (const auto &str : values_) {
    memcpy(cursor, str.c_str(), str.size() + 1);
    cursor += str.size() + 1;
  }
  return std::make_pair(buff, num_bytes);
}

const uint8_t*
DistArrayPartition<std::string>::Deserialize(const uint8_t *buffer) {
  const uint8_t* cursor = buffer;
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);
  keys_.resize(num_keys);
  values_.resize(num_keys);

  memcpy(keys_.data(), cursor, num_keys * sizeof(int64_t));
  cursor += sizeof(int64_t) * num_keys;
  for (size_t i = 0; i < num_keys; i++) {
    values_[i] = std::string(reinterpret_cast<const char*>(cursor));
    cursor += values_[i].size() + 1;
  }
  return cursor;
}

const uint8_t*
DistArrayPartition<std::string>::DeserializeAndAppend(const uint8_t *buffer) {

  const uint8_t* cursor = buffer;
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);
  size_t orig_num_keys = keys_.size();
  keys_.resize(orig_num_keys + num_keys);

  memcpy(keys_.data() + orig_num_keys, cursor,
         num_keys * sizeof(int64_t));
  cursor += sizeof(int64_t) * num_keys;
  for (size_t i = orig_num_keys; i < orig_num_keys + num_keys; i++) {
    values_[i] = std::string(reinterpret_cast<const char*>(cursor));
    cursor += values_[i].size() + 1;
  }
  return cursor;
}

void
DistArrayPartition<std::string>::GetJuliaValueArray(jl_value_t **value) {
  jl_value_t* value_array_type = nullptr;
  jl_value_t* string_jl = nullptr;
  JL_GC_PUSH2(&value_array_type, &string_jl);

  value_array_type = jl_apply_array_type(jl_string_type, 1);
  *value = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(value_array_type, values_.size()));
  for (size_t i = 0; i < values_.size(); i++) {
    string_jl = jl_cstr_to_string(values_[i].c_str());
    jl_arrayset(reinterpret_cast<jl_array_t*>(value), string_jl, i);
  }
  JL_GC_POP();
}

void
DistArrayPartition<std::string>::AppendJuliaValue(jl_value_t *value) {
  const char* c_str = jl_string_ptr(value);
  values_.emplace_back(c_str);
}

void
DistArrayPartition<std::string>::AppendJuliaValueArray(jl_value_t *value) {
  jl_value_t* string_jl = nullptr;
  JL_GC_PUSH1(&string_jl);
  size_t num_elements = jl_array_len(reinterpret_cast<jl_array_t*>(value));
  for (size_t i = 0; i < num_elements; i++) {
    string_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(value), i);
    const char* c_str = jl_string_ptr(string_jl);
    values_.emplace_back(c_str);
  }
  JL_GC_POP();
}
}

}
