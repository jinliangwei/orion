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
DistArrayPartition<std::string>::CreateAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 7);
  jl_value_t *&value_type_jl = jl_values[0];
  jl_value_t *&values_array_type_jl = jl_values[1];
  jl_value_t *&values_array_jl = jl_values[2];
  jl_value_t *&key_begin_jl = jl_values[3];
  jl_value_t *&keys_array_type_jl = jl_values[4];
  jl_value_t *&keys_array_jl = jl_values[5];
  jl_value_t *&string_jl = jl_values[6];

  jl_value_t *dist_array_jl = nullptr;
  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  JuliaEvaluator::GetVarJlValue(symbol, &dist_array_jl);
  bool is_dense = dist_array_meta.IsDense() && dist_array_meta.IsContiguousPartitions();
  value_type_jl = reinterpret_cast<jl_value_t*>(type::GetJlDataType(kValueType));
  values_array_type_jl = jl_apply_array_type(value_type_jl, 1);
  values_array_jl = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(
      values_array_type_jl, values_.size()));
  size_t i = 0;
  for (const auto &str : values_) {
    string_jl = jl_cstr_to_string(str.c_str());
    jl_arrayset(reinterpret_cast<jl_array_t*>(values_array_jl), string_jl, i);
    i++;
  }

  auto *create_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "create_dist_array_accessor");
  if (is_dense) {
    Sort();
    key_begin_jl = jl_box_int64(keys_.size() > 0 ? keys_[0] : 0);
    jl_call3(create_accessor_func, dist_array_jl, key_begin_jl,
             values_array_jl);
  } else {
    keys_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
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

void
DistArrayPartition<std::string>::ClearAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kAccessor);
  jl_value_t* string_jl = nullptr;
  jl_value_t* tuple_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH4(&string_jl, &tuple_jl, &keys_array_jl, &values_array_jl);

  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense() && dist_array_meta.IsContiguousPartitions();
  const std::string &symbol = dist_array_meta.GetSymbol();
  jl_value_t *dist_array_jl = nullptr;
  JuliaEvaluator::GetVarJlValue(symbol, &dist_array_jl);

  if (is_dense) {
    auto *get_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_values_vec");
    values_array_jl = jl_call1(get_values_vec_func, dist_array_jl);
    size_t num_values = jl_array_len(values_array_jl);
    values_.resize(num_values);
    for (size_t i = 0; i < num_values; i++) {
      string_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl), i);
      const char* c_str = jl_string_ptr(string_jl);
      values_[i] = std::string(c_str);
    }
  } else {
    auto *get_keys_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_keys_values_vec");
    tuple_jl = jl_call1(get_keys_values_vec_func, dist_array_jl);
    keys_array_jl = jl_get_nth_field(tuple_jl, 0);
    values_array_jl = jl_get_nth_field(tuple_jl, 1);
    auto *keys_vec = reinterpret_cast<int64_t*>(jl_array_data(keys_array_jl));
    size_t num_keys = jl_array_len(keys_array_jl);
    keys_.resize(num_keys);
    memcpy(keys_.data(), keys_vec, num_keys * sizeof(int64_t));

    size_t num_values = jl_array_len(values_array_jl);
    values_.resize(num_values);
    for (size_t i = 0; i < num_values; i++) {
      string_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl), i);
      const char* c_str = jl_string_ptr(string_jl);
      values_[i] = std::string(c_str);
    }
    sorted_ = false;
  }

  auto *delete_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "delete_dist_array_accessor");
  jl_call1(delete_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

void
DistArrayPartition<std::string>::CreateCacheAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 6);
  jl_value_t *&value_type_jl = jl_values[0];
  jl_value_t *&values_array_type_jl = jl_values[1];
  jl_value_t *&values_array_jl = jl_values[2];
  jl_value_t *&keys_array_type_jl = jl_values[3];
  jl_value_t *&keys_array_jl = jl_values[4];
  jl_value_t *&string_jl = jl_values[5];

  jl_value_t *dist_array_jl = nullptr;
  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  JuliaEvaluator::GetVarJlValue(symbol, &dist_array_jl);

  value_type_jl = reinterpret_cast<jl_value_t*>(type::GetJlDataType(kValueType));
  values_array_type_jl = jl_apply_array_type(value_type_jl, 1);
  values_array_jl = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(
      values_array_type_jl, values_.size()));
  size_t i = 0;
  for (const auto &str : values_) {
    string_jl = jl_cstr_to_string(str.c_str());
    jl_arrayset(reinterpret_cast<jl_array_t*>(values_array_jl), string_jl, i);
    i++;
  }
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

void
DistArrayPartition<std::string>::CreateBufferAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
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

void
DistArrayPartition<std::string>::ClearCacheAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kAccessor);
  jl_value_t* string_jl = nullptr;
  jl_value_t* tuple_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH4(&string_jl, &tuple_jl, &keys_array_jl, &values_array_jl);

  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  jl_value_t *dist_array_jl = nullptr;
  JuliaEvaluator::GetVarJlValue(symbol, &dist_array_jl);

  auto *get_keys_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
      "dist_array_get_accessor_keys_values_vec");
  tuple_jl = jl_call1(get_keys_values_vec_func, dist_array_jl);
  keys_array_jl = jl_get_nth_field(tuple_jl, 0);
  values_array_jl = jl_get_nth_field(tuple_jl, 1);

  size_t num_values = jl_array_len(values_array_jl);
  values_.resize(num_values);
  for (size_t i = 0; i < num_values; i++) {
    string_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl), i);
    const char* c_str = jl_string_ptr(string_jl);
    values_[i] = std::string(c_str);
  }

  auto *keys_vec = reinterpret_cast<int64_t*>(jl_array_data(keys_array_jl));
  size_t num_keys = jl_array_len(keys_array_jl);
  keys_.resize(num_keys);
  memcpy(keys_.data(), keys_vec, num_keys * sizeof(int64_t));

  auto *delete_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "delete_dist_array_accessor");
  jl_call1(delete_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  sorted_ = false;
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

void
DistArrayPartition<std::string>::ClearBufferAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kAccessor);
  jl_value_t* string_jl = nullptr;
  jl_value_t* tuple_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH4(&string_jl, &tuple_jl, &keys_array_jl, &values_array_jl);

  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  jl_value_t *dist_array_jl = nullptr;
  JuliaEvaluator::GetVarJlValue(symbol, &dist_array_jl);

  bool is_dense = dist_array_meta.IsDense();
  CHECK(dist_array_meta.IsContiguousPartitions());
  if (is_dense) {
    auto *get_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_values_vec");
    values_array_jl = jl_call1(get_values_vec_func, dist_array_jl);
    size_t num_values = jl_array_len(values_array_jl);
    values_.resize(num_values);
    for (size_t i = 0; i < num_values; i++) {
      string_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl), i);
      const char* c_str = jl_string_ptr(string_jl);
      values_[i] = std::string(c_str);
    }
    keys_.resize(num_values);
    for (size_t i = 0; i < num_values; i++) {
      keys_[i] = i;
    }
  } else {
    auto *get_keys_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_keys_values_vec");
    tuple_jl = jl_call1(get_keys_values_vec_func, dist_array_jl);
    keys_array_jl = jl_get_nth_field(tuple_jl, 0);
    values_array_jl = jl_get_nth_field(tuple_jl, 1);
    auto *keys_vec = reinterpret_cast<int64_t*>(jl_array_data(keys_array_jl));
    size_t num_keys = jl_array_len(keys_array_jl);
    keys_.resize(num_keys);
    memcpy(keys_.data(), keys_vec, num_keys * sizeof(int64_t));

    size_t num_values = jl_array_len(values_array_jl);
    values_.resize(num_values);
    for (size_t i = 0; i < num_values; i++) {
      string_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl), i);
      const char* c_str = jl_string_ptr(string_jl);
      values_[i] = std::string(c_str);
    }
    sorted_ = false;
  }

  auto *delete_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "delete_dist_array_accessor");
  jl_call1(delete_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  sorted_ = false;
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

void
DistArrayPartition<std::string>::BuildKeyValueBuffersFromSparseIndex() {
  if (storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer) return;
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  if (!keys_.empty()) return;
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
  sparse_index_.clear();
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

void
DistArrayPartition<std::string>::BuildDenseIndex() {
  Sort();
  if (keys_.size() > 0) key_start_ = keys_[0];
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
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
  storage_type_ = DistArrayPartitionStorageType::kSparseIndex;
}

void
DistArrayPartition<std::string>::GetAndSerializeValue(
    int64_t key, Blob *bytes_buff) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  auto iter = sparse_index_.find(key);
  CHECK(iter != sparse_index_.end());
  bytes_buff->resize(iter->second.size() + 1);
  memcpy(bytes_buff->data(),
         iter->second.c_str(),
         iter->second.size() + 1);
}

void
DistArrayPartition<std::string>::GetAndSerializeValues(const int64_t *keys,
                                                       size_t num_keys,
                                                       Blob *bytes_buff) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  size_t accum_size = 0;
  for (size_t i = 0; i < num_keys; i++) {
    auto key = keys[i];
    auto iter = sparse_index_.find(key);
    CHECK (iter != sparse_index_.end());
    accum_size += iter->second.size() + 1;
  }

  bytes_buff->resize(sizeof(bool) + sizeof(size_t)
                     + sizeof(int64_t) * num_keys + accum_size);

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
    memcpy(cursor, iter->second.c_str(), iter->second.size() + 1);
    cursor += iter->second.size() + 1;
  }
}

void
DistArrayPartition<std::string>::Sort() {
  if (sorted_) return;
  if (keys_.size() == 0) return;
  int64_t min_key = keys_[0];
  CHECK(values_.size() == keys_.size());
  for (auto key : keys_) {
    min_key = std::min(key, min_key);
  }
  std::vector<int64_t> perm(keys_.size());
  std::vector<std::string> values_temp(values_);

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

void
DistArrayPartition<std::string>::Clear() {
  CHECK(storage_type_ != DistArrayPartitionStorageType::kAccessor);
  keys_.clear();
  values_.clear();
  sparse_index_.clear();
  key_start_ = -1;
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

void
DistArrayPartition<std::string>::AppendKeyValue(int64_t key,
                                                const std::string &value) {
  keys_.push_back(key);
  values_.push_back(value);
  sorted_ = false;
}

void
DistArrayPartition<std::string>::AppendValue(const std::string &value) {
  values_.push_back(value);
  sorted_ = false;
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
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  size_t num_bytes = sizeof(bool) + sizeof(size_t)
                     + keys_.size() * sizeof(int64_t);
  for (const auto &str : values_) {
    num_bytes += str.size() + 1;
  }

  uint8_t* buff = new uint8_t[num_bytes];
  uint8_t* cursor = buff;
  *(reinterpret_cast<bool*>(cursor)) = sorted_;
  cursor += sizeof(bool);
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

void
DistArrayPartition<std::string>::HashSerialize(
    ExecutorDataBufferMap *data_buffer_map) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  std::unordered_map<int32_t, size_t> server_accum_size;
  std::unordered_map<int32_t, size_t> server_num_keys;
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    int32_t server_id = key % kConfig.kNumServers;
    auto &value = values_[i];
    auto iter = server_num_keys.find(server_id);
    if (iter == server_num_keys.end()) {
      server_accum_size[server_id] = sizeof(int64_t) + value.size() + 1;
      server_num_keys[server_id] = 1;
    } else {
      server_accum_size[server_id] += sizeof(int64_t) + value.size() + 1;
      server_num_keys[server_id] += 1;
    }
  }

  std::unordered_map<int32_t, uint8_t*> server_cursor;
  std::unordered_map<int32_t, uint8_t*> server_value_cursor;
  for (auto &accum_size_pair : server_accum_size) {
    int32_t server_id = accum_size_pair.first;
    size_t num_key_values = server_num_keys[server_id];
    accum_size_pair.second += sizeof(bool) + sizeof(size_t);

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
    auto& value = values_[i];
    memcpy(server_cursor[server_id], &key, sizeof(int64_t));
    server_cursor[server_id] += sizeof(int64_t);
    memcpy(server_value_cursor[server_id], value.c_str(), value.size() + 1);
    server_value_cursor[server_id] += value.size() + 1;
  }
}

const uint8_t*
DistArrayPartition<std::string>::Deserialize(const uint8_t *buffer) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  const uint8_t* cursor = buffer;
  sorted_ = *(reinterpret_cast<const bool*>(cursor));
  cursor += sizeof(bool);
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
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  sorted_ = false;
  const uint8_t* cursor = buffer;
  cursor += sizeof(bool);
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

const uint8_t*
DistArrayPartition<std::string>::DeserializeAndOverwrite(const uint8_t *buffer) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);

  const uint8_t* cursor = buffer;
  cursor += sizeof(bool);
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);
  const uint8_t* value_cursor = cursor + sizeof(int64_t) * num_keys;
  for (size_t i = 0; i < num_keys; i++) {
    auto key = *(reinterpret_cast<const int64_t*>(cursor));
    cursor += sizeof(int64_t);
    const char *str = reinterpret_cast<const char*>(value_cursor);
    sparse_index_[key] = std::string(str);
    value_cursor += sparse_index_[key].size() + 1;
  }
  return value_cursor;
}

void
DistArrayPartition<std::string>::GetJuliaValueArray(jl_value_t **value) {
  jl_value_t* value_array_type = nullptr;
  jl_value_t* string_jl = nullptr;
  JL_GC_PUSH2(&value_array_type, &string_jl);

  value_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_string_type), 1);
  *value = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(value_array_type, values_.size()));
  for (size_t i = 0; i < values_.size(); i++) {
    string_jl = jl_cstr_to_string(values_[i].c_str());
    jl_arrayset(reinterpret_cast<jl_array_t*>(value), string_jl, i);
  }
  JL_GC_POP();
}

void
DistArrayPartition<std::string>::GetJuliaValueArray(const std::vector<int64_t> &keys,
                                                  jl_value_t **value_array) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t* value_array_type = nullptr;
  jl_value_t* string_jl = nullptr;
  JL_GC_PUSH2(&value_array_type, &string_jl);

  value_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_string_type), 1);

  *value_array = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(
      value_array_type,
      keys.size()));

  for (size_t i = 0; i < keys.size(); i++) {
    auto key = keys[i];
    auto iter = sparse_index_.find(key);
    CHECK (iter != sparse_index_.end()) << " i = " << i
                                        << " key = " << key
                                        << " size = " << sparse_index_.size();
    auto value = iter->second;
    string_jl = jl_cstr_to_string(value.c_str());
    jl_arrayset(reinterpret_cast<jl_array_t*>(value_array), string_jl, i);
  }
  JL_GC_POP();
}

void
DistArrayPartition<std::string>::SetJuliaValues(const std::vector<int64_t> &keys,
                                                jl_value_t *values) {

  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t* string_jl = nullptr;
  JL_GC_PUSH1(&string_jl);
  for (size_t i = 0; i < keys.size(); i++) {
    auto key = keys[i];
    string_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values), i);
    const char* c_str = jl_string_ptr(string_jl);
    sparse_index_[key] = std::string(c_str);
  }

  JL_GC_POP();
}

void
DistArrayPartition<std::string>::AppendJuliaValue(jl_value_t *value) {
  const char* c_str = jl_string_ptr(value);
  values_.emplace_back(c_str);
  sorted_ = false;
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
  sorted_ = false;
}

void
DistArrayPartition<std::string>::ShrinkValueVecToFit() {
}

}

}
