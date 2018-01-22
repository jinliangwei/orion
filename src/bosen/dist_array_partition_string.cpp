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
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl);
  bool is_dense = dist_array_meta.IsDense();

  value_type_jl = reinterpret_cast<jl_value_t*>(type::GetJlDataType(kValueType));
  values_array_type_jl = jl_apply_array_type(
      reinterpret_cast<jl_datatype_t*>(value_type_jl), 1);
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
    if (keys_.size() > 0) key_start_ = keys_[0];
    key_begin_jl = jl_box_int64(key_start_);
    jl_call3(create_accessor_func, dist_array_jl, key_begin_jl,
             values_array_jl);
  } else {
    keys_array_type_jl = jl_apply_array_type(jl_int64_type, 1);
    keys_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
        keys_array_type_jl,
        keys_.data(), keys_.size(), 0));
    key_begin_jl = jl_box_int64(key_start_);
    jl_call3(create_accessor_func, dist_array_jl, keys_array_jl,
             values_array_jl);
  }
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
}

void
DistArrayPartition<std::string>::ClearAccessor() {
  jl_value_t* string_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH3(&string_jl, &keys_array_jl, &values_array_jl);

  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense();
  const std::string &symbol = dist_array_meta.GetSymbol();
  jl_value_t *dist_array_jl = nullptr;
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl);

  auto *get_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
    "dist_array_get_values_vec");
  values_array_jl = jl_call1(get_values_vec_func, dist_array_jl);
  size_t num_values = jl_array_len(values_array_jl);
  values_.resize(num_values);
  for (size_t i = 0; i < num_values; i++) {
    string_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl), i);
    const char* c_str = jl_string_ptr(string_jl);
    values_[i] = std::string(c_str);
  }

  if (!is_dense) {
    auto *get_keys_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_keys_vec");
    keys_array_jl = jl_call1(get_keys_vec_func, dist_array_jl);
    auto *keys_vec = reinterpret_cast<int64_t*>(jl_array_data(keys_array_jl));
    size_t num_keys = jl_array_len(keys_array_jl);
    keys_.resize(num_keys);
    memcpy(keys_.data(), keys_vec, num_keys * sizeof(int64_t));
    sorted_ = false;
  }

  auto *delete_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "delete_dist_array_accessor");
  jl_call1(delete_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
}

void
DistArrayPartition<std::string>::CreateCacheAccessor() {
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
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl);

  value_type_jl = reinterpret_cast<jl_value_t*>(type::GetJlDataType(kValueType));
  values_array_type_jl = jl_apply_array_type(
      reinterpret_cast<jl_datatype_t*>(value_type_jl), 1);
  values_array_jl = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(
      values_array_type_jl, values_.size()));
  size_t i = 0;
  for (const auto &str : values_) {
    string_jl = jl_cstr_to_string(str.c_str());
    jl_arrayset(reinterpret_cast<jl_array_t*>(values_array_jl), string_jl, i);
    i++;
  }
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

void
DistArrayPartition<std::string>::CreateBufferAccessor() {
  jl_value_t *dist_array_jl = nullptr;
  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl);
  auto *create_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "create_dist_array_buffer_accessor");
  jl_call1(create_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
}

void
DistArrayPartition<std::string>::ClearCacheOrBufferAccessor() {
  jl_value_t* string_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH3(&string_jl, &keys_array_jl, &values_array_jl);

  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  jl_value_t *dist_array_jl = nullptr;
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl);

  auto *get_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
    "dist_array_get_values_vec");
  values_array_jl = jl_call1(get_values_vec_func, dist_array_jl);
  size_t num_values = jl_array_len(values_array_jl);
  values_.resize(num_values);
  for (size_t i = 0; i < num_values; i++) {
    string_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl), i);
    const char* c_str = jl_string_ptr(string_jl);
    values_[i] = std::string(c_str);
  }

  auto *get_keys_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
      "dist_array_get_keys_vec");
  keys_array_jl = jl_call1(get_keys_vec_func, dist_array_jl);
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
}

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
  Sort();
  if (keys_.size() > 0) key_start_ = keys_[0];
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
DistArrayPartition<std::string>::Sort(){
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

  for (size_t i = 0; i < keys_.size(); i++) {
    keys_[i] = min_key + i;
  }
  sorted_ = true;
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
  sorted_ = false;
}

void
DistArrayPartition<std::string>::AppendValue(const std::string &value) {
  values_.push_back(value);
  sorted_ = false;
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

const uint8_t*
DistArrayPartition<std::string>::Deserialize(const uint8_t *buffer) {
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
}

}
