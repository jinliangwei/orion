#include <orion/bosen/dist_array_partition.hpp>
#include <orion/bosen/julia_module.hpp>
#include <sstream>

namespace orion {
namespace bosen {
/*---- template const char* implementation -----*/
DistArrayPartition<void>::DistArrayPartition(
    DistArray *dist_array,
    const Config &config,
    type::PrimitiveType value_type,
    JuliaThreadRequester *julia_requester):
    AbstractDistArrayPartition(dist_array, config, value_type, julia_requester) {
  std::stringstream ss;
  ss << static_cast<const void*>(this);
  ptr_str_ = ss.str();
  JuliaEvaluator::CreateDistArrayPartition(dist_array_, ptr_str_);
}


DistArrayPartition<void>::~DistArrayPartition() {
  JuliaEvaluator::DeleteDistArrayPartition(dist_array_, ptr_str_);
}

void
DistArrayPartition<void>::CreateAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t *key_begin_jl = nullptr;
  jl_value_t *keys_array_type_jl = nullptr;
  jl_value_t *keys_array_jl = nullptr;
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH4(&key_begin_jl, &keys_array_type_jl,
              &keys_array_jl, &values_array_jl);

  jl_value_t *dist_array_jl = dist_array_->GetJuliaDistArray();
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense() && dist_array_meta.IsContiguousPartitions();
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
DistArrayPartition<void>::ClearAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kAccessor);
  jl_value_t* tuple_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH3(&tuple_jl, &keys_array_jl, &values_array_jl);

  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense() && dist_array_meta.IsContiguousPartitions();
  jl_value_t *dist_array_jl = dist_array_->GetJuliaDistArray();

  if (is_dense) {
    auto *get_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_values_vec");
    values_array_jl = jl_call1(get_values_vec_func, dist_array_jl);
    JuliaEvaluator::SetDistArrayPartition(dist_array_, ptr_str_, values_array_jl);
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

    JuliaEvaluator::SetDistArrayPartition(dist_array_, ptr_str_, values_array_jl);
  }

  auto *delete_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "delete_dist_array_accessor");
  jl_call1(delete_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
  LOG(INFO) << __func__ << " dist_array_id = " << dist_array_->kId
            << " done!";
}

void
DistArrayPartition<void>::CreateCacheAccessor() {
  LOG(INFO) << __func__;
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t *key_begin_jl = nullptr;
  jl_value_t *keys_array_type_jl = nullptr;
  jl_value_t *keys_array_jl = nullptr;
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH4(&key_begin_jl, &keys_array_type_jl,
              &keys_array_jl, &values_array_jl);

  jl_value_t *dist_array_jl = dist_array_->GetJuliaDistArray();
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
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
DistArrayPartition<void>::CreateBufferAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t *dist_array_jl = dist_array_->GetJuliaDistArray();
  auto *create_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "create_dist_array_buffer_accessor");
  jl_call1(create_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  storage_type_ = DistArrayPartitionStorageType::kAccessor;
}

void
DistArrayPartition<void>::ClearCacheAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kAccessor);
  LOG(INFO) << __func__ << " dist_array_id = " << dist_array_->kId;
  jl_value_t* tuple_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH3(&tuple_jl, &keys_array_jl, &values_array_jl);

  jl_value_t *dist_array_jl = dist_array_->GetJuliaDistArray();

  auto *get_keys_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
      "dist_array_get_accessor_keys_values_vec");
  tuple_jl = jl_call1(get_keys_values_vec_func, dist_array_jl);
  keys_array_jl = jl_get_nth_field(tuple_jl, 0);
  values_array_jl = jl_get_nth_field(tuple_jl, 1);
  JuliaEvaluator::SetDistArrayPartition(dist_array_, ptr_str_, values_array_jl);

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
DistArrayPartition<void>::ClearBufferAccessor() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kAccessor);
  LOG(INFO) << __func__ << " dist_array_id = " << dist_array_->kId;
  jl_value_t* tuple_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH3(&tuple_jl, &keys_array_jl, &values_array_jl);

  auto &dist_array_meta = dist_array_->GetMeta();
  jl_value_t *dist_array_jl = dist_array_->GetJuliaDistArray();
  bool is_dense = dist_array_meta.IsDense();
  CHECK(dist_array_meta.IsContiguousPartitions());
  if (is_dense) {
    auto *get_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_values_vec");
    values_array_jl = jl_call1(get_values_vec_func, dist_array_jl);
    JuliaEvaluator::SetDistArrayPartition(dist_array_, ptr_str_, values_array_jl);
    size_t num_values = jl_array_len(values_array_jl);
    keys_.resize(num_values);
    for (size_t i = 0; i < num_values; i++) {
      keys_[i] = i;
    }
  } else {
    auto *get_keys_values_vec_func = JuliaEvaluator::GetOrionWorkerFunction(
        "dist_array_get_accessor_keys_values_vec");
    tuple_jl = jl_call1(get_keys_values_vec_func, dist_array_jl);
    JuliaEvaluator::AbortIfException();
    keys_array_jl = jl_get_nth_field(tuple_jl, 0);
    values_array_jl = jl_get_nth_field(tuple_jl, 1);

    auto *keys_vec = reinterpret_cast<int64_t*>(jl_array_data(keys_array_jl));
    size_t num_keys = jl_array_len(keys_array_jl);
    keys_.resize(num_keys);
    memcpy(keys_.data(), keys_vec, num_keys * sizeof(int64_t));
    JuliaEvaluator::SetDistArrayPartition(dist_array_, ptr_str_, values_array_jl);
  }
  auto *delete_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "delete_dist_array_accessor");
  jl_call1(delete_accessor_func, dist_array_jl);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  sorted_ = false;
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
  LOG(INFO) << __func__ << " dist_array_id = " << dist_array_->kId
            << " done!";
}

void
DistArrayPartition<void>::BuildKeyValueBuffersFromSparseIndex() {
  if (storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer) return;
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  if (!keys_.empty()) return;
  keys_.resize(sparse_index_.size());
  auto iter = sparse_index_.begin();
  size_t i = 0;
  for (; iter != sparse_index_.end(); iter++) {
    int64_t key = iter->first;
    keys_[i++] = key;
  }
  sorted_ = false;
  sparse_index_.clear();
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

void
DistArrayPartition<void>::BuildDenseIndex() {
  Sort();
  if (keys_.size() > 0) key_start_ = keys_[0];
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

void
DistArrayPartition<void>::BuildSparseIndex() {
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    sparse_index_[key] = i;
  }
  keys_.clear();
  storage_type_ = DistArrayPartitionStorageType::kSparseIndex;
}

void
DistArrayPartition<void>::GetAndSerializeValue(
    int64_t key, Blob *bytes_buff) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t *buff = nullptr;
  jl_value_t *serialized_result_array = nullptr;
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH3(&buff, &serialized_result_array,
              &values_array_jl);
  jl_function_t *io_buffer_func
      = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");
  jl_function_t *serialize_func
      = JuliaEvaluator::GetFunction(jl_base_module, "serialize");
  jl_function_t *takebuff_array_func
      = JuliaEvaluator::GetFunction(jl_base_module, "take!");

  auto iter = sparse_index_.find(key);
  CHECK(iter != sparse_index_.end());
  size_t offset = iter->second;
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
  jl_value_t *value = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl),
                                  offset);
  buff = jl_call0(io_buffer_func);
  jl_call2(serialize_func, buff, value);
  serialized_result_array = jl_call1(takebuff_array_func, buff);
  size_t result_array_length = jl_array_len(serialized_result_array);
  uint8_t* array_bytes = reinterpret_cast<uint8_t*>(jl_array_data(serialized_result_array));
  bytes_buff->resize(result_array_length);
  memcpy(bytes_buff->data(), array_bytes, result_array_length);
  JL_GC_POP();
}

void
DistArrayPartition<void>::GetAndSerializeValues(
    const int64_t *keys,
    size_t num_keys,
    Blob *bytes_buff) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t *buff = nullptr;
  jl_value_t *serialized_result_array = nullptr;
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH3(&buff, &serialized_result_array,
              &values_array_jl);
  std::vector<Blob> value_buff_vec(num_keys);
  size_t accum_size = 0;
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
  jl_function_t *io_buffer_func
      = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");
  jl_function_t *serialize_func
      = JuliaEvaluator::GetFunction(jl_base_module, "serialize");
  jl_function_t *takebuff_array_func
      = JuliaEvaluator::GetFunction(jl_base_module, "take!");
  buff = jl_call0(io_buffer_func);

  for (size_t i = 0; i < num_keys; i++) {
    auto key = keys[i];
    auto iter = sparse_index_.find(key);
    CHECK(iter != sparse_index_.end());
    size_t offset = iter->second;
    jl_value_t *value = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl),
                                    offset);
    jl_call2(serialize_func, buff, value);
    serialized_result_array = jl_call1(takebuff_array_func, buff);

    auto &value_buff = value_buff_vec[i];
    size_t result_array_length = jl_array_len(serialized_result_array);
    uint8_t* array_bytes = reinterpret_cast<uint8_t*>(jl_array_data(serialized_result_array));
    accum_size += sizeof(size_t) + result_array_length;
    value_buff.resize(result_array_length);
    memcpy(value_buff.data(), array_bytes, result_array_length);
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

  for (auto &value_buff : value_buff_vec) {
    *reinterpret_cast<size_t*>(cursor) = value_buff.size();
    cursor += sizeof(size_t);
    memcpy(cursor, value_buff.data(), value_buff.size());
    cursor += value_buff.size();
  }
  JL_GC_POP();
}

void
DistArrayPartition<void>::Sort() {
  if (sorted_) return;
  if (keys_.size() == 0) return;
  int64_t min_key = keys_[0];
  for (auto key : keys_) {
    min_key = std::min(key, min_key);
  }
  key_start_ = min_key;
  std::vector<int64_t> perm(keys_.size());
  std::vector<size_t> julia_index(keys_.size());
  for (size_t i = 0; i < keys_.size(); i++) {
    julia_index[i] = i;
  }

  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&] (const size_t &i, const size_t &j) {
              return keys_[i] < keys_[j];
            });
  std::transform(perm.begin(), perm.end(), julia_index.begin(),
                 [&](size_t i) { return julia_index[i]; });

  auto &meta = dist_array_->GetMeta();
  if (meta.IsContiguousPartitions()) {
    for (size_t i = 0; i < keys_.size(); i++) {
      keys_[i] = min_key + i;
    }
  } else {
    std::sort(keys_.begin(), keys_.end());
  }

  jl_value_t* value_type = nullptr;
  jl_value_t* value_array_type = nullptr;
  jl_value_t *value_jl = nullptr;
  jl_value_t *new_values_array_jl = nullptr;
  jl_value_t *old_values_array_jl = nullptr;
  JL_GC_PUSH5(&value_type, &value_array_type, &value_jl, &new_values_array_jl,
              &old_values_array_jl);
  jl_value_t *dist_array_jl = dist_array_->GetJuliaDistArray();
  JuliaEvaluator::GetDistArrayValueType(dist_array_jl,
                                        reinterpret_cast<jl_datatype_t**>(&value_type));
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &old_values_array_jl);
  value_array_type = jl_apply_array_type(value_type, 1);
  new_values_array_jl = reinterpret_cast<jl_value_t*>(
      jl_alloc_array_1d(value_array_type, keys_.size()));

  for (size_t i = 0; i < keys_.size(); i++) {
    size_t index = julia_index[i];
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(old_values_array_jl), index);
    jl_arrayset(reinterpret_cast<jl_array_t*>(new_values_array_jl), value_jl, i);
  }
  JuliaEvaluator::SetDistArrayPartition(dist_array_, ptr_str_, new_values_array_jl);
  JL_GC_POP();
  sorted_ = true;
}

void
DistArrayPartition<void>::Repartition(
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
DistArrayPartition<void>::RepartitionSpaceTime(
    const int32_t *repartition_ids) {
  jl_value_t *value_jl = nullptr;
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH2(&value_jl, &values_array_jl);
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl), i);
    int32_t space_partition_id = repartition_ids[i * 2];
    int32_t time_partition_id = repartition_ids[i * 2 + 1];
    auto new_partition_pair = dist_array_->GetAndCreateLocalPartition(space_partition_id,
                                                                      time_partition_id);
    auto *partition_to_add = dynamic_cast<DistArrayPartition<void>*>(new_partition_pair.first);
    partition_to_add->AppendKeyValue(key, value_jl);
  }
  JL_GC_POP();
}

void
DistArrayPartition<void>::Repartition1D(
    const int32_t *repartition_ids) {
  jl_value_t *value_jl = nullptr;
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH2(&value_jl, &values_array_jl);
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl), i);
    int32_t repartition_id = repartition_ids[i];
    auto new_partition_pair = dist_array_->GetAndCreateLocalPartition(repartition_id);
    auto *partition_to_add = dynamic_cast<DistArrayPartition<void>*>(new_partition_pair.first);
    partition_to_add->AppendKeyValue(key, value_jl);
  }
  JL_GC_POP();
}

SendDataBuffer
DistArrayPartition<void>::Serialize() {
  jl_value_t* buff_jl = nullptr;
  jl_value_t* serialized_value_array = nullptr;
  jl_value_t* value_jl = nullptr;
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH4(&buff_jl, &serialized_value_array, &value_jl,
              &values_array_jl);
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
  jl_function_t *io_buffer_func
      = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");
  buff_jl = jl_call0(io_buffer_func);
  jl_function_t *serialize_func
      = JuliaEvaluator::GetFunction(jl_base_module, "serialize");

  size_t num_bytes = sizeof(bool) + sizeof(size_t) + keys_.size() * sizeof(int64_t);
  std::vector<Blob> serialized_values(keys_.size());
  size_t num_values = jl_array_len(values_array_jl);
  for (size_t i = 0; i < num_values; i++) {
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl), i);
    jl_call2(serialize_func, buff_jl, value_jl);
    jl_function_t *takebuff_array_func
        = JuliaEvaluator::GetFunction(jl_base_module, "take!");
    serialized_value_array = jl_call1(takebuff_array_func, buff_jl);
    size_t result_array_length = jl_array_len(serialized_value_array);
    num_bytes += result_array_length + sizeof(size_t);
    uint8_t* array_bytes = reinterpret_cast<uint8_t*>(jl_array_data(serialized_value_array));
    serialized_values[i].resize(result_array_length);
    memcpy(serialized_values[i].data(), array_bytes, result_array_length);
  }

  uint8_t* buff = new uint8_t[num_bytes];
  uint8_t* cursor = buff;
  *(reinterpret_cast<bool*>(cursor)) = sorted_;
  cursor += sizeof(bool);
  *(reinterpret_cast<size_t*>(cursor)) = keys_.size();
  cursor += sizeof(size_t);
  memcpy(cursor, keys_.data(), keys_.size() * sizeof(int64_t));
  cursor += sizeof(int64_t) * keys_.size();
  for (const auto &serialized_value : serialized_values) {
    *reinterpret_cast<size_t*>(cursor) = serialized_value.size();
    cursor += sizeof(size_t);
    memcpy(cursor, serialized_value.data(), serialized_value.size());
    cursor += serialized_value.size();
  }
  JL_GC_POP();
  return std::make_pair(buff, num_bytes);
}

void
DistArrayPartition<void>::HashSerialize(
    ExecutorDataBufferMap *data_buffer_map) {
  jl_value_t* buff_jl = nullptr;
  jl_value_t* serialized_value_array = nullptr;
  jl_value_t* value_jl = nullptr;
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH4(&buff_jl, &serialized_value_array, &value_jl,
              &values_array_jl);
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);

  jl_function_t *io_buffer_func
      = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");
  buff_jl = jl_call0(io_buffer_func);
  jl_function_t *serialize_func
      = JuliaEvaluator::GetFunction(jl_base_module, "serialize");

  size_t num_values = jl_array_len(values_array_jl);
  CHECK_EQ(keys_.size(), num_values);
  std::vector<Blob> serialized_values(num_values);
  for (size_t i = 0; i < num_values; i++) {
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl), i);
    jl_call2(serialize_func, buff_jl, value_jl);
    JuliaEvaluator::AbortIfException();
    jl_function_t *takebuff_array_func
        = JuliaEvaluator::GetFunction(jl_base_module, "take!");
    serialized_value_array = jl_call1(takebuff_array_func, buff_jl);
    JuliaEvaluator::AbortIfException();
    size_t result_array_length = jl_array_len(serialized_value_array);
    uint8_t* array_bytes = reinterpret_cast<uint8_t*>(jl_array_data(serialized_value_array));
    serialized_values[i].resize(result_array_length);
    memcpy(serialized_values[i].data(), array_bytes, result_array_length);
  }

  std::unordered_map<int32_t, size_t> server_accum_size;
  std::unordered_map<int32_t, size_t> server_num_keys;
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    int32_t server_id = key % kConfig.kNumServers;
    auto &value = serialized_values[i];
    auto iter = server_num_keys.find(server_id);
    if (iter == server_num_keys.end()) {
      server_accum_size[server_id] = sizeof(int64_t) + sizeof(size_t) + value.size();
      server_num_keys[server_id] = 1;
    } else {
      server_accum_size[server_id] += sizeof(int64_t) + sizeof(size_t) + value.size();
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
    auto& serialized_value = serialized_values[i];
    *reinterpret_cast<int64_t*>(server_cursor[server_id]) = key;
    server_cursor[server_id] += sizeof(int64_t);
    *reinterpret_cast<size_t*>(server_value_cursor[server_id]) = serialized_value.size();
    server_value_cursor[server_id] += sizeof(size_t);
    memcpy(server_value_cursor[server_id], serialized_value.data(), serialized_value.size());
    server_value_cursor[server_id] += serialized_value.size();
  }
  JL_GC_POP();
}

const uint8_t*
DistArrayPartition<void>::Deserialize(const uint8_t *buffer) {
  jl_value_t* buff_jl = nullptr;
  jl_value_t* serialized_value_array = nullptr;
  jl_value_t* value_jl = nullptr;
  jl_value_t* serialized_value_array_type = nullptr;
  jl_value_t* uint64_jl = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH6(&buff_jl, &serialized_value_array,
              &value_jl, &serialized_value_array_type,
              &uint64_jl, &values_array_jl);
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
  serialized_value_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_uint8_type), 1);
  jl_function_t *io_buffer_func = JuliaEvaluator::GetFunction(
      jl_base_module, "IOBuffer");
  buff_jl = jl_call0(io_buffer_func);
  jl_function_t *deserialize_func = JuliaEvaluator::GetFunction(
      jl_base_module, "deserialize");
  jl_function_t *resize_vec_func = JuliaEvaluator::GetFunction(jl_base_module,
                                                               "resize!");
  const uint8_t* cursor = buffer;
  sorted_ = *(reinterpret_cast<const bool*>(cursor));
  cursor += sizeof(bool);
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  uint64_jl = jl_box_uint64(num_keys);

  jl_call2(resize_vec_func, values_array_jl, uint64_jl);
  cursor += sizeof(size_t);
  keys_.resize(num_keys);

  memcpy(keys_.data(), cursor, num_keys * sizeof(int64_t));
  cursor += sizeof(int64_t) * num_keys;
  for (size_t i = 0; i < num_keys; i++) {
    size_t serialized_value_size = *reinterpret_cast<const size_t*>(cursor);
    cursor += sizeof(size_t);
    std::vector<uint8_t> temp(serialized_value_size);
    memcpy(temp.data(), cursor, serialized_value_size);
    serialized_value_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
        serialized_value_array_type,
        temp.data(),
        serialized_value_size, 0));
    buff_jl = jl_call1(io_buffer_func, serialized_value_array);
    JuliaEvaluator::AbortIfException();
    value_jl = jl_call1(deserialize_func, buff_jl);
    JuliaEvaluator::AbortIfException();
    jl_arrayset(reinterpret_cast<jl_array_t*>(values_array_jl), value_jl, i);
    cursor += serialized_value_size;
  }
  JL_GC_POP();
  return cursor;
}

const uint8_t*
DistArrayPartition<void>::DeserializeAndAppend(const uint8_t *buffer) {
  sorted_ = false;
  jl_value_t* buff_jl = nullptr;
  jl_value_t* serialized_value_array = nullptr;
  jl_value_t* value_jl = nullptr;
  jl_value_t* serialized_value_array_type = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH5(&buff_jl, &serialized_value_array, &value_jl,
              &serialized_value_array_type, &values_array_jl);
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
  serialized_value_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_uint8_type), 1);
  jl_function_t *io_buffer_func
      = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");
  buff_jl = jl_call0(io_buffer_func);
  jl_function_t *deserialize_func
      = JuliaEvaluator::GetFunction(jl_base_module, "deserialize");

  const uint8_t* cursor = buffer;
  cursor += sizeof(bool);
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);

  size_t orig_num_keys = keys_.size();
  keys_.resize(orig_num_keys + num_keys);
  memcpy(keys_.data() + orig_num_keys, cursor, num_keys * sizeof(int64_t));
  cursor += sizeof(int64_t) * num_keys;
  for (size_t i = 0; i < num_keys; i++) {
    size_t serialized_value_size = *reinterpret_cast<const size_t*>(cursor);
    cursor += sizeof(size_t);
    std::vector<uint8_t> temp(serialized_value_size);
    memcpy(temp.data(), cursor, serialized_value_size);
    cursor += serialized_value_size;
    serialized_value_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
        serialized_value_array_type,
        temp.data(),
        serialized_value_size, 0));
    buff_jl = jl_call1(io_buffer_func, serialized_value_array);
    value_jl = jl_call1(deserialize_func, buff_jl);
    jl_array_ptr_1d_push(reinterpret_cast<jl_array_t*>(values_array_jl), value_jl);
  }
  JL_GC_POP();
  return cursor;
}

const uint8_t*
DistArrayPartition<void>::DeserializeAndOverwrite(
    const uint8_t *buffer) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t* buff_jl = nullptr;
  jl_value_t* serialized_value_array = nullptr;
  jl_value_t* value_jl = nullptr;
  jl_value_t* serialized_value_array_type = nullptr;
  jl_value_t* values_array_jl = nullptr;
  JL_GC_PUSH5(&buff_jl, &serialized_value_array, &value_jl,
              &serialized_value_array_type, &values_array_jl);
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);

  serialized_value_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_uint8_type), 1);
  jl_function_t *io_buffer_func
      = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");
  buff_jl = jl_call0(io_buffer_func);
  jl_function_t *deserialize_func
      = JuliaEvaluator::GetFunction(jl_base_module, "deserialize");

  const uint8_t* cursor = buffer;
  cursor += sizeof(bool);
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);

  const uint8_t* value_cursor = cursor + sizeof(int64_t) * num_keys;
  for (size_t i = 0; i < num_keys; i++) {
    auto key = *(reinterpret_cast<const int64_t*>(cursor));
    cursor += sizeof(int64_t);
    size_t serialized_value_size = *reinterpret_cast<const size_t*>(value_cursor);
    value_cursor += sizeof(size_t);
    std::vector<uint8_t> temp(serialized_value_size);
    memcpy(temp.data(), value_cursor, serialized_value_size);
    serialized_value_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
        serialized_value_array_type,
        temp.data(),
        serialized_value_size, 0));
    buff_jl = jl_call1(io_buffer_func, serialized_value_array);
    value_jl = jl_call1(deserialize_func, buff_jl);

    auto iter = sparse_index_.find(key);
    if (iter == sparse_index_.end()) {
      jl_array_ptr_1d_push(reinterpret_cast<jl_array_t*>(values_array_jl), value_jl);
      sparse_index_[key] = jl_array_len(reinterpret_cast<jl_array_t*>(values_array_jl)) - 1;
    } else {
      auto index = iter->second;
      jl_arrayset(reinterpret_cast<jl_array_t*>(values_array_jl), value_jl, index);
    }
  }
  return value_cursor;
}

void
DistArrayPartition<void>::Clear() {
  CHECK(storage_type_ != DistArrayPartitionStorageType::kAccessor);
  keys_.clear();
  sparse_index_.clear();
  key_start_ = -1;
  JuliaEvaluator::ClearDistArrayPartition(dist_array_, ptr_str_);
  JuliaEvaluator::AbortIfException();
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

void
DistArrayPartition<void>::GetJuliaValueArray(jl_value_t **value) {
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, value);
}

void
DistArrayPartition<void>::GetJuliaValueArray(const std::vector<int64_t> &keys,
                                             jl_value_t **value_array) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t* value_type = nullptr;
  jl_value_t* value_array_type = nullptr;
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH3(&value_type, &value_array_type, &values_array_jl);

  jl_value_t *dist_array_jl = dist_array_->GetJuliaDistArray();
  JuliaEvaluator::GetDistArrayValueType(dist_array_jl,
                                        reinterpret_cast<jl_datatype_t**>(&value_type));
  value_array_type = jl_apply_array_type(value_type, 1);
  *value_array = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(
      value_array_type,
      keys.size()));
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);

  for (size_t i = 0; i < keys.size(); i++) {
    auto key = keys[i];
    auto iter = sparse_index_.find(key);
    CHECK(iter != sparse_index_.end());
    size_t offset = iter->second;
    jl_value_t *value = jl_arrayref(reinterpret_cast<jl_array_t*>(values_array_jl),
                                    offset);
    jl_arrayset(reinterpret_cast<jl_array_t*>(*value_array), value, i);
  }
  JL_GC_POP();
}

void
DistArrayPartition<void>::SetJuliaValues(const std::vector<int64_t> &keys,
                                         jl_value_t *values) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH1(&values_array_jl);
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
  for (size_t i = 0; i < keys.size(); i++) {
    auto key = keys[i];
    auto iter = sparse_index_.find(key);
    CHECK(iter != sparse_index_.end());
    size_t offset = iter->second;
    jl_value_t *value = jl_arrayref(reinterpret_cast<jl_array_t*>(values),
                                    i);
    jl_arrayset(reinterpret_cast<jl_array_t*>(values_array_jl), value, offset);
  }
  JL_GC_POP();
}

void
DistArrayPartition<void>::AppendJuliaValue(jl_value_t *value) {
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH1(&values_array_jl);
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
  jl_array_ptr_1d_push(reinterpret_cast<jl_array_t*>(values_array_jl), value);
  sorted_ = false;
  JL_GC_POP();
}

void
DistArrayPartition<void>::AppendJuliaValueArray(jl_value_t *value) {
  jl_value_t* value_jl = nullptr;
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH2(&values_array_jl, &value_jl);
  JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
  size_t num_elements = jl_array_len(reinterpret_cast<jl_array_t*>(value));
  for (size_t i = 0; i < num_elements; i++) {
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(value), i);
    jl_array_ptr_1d_push(reinterpret_cast<jl_array_t*>(values_array_jl), value_jl);
  }
  JL_GC_POP();
  sorted_ = false;
}

}
}
