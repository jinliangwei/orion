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
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer)
      << " storage_type = " << static_cast<int>(storage_type_)
      << " dist_array_id = " << dist_array_->kId;
  jl_value_t *key_begin_jl = nullptr;
  jl_value_t *keys_array_type_jl = nullptr;
  jl_value_t *keys_array_jl = nullptr;
  jl_value_t *values_array_jl = nullptr;
  JL_GC_PUSH4(&key_begin_jl, &keys_array_type_jl,
              &keys_array_jl, &values_array_jl);

  jl_value_t *dist_array_jl = dist_array_->GetJuliaDistArray();
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense() && dist_array_meta.IsContiguousPartitions();
  auto *create_accessor_func = JuliaEvaluator::GetOrionWorkerFunction(
      "create_dist_array_accessor");
  if (is_dense) {
    Sort();
    key_begin_jl = jl_box_int64(keys_.size() > 0 ? keys_[0] : 0);
    JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
    jl_call3(create_accessor_func, dist_array_jl, key_begin_jl,
             values_array_jl);
  } else {
    keys_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
    keys_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
        keys_array_type_jl,
        keys_.data(), keys_.size(), 0));
    JuliaEvaluator::GetDistArrayPartition(dist_array_, ptr_str_, &values_array_jl);
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
}

void
DistArrayPartition<void>::CreateCacheAccessor() {
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
  CHECK(storage_type_ == DistArrayPartitionStorageType::kAccessor)
      << " dist_array_id = " << dist_array_->kId;
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
}

void
DistArrayPartition<void>::BuildKeyValueBuffersFromSparseIndex() {
  if (storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer) return;
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  CHECK(keys_.empty());
  //if (!keys_.empty()) return;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* ptr_str_jl = nullptr;
  JL_GC_PUSH2(&keys_array_jl, &ptr_str_jl);

  jl_value_t* dist_array_jl =dist_array_->GetJuliaDistArray();
  ptr_str_jl = jl_cstr_to_string(ptr_str_.c_str());
  jl_function_t *convert_func
      = JuliaEvaluator::GetOrionWorkerFunction("dist_array_partition_sparse_to_dense");
  keys_array_jl = jl_call2(convert_func, dist_array_jl, ptr_str_jl);
  JuliaEvaluator::AbortIfException();
  size_t num_keys = jl_array_len(keys_array_jl);
  uint8_t* key_bytes = reinterpret_cast<uint8_t*>(jl_array_data(keys_array_jl));

  keys_.resize(num_keys);
  sorted_ = true;
  memcpy(keys_.data(), key_bytes, num_keys * sizeof(size_t));
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
  JL_GC_POP();
}

void
DistArrayPartition<void>::BuildDenseIndex() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  Sort();
  if (keys_.size() > 0) key_start_ = keys_[0];
  storage_type_ = DistArrayPartitionStorageType::kKeyValueBuffer;
}

void
DistArrayPartition<void>::BuildSparseIndex() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t* keys_array_type_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* ptr_str_jl = nullptr;
  JL_GC_PUSH3(&keys_array_type_jl,
              &keys_array_jl,
              &ptr_str_jl);

  jl_value_t* dist_array_jl = dist_array_->GetJuliaDistArray();
  ptr_str_jl = jl_cstr_to_string(ptr_str_.c_str());
  keys_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  keys_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl,
      keys_.data(), keys_.size(), 0));
  jl_function_t *convert_func
      = JuliaEvaluator::GetOrionWorkerFunction("dist_array_partition_dense_to_sparse");
  jl_call3(convert_func, dist_array_jl, ptr_str_jl, keys_array_jl);

  std::vector<int64_t> empty_buff;
  keys_.swap(empty_buff);
  storage_type_ = DistArrayPartitionStorageType::kSparseIndex;
  JL_GC_POP();
}

void
DistArrayPartition<void>::GetAndSerializeValue(
    int64_t key, Blob *bytes_buff) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t* serialized_result_array = nullptr;
  jl_value_t* ptr_str_jl = nullptr;
  jl_value_t* key_jl = nullptr;
  JL_GC_PUSH3(&serialized_result_array,
              &ptr_str_jl,
              &key_jl);
  jl_value_t* dist_array_jl = dist_array_->GetJuliaDistArray();
  ptr_str_jl = jl_cstr_to_string(ptr_str_.c_str());
  key_jl = jl_box_int64(key);
  jl_function_t* get_value_func =
      JuliaEvaluator::GetOrionWorkerFunction("dist_array_partition_get_and_serialize_value");
  serialized_result_array = jl_call3(get_value_func, dist_array_jl, ptr_str_jl, key_jl);
  JuliaEvaluator::AbortIfException();
  size_t result_array_length = jl_array_len(serialized_result_array);
  uint8_t* array_bytes = reinterpret_cast<uint8_t*>(jl_array_data(serialized_result_array));
  bytes_buff->resize(result_array_length);
  memcpy(bytes_buff->data(), array_bytes, result_array_length);
  JL_GC_POP();
}

void
DistArrayPartition<void>::GetAndSerializeValues(
    int64_t *keys,
    size_t num_keys,
    Blob *bytes_buff) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t* serialized_result_array = nullptr;
  jl_value_t* ptr_str_jl = nullptr;
  jl_value_t* keys_array_type_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  JL_GC_PUSH4(&serialized_result_array,
              &ptr_str_jl,
              &keys_array_type_jl,
              &keys_array_jl);


  jl_value_t* dist_array_jl = dist_array_->GetJuliaDistArray();
  ptr_str_jl = jl_cstr_to_string(ptr_str_.c_str());
  keys_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  keys_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl, keys, num_keys, 0));
  jl_function_t* get_value_func =
      JuliaEvaluator::GetOrionWorkerFunction("dist_array_partition_get_and_serialize_values");

  serialized_result_array = jl_call3(get_value_func, dist_array_jl, ptr_str_jl,
                                     keys_array_jl);
  JuliaEvaluator::AbortIfException();
  size_t result_array_length = jl_array_len(serialized_result_array);
  uint8_t* array_bytes = reinterpret_cast<uint8_t*>(jl_array_data(serialized_result_array));

  bytes_buff->resize(sizeof(bool) + sizeof(size_t)
                     + sizeof(int64_t) * num_keys + sizeof(size_t) + result_array_length);

  auto *cursor = bytes_buff->data();
  *reinterpret_cast<bool*>(cursor) = false;
  cursor += sizeof(bool);
  *reinterpret_cast<size_t*>(cursor) = num_keys;
  cursor += sizeof(size_t);
  memcpy(cursor, keys, sizeof(int64_t) * num_keys);
  cursor += sizeof(int64_t) * num_keys;
  *reinterpret_cast<size_t*>(cursor) = result_array_length;
  cursor += sizeof(size_t);
  memcpy(cursor, array_bytes, result_array_length);
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
  std::vector<size_t> perm(keys_.size());

  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&] (const size_t &i, const size_t &j) {
              return keys_[i] < keys_[j];
            });

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
    size_t index = perm[i];
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(old_values_array_jl), index);
    jl_arrayset(reinterpret_cast<jl_array_t*>(new_values_array_jl), value_jl, i);
  }
  JuliaEvaluator::SetDistArrayPartition(dist_array_, ptr_str_, new_values_array_jl);
  JL_GC_POP();
  sorted_ = true;
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
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer)
      << " storage_type = " << static_cast<int>(storage_type_);

  jl_value_t* serialized_partition_array = nullptr;
  jl_value_t* ptr_str_jl = nullptr;
  JL_GC_PUSH2(&serialized_partition_array,
              &ptr_str_jl);
  jl_value_t* dist_array_jl = dist_array_->GetJuliaDistArray();
  ptr_str_jl = jl_cstr_to_string(ptr_str_.c_str());
  jl_function_t *serialize_func
      = JuliaEvaluator::GetOrionWorkerFunction("dist_array_serialize_partition");
  serialized_partition_array = jl_call2(serialize_func, dist_array_jl, ptr_str_jl);
  size_t serialized_partition_length = jl_array_len(serialized_partition_array);

  size_t num_bytes = sizeof(bool) + sizeof(size_t) + keys_.size() * sizeof(int64_t)
                     + sizeof(size_t) + serialized_partition_length;
  uint8_t* buff = new uint8_t[num_bytes];
  uint8_t* cursor = buff;
  *(reinterpret_cast<bool*>(cursor)) = sorted_;
  cursor += sizeof(bool);
  *(reinterpret_cast<size_t*>(cursor)) = keys_.size();
  cursor += sizeof(size_t);
  memcpy(cursor, keys_.data(), keys_.size() * sizeof(int64_t));
  cursor += sizeof(int64_t) * keys_.size();
  *(reinterpret_cast<size_t*>(cursor)) = serialized_partition_length;
  cursor += sizeof(size_t);
  uint8_t* serialized_bytes = reinterpret_cast<uint8_t*>(jl_array_data(
      serialized_partition_array));
  memcpy(cursor, serialized_bytes, serialized_partition_length);

  cursor += serialized_partition_length;
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  return std::make_pair(buff, num_bytes);
}

void
DistArrayPartition<void>::ModuloSerialize(
    ExecutorDataBufferMap *data_buffer_map) {
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 5);
  jl_value_t* &serialized_keys_values_tuples = jl_values[0];
  jl_value_t* &ptr_str_jl = jl_values[1];
  jl_value_t* &keys_array_type_jl = jl_values[2];
  jl_value_t* &keys_array_jl = jl_values[3];
  jl_value_t* &num_dests_jl = jl_values[4];

  jl_function_t* serialize_func
      = JuliaEvaluator::GetOrionWorkerFunction("dist_array_modulo_serialize_partition");
  keys_array_type_jl = jl_apply_array_type(
      reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  keys_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(keys_array_type_jl,
                         keys_.data(),
                         keys_.size(), 0));

  jl_value_t* dist_array_jl = dist_array_->GetJuliaDistArray();
  ptr_str_jl = jl_cstr_to_string(ptr_str_.c_str());
  num_dests_jl = jl_box_uint64(kConfig.kNumServers);
  jl_value_t* args[4];
  args[0] = dist_array_jl;
  args[1] = ptr_str_jl;
  args[2] = keys_array_jl;
  args[3] = num_dests_jl;
  serialized_keys_values_tuples = jl_call(serialize_func, args, 4);
  JuliaEvaluator::AbortIfException();
  jl_value_t* keys_by_dest_jl = jl_get_nth_field(serialized_keys_values_tuples, 0);
  jl_value_t* values_by_dest_jl = jl_get_nth_field(serialized_keys_values_tuples, 1);

  for (size_t i = 0; i < kConfig.kNumServers; i++) {
    jl_value_t* keys_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(keys_by_dest_jl),
                                      i);
    size_t num_keys = jl_array_len(reinterpret_cast<jl_array_t*>(keys_jl));
    jl_value_t * values_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(values_by_dest_jl), i);
    size_t serialized_value_size  = jl_array_len(reinterpret_cast<jl_array_t*>(values_jl));
    if (num_keys == 0) continue;
    size_t num_bytes = sizeof(bool) + sizeof(size_t)
                       + num_keys * sizeof(int64_t) + serialized_value_size
                       + sizeof(size_t);
    auto iter_pair = data_buffer_map->emplace(i,
                                              Blob(num_bytes));
    uint8_t* buff = iter_pair.first->second.data();
    uint8_t* cursor = buff;
    *reinterpret_cast<bool*>(cursor) = false;
    cursor += sizeof(bool);
    *reinterpret_cast<size_t*>(cursor) = num_keys;
    cursor += sizeof(size_t);
    auto* keys_bytes = reinterpret_cast<uint8_t*>(jl_array_data(keys_jl));
    memcpy(cursor, keys_bytes, num_keys * sizeof(int64_t));
    cursor += num_keys * sizeof(int64_t);
    *reinterpret_cast<size_t*>(cursor) = serialized_value_size;
    cursor += sizeof(size_t);
    auto *serialized_value_bytes = reinterpret_cast<uint8_t*>(jl_array_data(values_jl));
    memcpy(cursor, serialized_value_bytes, serialized_value_size);
  }
  JL_GC_POP();
}

uint8_t*
DistArrayPartition<void>::Deserialize(uint8_t *buffer) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer)
      << " storage_type = " << static_cast<int>(storage_type_);

  jl_value_t* serialized_partition_array = nullptr;
  jl_value_t* ptr_str_jl = nullptr;
  jl_value_t* serialized_partition_array_type = nullptr;
  JL_GC_PUSH3(&serialized_partition_array,
              &ptr_str_jl,
              &serialized_partition_array_type);

  uint8_t* cursor = buffer;
  sorted_ = *(reinterpret_cast<const bool*>(cursor));
  cursor += sizeof(bool);
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);
  keys_.resize(num_keys);
  memcpy(keys_.data(), cursor, num_keys * sizeof(int64_t));
  cursor += sizeof(int64_t) * num_keys;

  size_t serialized_partition_size = *(reinterpret_cast<size_t*>(cursor));
  cursor += sizeof(size_t);
  serialized_partition_array_type = jl_apply_array_type(
      reinterpret_cast<jl_value_t*>(jl_uint8_type), 1);
  serialized_partition_array = reinterpret_cast<jl_value_t*>(
     jl_ptr_to_array_1d(reinterpret_cast<jl_value_t*>(serialized_partition_array_type),
                        cursor, serialized_partition_size, 0));

  jl_value_t* dist_array_jl = dist_array_->GetJuliaDistArray();
  ptr_str_jl = jl_cstr_to_string(ptr_str_.c_str());
  jl_function_t *deserialize_func
      = JuliaEvaluator::GetOrionWorkerFunction("dist_array_deserialize_partition");

  jl_call3(deserialize_func, dist_array_jl,
           ptr_str_jl,
           serialized_partition_array);
  cursor += serialized_partition_size;
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  return cursor;
}

uint8_t*
DistArrayPartition<void>::DeserializeAndAppend(uint8_t *buffer) {
  sorted_ = false;
  jl_value_t* serialized_partition_array = nullptr;
  jl_value_t* ptr_str_jl = nullptr;
  jl_value_t* serialized_partition_array_type = nullptr;
  JL_GC_PUSH3(&serialized_partition_array,
              &ptr_str_jl,
              &serialized_partition_array_type);

  uint8_t* cursor = buffer;
  cursor += sizeof(bool);
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);
  size_t orig_num_keys = keys_.size();
  keys_.resize(orig_num_keys + num_keys);
  memcpy(keys_.data() + orig_num_keys, cursor, num_keys * sizeof(int64_t));
  cursor += sizeof(int64_t) * num_keys;
  size_t serialized_partition_size = *(reinterpret_cast<size_t*>(cursor));
  cursor += sizeof(size_t);
  serialized_partition_array_type = jl_apply_array_type(
      reinterpret_cast<jl_value_t*>(jl_uint8_type), 1);
  serialized_partition_array = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(reinterpret_cast<jl_value_t*>(serialized_partition_array_type), cursor,
                         serialized_partition_size, 0));
  jl_value_t* dist_array_jl = dist_array_->GetJuliaDistArray();
  ptr_str_jl = jl_cstr_to_string(ptr_str_.c_str());
  jl_function_t *deserialize_func
      = JuliaEvaluator::GetOrionWorkerFunction("dist_array_deserialize_and_append_partition");
  jl_call3(deserialize_func, dist_array_jl, ptr_str_jl, serialized_partition_array);
  cursor += serialized_partition_size;
  JL_GC_POP();
  return cursor;
}

uint8_t*
DistArrayPartition<void>::DeserializeAndOverwrite(
    uint8_t *buffer) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 5);
  jl_value_t* &serialized_partition_array_type = jl_values[0];
  jl_value_t* &serialized_partition_array = jl_values[1];
  jl_value_t* &ptr_str_jl = jl_values[2];
  jl_value_t* &keys_array_type_jl = jl_values[3];
  jl_value_t* &keys_array_jl = jl_values[4];

  uint8_t* cursor = buffer;
  cursor += sizeof(bool);
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);

  keys_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  keys_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl,
      cursor, num_keys, 0));
  cursor += sizeof(int64_t) * num_keys;
  size_t serialized_partition_size = *(reinterpret_cast<size_t*>(cursor));
  cursor += sizeof(size_t);
  serialized_partition_array_type = jl_apply_array_type(
      reinterpret_cast<jl_value_t*>(jl_uint8_type), 1);
  serialized_partition_array = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(reinterpret_cast<jl_value_t*>(serialized_partition_array_type), cursor,
                         serialized_partition_size, 0));
  cursor += serialized_partition_size;

  jl_value_t* dist_array_jl = dist_array_->GetJuliaDistArray();
  ptr_str_jl = jl_cstr_to_string(ptr_str_.c_str());
  jl_function_t *deserialize_func
      = JuliaEvaluator::GetOrionWorkerFunction("dist_array_deserialize_and_overwrite_partition");

  jl_value_t *args[4];
  args[0] = dist_array_jl;
  args[1] = ptr_str_jl;
  args[2] = keys_array_jl;
  args[3] = serialized_partition_array;
  jl_call(deserialize_func, args, 4);
  JL_GC_POP();
  return cursor;
}

void
DistArrayPartition<void>::Clear() {
  CHECK(storage_type_ != DistArrayPartitionStorageType::kAccessor);
  {
    std::vector<int64_t> empty_buff;
    keys_.swap(empty_buff);
  }
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
DistArrayPartition<void>::GetJuliaValueArray(std::vector<int64_t> &keys,
                                             jl_value_t **value_array) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);

  jl_value_t* keys_array_type_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* ptr_str_jl = nullptr;
  JL_GC_PUSH3(&keys_array_type_jl,
              &keys_array_jl,
              &ptr_str_jl);

  jl_value_t *dist_array_jl = dist_array_->GetJuliaDistArray();
  keys_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  keys_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(keys_array_type_jl, keys.data(), keys.size(), 0));
  ptr_str_jl = jl_cstr_to_string(ptr_str_.c_str());
  JuliaEvaluator::AbortIfException();
  jl_function_t* get_value_array_func
      = JuliaEvaluator::GetOrionWorkerFunction("dist_array_partition_get_value_array_by_keys");
  *value_array = jl_call3(get_value_array_func, dist_array_jl, ptr_str_jl,
                           keys_array_jl);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
}

void
DistArrayPartition<void>::SetJuliaValues(std::vector<int64_t> &keys,
                                         jl_value_t *values) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kSparseIndex);
  jl_value_t* keys_array_type_jl = nullptr;
  jl_value_t* keys_array_jl = nullptr;
  jl_value_t* ptr_str_jl = nullptr;
  JL_GC_PUSH3(&keys_array_type_jl, &keys_array_jl,
              &ptr_str_jl);
  jl_value_t *dist_array_jl = dist_array_->GetJuliaDistArray();
  keys_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  keys_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(keys_array_type_jl, keys.data(), keys.size(), 0));
  ptr_str_jl = jl_cstr_to_string(ptr_str_.c_str());
  jl_function_t* set_value_func
      = JuliaEvaluator::GetOrionWorkerFunction("dist_array_partition_set_values");

  jl_value_t* args[4];
  args[0] = dist_array_jl;
  args[1] = ptr_str_jl;
  args[2] = keys_array_jl;
  args[3] = values;
  jl_call(set_value_func, args, 4);
  JuliaEvaluator::AbortIfException();
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

void
DistArrayPartition<void>::ShrinkValueVecToFit() {
  JuliaEvaluator::ShrinkDistArrayPartitionToFit(dist_array_, ptr_str_);
}

}
}
