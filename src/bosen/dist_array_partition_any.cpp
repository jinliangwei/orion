#include <orion/bosen/dist_array_partition.hpp>
#include <orion/bosen/julia_module.hpp>

namespace orion {
namespace bosen {
/*---- template const char* implementation -----*/
DistArrayPartition<void>::DistArrayPartition(
    DistArray *dist_array,
    const Config &config,
    type::PrimitiveType value_type,
    JuliaThreadRequester *julia_requester):
    AbstractDistArrayPartition(dist_array, config, value_type, julia_requester),
    orion_worker_module_(GetOrionWorkerModule()) {
  JL_GC_PUSH2(&dist_array_jl_, &partition_jl_);

  auto &dist_array_meta = dist_array_->GetMeta();
  const std::string &symbol = dist_array_meta.GetSymbol();
  JuliaEvaluator::GetDistArray(symbol, &dist_array_jl_);
  jl_function_t *create_partition_func = JuliaEvaluator::GetFunction(
      jl_main_module,
      "orionres_dist_array_create_and_append_partition");
  partition_jl_ = jl_call1(create_partition_func, dist_array_jl_);
  JL_GC_POP();
}

DistArrayPartition<void>::~DistArrayPartition() { }

void
DistArrayPartition<void>::BuildKeyValueBuffersFromSparseIndex() {
  if (!sparse_index_exists_) return;
  if (!keys_.empty()) return;
  keys_.resize(sparse_index_.size());
  julia_array_index_.resize(sparse_index_.size());
  auto iter = sparse_index_.begin();
  size_t i = 0;
  for (; iter != sparse_index_.end(); iter++) {
    int64_t key = iter->first;
    auto index = iter->second;
    keys_[i] = key;
    julia_array_index_[i] = index;
  }
  sparse_index_.clear();
  sparse_index_exists_ = false;
}

void
DistArrayPartition<void>::ReadRangeDense(int64_t key_begin, size_t num_elements,
                                         jl_value_t* buff) {
  jl_value_t *value_jl = nullptr;
  JL_GC_PUSH1(&value_jl);
  size_t offset = key_begin - key_start_;
  for (size_t i = 0; i < num_elements; i++) {
    size_t index = julia_array_index_[i + offset];
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(partition_jl_), index);
    jl_arrayset(reinterpret_cast<jl_array_t*>(buff), value_jl, i);
  }
  JL_GC_POP();
}

void
DistArrayPartition<void>::ReadRangeSparse(int64_t key_begin,
                                          size_t num_elements,
                                          jl_value_t** key_buff,
                                          jl_value_t** value_buff) {
  jl_value_t* key_array_type = nullptr;
  jl_value_t* value_type = nullptr;
  jl_value_t* value_array_type = nullptr;
  jl_value_t *value_jl = nullptr;
  JL_GC_PUSH4(&key_array_type, &value_type,
              &value_array_type, &value_jl);

  key_array_type = jl_apply_array_type(jl_int64_type, 1);
  JuliaEvaluator::GetDistArrayValueType(dist_array_jl_, reinterpret_cast<jl_datatype_t**>(&value_type));
  value_array_type = jl_apply_array_type(reinterpret_cast<jl_datatype_t*>(value_type), 1);

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
    size_t array_index = iter->second;
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(partition_jl_), array_index);

    key_mem[index] = curr_key;
    jl_arrayset(reinterpret_cast<jl_array_t*>(*value_buff), value_jl, index);

    iter++;
    index++;
  }

  CHECK(index == num_values);
  JL_GC_POP();
}

void
DistArrayPartition<void>::ReadRangeSparseWithInitValue(
    int64_t key_begin, size_t num_elements,
    jl_value_t* value_buff) {
  auto &dist_array_meta = dist_array_->GetMeta();
  const auto &init_value_vec = dist_array_meta.GetInitValue();

  jl_value_t* buff_jl = nullptr;
  jl_value_t* serialized_value_array = nullptr;
  jl_value_t *serialized_value_array_type = nullptr;
  jl_value_t *init_value_jl = nullptr;
  jl_value_t *value_jl = nullptr;
  JL_GC_PUSH5(&buff_jl, &serialized_value_array,
              &serialized_value_array_type,
              &init_value_jl, &value_jl);
  std::vector<uint8_t> temp_init_value_vec = init_value_vec;

  jl_function_t *io_buffer_func
      = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");
  buff_jl = jl_call0(io_buffer_func);
  jl_function_t *deserialize_func
      = JuliaEvaluator::GetFunction(jl_base_module, "deserialize");

  serialized_value_array_type = jl_apply_array_type(jl_uint8_type, 1);
  serialized_value_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      serialized_value_array_type,
      temp_init_value_vec.data(),
      temp_init_value_vec.size(), 0));

  buff_jl = jl_call1(io_buffer_func, serialized_value_array);
  init_value_jl = jl_call1(deserialize_func, buff_jl);

  int64_t curr_key = key_begin;
  int64_t max_key = key_begin + num_elements;

  size_t index = 0;
  auto iter = sparse_index_.find(curr_key);

  while (iter == sparse_index_.end() &&
         curr_key + 1 < max_key) {
    jl_arrayset(reinterpret_cast<jl_array_t*>(value_buff), init_value_jl, index);
    curr_key++;
    index++;
    iter = sparse_index_.find(curr_key);
  }

  if (iter != sparse_index_.end()) {
    while (iter != sparse_index_.end() &&
           iter->first < max_key) {
      while (curr_key <= iter->first) {
        size_t array_index = iter->second;
        value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(partition_jl_), array_index);
        jl_arrayset(reinterpret_cast<jl_array_t*>(value_buff), value_jl, index);
        index++;
        curr_key++;
      }
      iter++;
    }
    for (; curr_key < max_key; curr_key++) {
      jl_arrayset(reinterpret_cast<jl_array_t*>(value_buff), init_value_jl, index);
      index++;
    }
  } else {
    jl_arrayset(reinterpret_cast<jl_array_t*>(value_buff), init_value_jl, index);
    index++;
  }

  CHECK(index == num_elements);
  JL_GC_POP();
}

void
DistArrayPartition<void>::ReadRangeSparseWithRequest(
    int64_t key_begin, size_t num_elements,
    jl_value_t** key_buff, jl_value_t** value_buff) {
  jl_value_t* key_array_type = nullptr;
  jl_value_t* value_array_type = nullptr;
  jl_value_t* key_jl = nullptr;
  jl_value_t* temp_value = nullptr;
  JL_GC_PUSH4(&key_array_type, &value_array_type,
              &key_jl, &temp_value);

  key_array_type = reinterpret_cast<jl_value_t*>(jl_apply_array_type(jl_int64_type, 1));
  value_array_type = reinterpret_cast<jl_value_t*>(jl_apply_array_type(type::GetJlDataType(kValueType), 1));

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
      size_t array_index = iter->second;
      temp_value = jl_arrayref(reinterpret_cast<jl_array_t*>(partition_jl_), array_index);
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
DistArrayPartition<void>::WriteRange(int64_t key_begin, size_t num_elements,
                                     jl_value_t* buff) {
  jl_value_t *value_jl = nullptr;
  JL_GC_PUSH1(&value_jl);
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense();
  CHECK_LE(key_start_, 0);
  if (is_dense) {
      size_t offset = key_begin - key_start_;
    for (size_t i = 0; i < num_elements; i++) {
      value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(buff), i);
      size_t index = julia_array_index_[offset + i];
      jl_arrayset(reinterpret_cast<jl_array_t*>(partition_jl_), value_jl, index);
    }
  } else {
    for (size_t i = 0; i < num_elements; i++) {
      value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(buff), i);
      auto iter = sparse_index_.find(key_begin + i);
      if (iter == sparse_index_.end()) {
        jl_array_ptr_1d_push(reinterpret_cast<jl_array_t*>(partition_jl_), value_jl);
        size_t index = jl_array_len(reinterpret_cast<jl_array_t*>(partition_jl_));
        sparse_index_[key_begin + i] = index;
      } else {
        size_t index = iter->second;
        jl_arrayset(reinterpret_cast<jl_array_t*>(partition_jl_), value_jl, index);
      }
    }
  }
  JL_GC_POP();
}

void
DistArrayPartition<void>::BuildIndex() {
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense();
  if (is_dense) {
    BuildDenseIndex();
  } else {
    BuildSparseIndex();
  }
}

void
DistArrayPartition<void>::BuildDenseIndex() {
  if (keys_.size() == 0) return;
  int64_t min_key = keys_[0];
  for (auto key : keys_) {
    min_key = std::min(key, min_key);
  }
  key_start_ = min_key;
  std::vector<int64_t> perm(keys_.size());
  std::vector<size_t> julia_index_temp(julia_array_index_);

  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&] (const size_t &i, const size_t &j) {
              return keys_[i] < keys_[j];
            });
  std::transform(perm.begin(), perm.end(), julia_array_index_.begin(),
                 [&](size_t i) { return julia_index_temp[i]; });

  for (size_t i = 0; i < keys_.size(); i++) {
    keys_[i] = min_key + i;
  }
}

void
DistArrayPartition<void>::BuildSparseIndex() {
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    const auto& index = julia_array_index_[i];
    sparse_index_[key] = index;
  }
  keys_.clear();
  julia_array_index_.clear();
  sparse_index_exists_ = true;
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
  JL_GC_PUSH1(&value_jl);
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    size_t index = julia_array_index_[i];
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(partition_jl_), index);
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
  JL_GC_PUSH1(&value_jl);
  for (size_t i = 0; i < keys_.size(); i++) {
    int64_t key = keys_[i];
    size_t index = julia_array_index_[i];
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(partition_jl_), index);
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
  JL_GC_PUSH3(&buff_jl, &serialized_value_array, &value_jl);

  jl_function_t *io_buffer_func
      = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");
  buff_jl = jl_call0(io_buffer_func);
  jl_function_t *serialize_func
      = JuliaEvaluator::GetFunction(jl_base_module, "serialize");

  size_t num_bytes = sizeof(size_t) + keys_.size() * sizeof(int64_t);
  for (size_t index : julia_array_index_) {
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(partition_jl_), index);
    jl_call2(serialize_func, buff_jl, value_jl);
    jl_function_t *takebuff_array_func
        = JuliaEvaluator::GetFunction(jl_base_module, "takebuf_array");
    serialized_value_array = jl_call1(takebuff_array_func, buff_jl);
    size_t result_array_length = jl_array_len(serialized_value_array);
    num_bytes += result_array_length + sizeof(size_t);
  }

  uint8_t* buff = new uint8_t[num_bytes];
  uint8_t* cursor = buff;
  *(reinterpret_cast<size_t*>(cursor)) = keys_.size();
  cursor += sizeof(size_t);
  memcpy(cursor, keys_.data(), keys_.size() * sizeof(int64_t));
  cursor += sizeof(int64_t) * keys_.size();
  for (size_t index : julia_array_index_) {
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(partition_jl_), index);
    jl_call2(serialize_func, buff_jl, value_jl);
    jl_function_t *takebuff_array_func
        = JuliaEvaluator::GetFunction(jl_base_module, "takebuf_array");
    serialized_value_array = jl_call1(takebuff_array_func, buff_jl);
    size_t result_array_length = jl_array_len(reinterpret_cast<jl_array_t*>(serialized_value_array));
    *(reinterpret_cast<size_t*>(cursor)) = result_array_length;
    cursor += sizeof(size_t);
    uint8_t* array_bytes = reinterpret_cast<uint8_t*>(jl_array_data(serialized_value_array));
    memcpy(cursor, array_bytes, result_array_length);
  }
  JL_GC_POP();
  return std::make_pair(buff, num_bytes);
}

const uint8_t*
DistArrayPartition<void>::Deserialize(const uint8_t *buffer) {
  jl_value_t* buff_jl = nullptr;
  jl_value_t* serialized_value_array = nullptr;
  jl_value_t* value_jl = nullptr;
  jl_value_t *serialized_value_array_type = nullptr;
  jl_value_t *uint64_jl = nullptr;
  JL_GC_PUSH5(&buff_jl, &serialized_value_array, &value_jl,
              &serialized_value_array_type, &uint64_jl);

  serialized_value_array_type = jl_apply_array_type(jl_uint8_type, 1);
  jl_function_t *io_buffer_func
      = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");
  buff_jl = jl_call0(io_buffer_func);
  jl_function_t *deserialize_func
      = JuliaEvaluator::GetFunction(jl_base_module, "deserialize");
  jl_function_t *resize_vec_func
      = JuliaEvaluator::GetFunction(jl_base_module, "resize!");

  const uint8_t* cursor = buffer;
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  uint64_jl = jl_box_uint64(num_keys);

  cursor += sizeof(size_t);
  keys_.resize(num_keys);
  julia_array_index_.resize(num_keys);
  jl_call2(resize_vec_func, partition_jl_, uint64_jl);
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
    value_jl = jl_call1(deserialize_func, buff_jl);
    jl_arrayset(reinterpret_cast<jl_array_t*>(partition_jl_), value_jl, i);
    size_t index = jl_array_len(reinterpret_cast<jl_array_t*>(partition_jl_));
    julia_array_index_[i] = index;
    cursor += serialized_value_size;
  }
  JL_GC_POP();
  return cursor;
}

const uint8_t*
DistArrayPartition<void>::DeserializeAndAppend(const uint8_t *buffer) {
  jl_value_t* buff_jl = nullptr;
  jl_value_t* serialized_value_array = nullptr;
  jl_value_t* value_jl = nullptr;
  jl_value_t *serialized_value_array_type = nullptr;
  JL_GC_PUSH4(&buff_jl, &serialized_value_array, &value_jl,
              &serialized_value_array_type);

  serialized_value_array_type = jl_apply_array_type(jl_uint8_type, 1);
  jl_function_t *io_buffer_func
      = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");
  buff_jl = jl_call0(io_buffer_func);
  jl_function_t *deserialize_func
      = JuliaEvaluator::GetFunction(jl_base_module, "deserialize");

  const uint8_t* cursor = buffer;
  size_t num_keys = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);

  size_t orig_num_keys = keys_.size();
  keys_.resize(orig_num_keys + num_keys);
  julia_array_index_.resize(orig_num_keys + num_keys);
  memcpy(keys_.data() + orig_num_keys, cursor, num_keys * sizeof(int64_t));
  cursor += sizeof(int64_t) * num_keys;
  for (size_t i = orig_num_keys; i < orig_num_keys + num_keys; i++) {
    size_t serialized_value_size = *reinterpret_cast<const size_t*>(cursor);
    cursor += sizeof(size_t);
    std::vector<uint8_t> temp(serialized_value_size);
    memcpy(temp.data(), cursor, serialized_value_size);
    serialized_value_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
        serialized_value_array_type,
        temp.data(),
        serialized_value_size, 0));
    buff_jl = jl_call1(io_buffer_func, serialized_value_array);
    value_jl = jl_call1(deserialize_func, buff_jl);
    jl_array_ptr_1d_push(reinterpret_cast<jl_array_t*>(partition_jl_), value_jl);
    size_t index = jl_array_len(reinterpret_cast<jl_array_t*>(partition_jl_));
    julia_array_index_[i] = index;
    cursor += serialized_value_size;
  }
  JL_GC_POP();
  return cursor;
}

void
DistArrayPartition<void>::Clear() {
  keys_.clear();
  julia_array_index_.clear();
  sparse_index_.clear();
  sparse_index_exists_ = false;
  jl_function_t *clear_partition_func
      = JuliaEvaluator::GetFunction(jl_main_module,
                                    "orionres_dist_array_clear_partition");
  jl_call1(clear_partition_func, partition_jl_);
  JuliaEvaluator::AbortIfException();
}

void
DistArrayPartition<void>::GetJuliaValueArray(jl_value_t **value) {
  jl_value_t* value_type = nullptr;
  jl_value_t* value_array_type = nullptr;
  jl_value_t *value_jl = nullptr;
  JL_GC_PUSH3(&value_type, &value_array_type, &value_jl);

  JuliaEvaluator::GetDistArrayValueType(dist_array_jl_,
                                        reinterpret_cast<jl_datatype_t**>(&value_type));
  value_array_type = jl_apply_array_type(reinterpret_cast<jl_datatype_t*>(value_type), 1);

  *value = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(value_array_type,
                                                           julia_array_index_.size()));
  for (size_t i = 0; i < julia_array_index_.size(); i++) {
    size_t index = julia_array_index_[i];
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(partition_jl_), index);
    jl_arrayset(reinterpret_cast<jl_array_t*>(*value), value_jl, i);
  }
  JL_GC_POP();
}

void
DistArrayPartition<void>::AppendJuliaValue(jl_value_t *value) {
  jl_array_ptr_1d_push(reinterpret_cast<jl_array_t*>(partition_jl_), value);
  size_t index = jl_array_len(reinterpret_cast<jl_array_t*>(partition_jl_));
  julia_array_index_.push_back(index);
}

void
DistArrayPartition<void>::AppendJuliaValueArray(jl_value_t *value) {
  jl_value_t* value_jl = nullptr;
  JL_GC_PUSH1(&value_jl);
  size_t num_elements = jl_array_len(reinterpret_cast<jl_array_t*>(value));
  for (size_t i = 0; i < num_elements; i++) {
    value_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(value), i);
    jl_array_ptr_1d_push(reinterpret_cast<jl_array_t*>(partition_jl_), value_jl);
    size_t index = jl_array_len(partition_jl_);
    julia_array_index_.push_back(index);
  }
  JL_GC_POP();
}

}
}
