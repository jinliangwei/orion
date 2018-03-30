#include <orion/bosen/abstract_exec_for_loop.hpp>
#include <orion/bosen/julia_evaluator.hpp>
#include <orion/bosen/julia_module.hpp>
#include <vector>
#include <memory>
namespace orion {
namespace bosen {

AbstractExecForLoop::AbstractExecForLoop(
    int32_t executor_id,
    size_t num_executors,
    size_t num_servers,
    int32_t iteration_space_id,
    const int32_t *space_partitioned_dist_array_ids,
    size_t num_space_partitioned_dist_arrays,
    const int32_t *time_partitioned_dist_array_ids,
    size_t num_time_partitioned_dist_arrays,
    const int32_t *global_indexed_dist_array_ids,
    size_t num_global_indexed_dist_arrays,
    const int32_t *dist_array_buffer_ids,
    size_t num_dist_array_buffers,
    const int32_t *written_dist_array_ids,
    size_t num_written_dist_array_ids,
    const int32_t *accessed_dist_array_ids,
    size_t num_accessed_dist_arrays,
    const std::string * const *global_read_only_var_vals,
    size_t num_global_read_only_var_vals,
    const std::string * const *accumulator_var_syms,
    size_t num_accumulator_var_syms,
    const char* loop_batch_func_name,
    const char *prefetch_batch_func_name,
    std::unordered_map<int32_t, DistArray> *dist_arrays,
    std::unordered_map<int32_t, DistArray> *dist_array_buffers):
    kExecutorId(executor_id),
    kNumExecutors(num_executors),
    kNumServers(num_servers),
    kLoopBatchFuncName(loop_batch_func_name),
    kPrefetchBatchFuncName(prefetch_batch_func_name) {

  auto iter = dist_arrays->find(iteration_space_id);
  CHECK(iter != dist_arrays->end());
  iteration_space_ = &(iter->second);

  for (size_t i = 0; i < num_space_partitioned_dist_arrays; i++) {
    int32_t id = space_partitioned_dist_array_ids[i];
    auto iter = dist_arrays->find(id);
    CHECK(iter != dist_arrays->end());
    space_partitioned_dist_arrays_.emplace(std::make_pair(id, &(iter->second)));
  }

  for (size_t i = 0; i < num_time_partitioned_dist_arrays; i ++) {
    int32_t id = time_partitioned_dist_array_ids[i];
    auto iter = dist_arrays->find(id);
    CHECK(iter != dist_arrays->end());
    time_partitioned_dist_arrays_.emplace(std::make_pair(id, &(iter->second)));
  }

  for (size_t i = 0; i < num_global_indexed_dist_arrays; i++) {
    int32_t id = global_indexed_dist_array_ids[i];
    auto iter = dist_arrays->find(id);
    CHECK(iter != dist_arrays->end());
    auto *dist_array_ptr = &(iter->second);
    global_indexed_dist_arrays_.emplace(std::make_pair(id, dist_array_ptr));
    dist_array_cache_.emplace(id,
                              std::unique_ptr<AbstractDistArrayPartition>(dist_array_ptr->CreatePartition()));
  }

  for (size_t i = 0; i < num_dist_array_buffers; i++) {
    int32_t dist_array_buffer_id = dist_array_buffer_ids[i];
    auto iter = dist_array_buffers->find(dist_array_buffer_id);
    CHECK(iter != dist_array_buffers->end());
    auto *dist_array_buffer = &(iter->second);
    dist_array_buffers_.emplace(dist_array_buffer_id, dist_array_buffer);
  }

  for (size_t i = 0; i < num_written_dist_array_ids; i++) {
    written_dist_array_ids_.emplace(written_dist_array_ids[i]);
  }

  for (size_t i = 0; i < num_accessed_dist_arrays; i++) {
    int32_t dist_array_id = accessed_dist_array_ids[i];
    auto iter = dist_arrays->find(dist_array_id);
    CHECK(iter != dist_arrays->end());
    auto *dist_array_ptr = &(iter->second);
    accessed_dist_array_syms_.emplace_back(dist_array_ptr->GetMeta().GetSymbol());
  }

  for (size_t i = 0; i < num_global_read_only_var_vals; i++) {
    global_read_only_var_vals_.emplace_back(*global_read_only_var_vals[i]);
  }

  for (size_t i = 0; i < num_accumulator_var_syms; i++) {
    accumulator_var_syms_.emplace_back(*accumulator_var_syms[i]);
  }
}

AbstractExecForLoop::~AbstractExecForLoop() { }

void
AbstractExecForLoop::Init() {
  jl_value_t *num_global_read_only_var_vals_jl = nullptr;
  jl_value_t *index_jl = nullptr;
  jl_value_t *serialized_value_array = nullptr;
  jl_value_t *serialized_value_array_type = nullptr;

  JL_GC_PUSH4(&num_global_read_only_var_vals_jl,
              &index_jl,
              &serialized_value_array,
              &serialized_value_array_type);

  accessed_dist_arrays_.resize(accessed_dist_array_syms_.size());
  jl_value_t *dist_array = nullptr;
  for (size_t i = 0; i < accessed_dist_array_syms_.size(); i++) {
    JuliaEvaluator::GetDistArray(accessed_dist_array_syms_[i], &dist_array);
    accessed_dist_arrays_[i] = dist_array;
  }

  size_t num_global_read_only_var_vals = global_read_only_var_vals_.size();
  global_read_only_var_jl_vals_.resize(num_global_read_only_var_vals);

  jl_module_t *orion_worker_module = GetJlModule(JuliaModule::kOrionWorker);
  jl_function_t *resize_global_read_only_var_buff_func
      = JuliaEvaluator::GetFunction(orion_worker_module,
                                    "resize_global_read_only_var_buff");
  jl_function_t *deserialize_func
      = JuliaEvaluator::GetFunction(orion_worker_module,
                                    "global_read_only_var_buff_deserialize_and_set");
  num_global_read_only_var_vals_jl = jl_box_uint64(
      num_global_read_only_var_vals);
  jl_call1(resize_global_read_only_var_buff_func,
           num_global_read_only_var_vals_jl);
  serialized_value_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_uint8_type), 1);
  for (size_t i = 0; i < global_read_only_var_vals_.size(); i++) {
    const std::string &var_val = global_read_only_var_vals_[i];
    std::vector<uint8_t> temp_var(var_val.size());
    memcpy(temp_var.data(), var_val.data(), var_val.size());
    index_jl = jl_box_uint64(i + 1);
    serialized_value_array = reinterpret_cast<jl_value_t*>(
        jl_ptr_to_array_1d(serialized_value_array_type,
                           temp_var.data(),
                           temp_var.size(), 0));
    jl_value_t *value_jl = jl_call2(deserialize_func,
                                    index_jl,
                                    serialized_value_array);
    global_read_only_var_jl_vals_[i] = value_jl;
  }
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
}

void
AbstractExecForLoop::Clear() {
  jl_module_t *orion_worker_module = GetJlModule(JuliaModule::kOrionWorker);
  jl_function_t *clear_func = JuliaEvaluator::GetFunction(
      orion_worker_module,
      "clear_global_read_only_var_buff");
  jl_call0(clear_func);
  JuliaEvaluator::AbortIfException();
}

void
AbstractExecForLoop::SentAllPrefetchRequests() {
  prefetch_status_ = PrefetchStatus::kPrefetchSent;
}

bool
AbstractExecForLoop::HasSentAllPrefetchRequests() const {
  return (prefetch_status_ != PrefetchStatus::kNotPrefetched);
}

bool
AbstractExecForLoop::HasRecvedAllPrefetches() const {
  return (prefetch_status_ == PrefetchStatus::kPrefetchRecved);
}

bool
AbstractExecForLoop::HasRecvedAllTimePartitionedDistArrays(
    int32_t time_partition_id) const {
  for (auto &dist_array_pair : time_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    auto* partition = dist_array->GetLocalPartition(time_partition_id);
    if (partition == nullptr) return false;
  }
  return true;
}

void
AbstractExecForLoop::ComputePrefetchIndinces() {
  PrepareToExecCurrPartition();
  std::vector<int32_t> dist_array_ids_vec;

  for (const auto &dist_array_pair : global_indexed_dist_arrays_) {
    dist_array_ids_vec.push_back(dist_array_pair.first);
  }

  curr_partition_->ComputePrefetchIndinces(
      kPrefetchBatchFuncName,
      dist_array_ids_vec,
      global_indexed_dist_arrays_,
      accessed_dist_arrays_,
      global_read_only_var_jl_vals_,
      accumulator_var_syms_,
      &point_prefetch_dist_array_map_);
}

void
AbstractExecForLoop::ExecuteForLoopPartition() {
  PrepareToExecCurrPartition();
  PrepareDistArrayCachePartitions();
  curr_partition_->Execute(kLoopBatchFuncName,
                           accessed_dist_arrays_,
                           global_read_only_var_jl_vals_,
                           accumulator_var_syms_);
  ClearCurrPartition();
}

void
AbstractExecForLoop::SerializeAndClearPrefetchIds(ExecutorSendBufferMap *send_buffer_map) {
  std::unordered_map<int32_t, PointQueryKeyDistArrayMap> server_point_key_map;

  for (const auto &dist_array_key_pair : point_prefetch_dist_array_map_) {
    int32_t dist_array_id = dist_array_key_pair.first;
    const auto &point_key_vec = dist_array_key_pair.second;
    for (auto key : point_key_vec) {
      int32_t server_id = key % kNumServers;
      server_point_key_map[server_id][dist_array_id].push_back(key);
    }
  }

  num_pending_prefetch_requests_ = server_point_key_map.size();
  for (const auto &server_point_key_pair : server_point_key_map) {
    int32_t server_id = server_point_key_pair.first;
    const auto &dist_array_point_key_map = server_point_key_pair.second;
    size_t server_num_bytes = sizeof(size_t); // num_dist_arrays
    for (const auto &dist_array_point_key_pair : dist_array_point_key_map) {
      const auto &point_key_vec = dist_array_point_key_pair.second;
      // dist_array_id, num_keys, key_vec
      server_num_bytes += sizeof(int32_t) + sizeof(size_t)
                          + point_key_vec.size() * sizeof(int64_t);
    }
    uint8_t *server_buff = new uint8_t[server_num_bytes];
    (*send_buffer_map)[server_id] = std::make_pair(server_buff, server_num_bytes);
    uint8_t *cursor = server_buff;
    // num_dist_arrays
    *reinterpret_cast<size_t*>(cursor) = dist_array_point_key_map.size();
    cursor += sizeof(size_t);

    for (const auto &dist_array_point_key_pair : dist_array_point_key_map) {
      int32_t dist_array_id = dist_array_point_key_pair.first;
      const auto &point_key_vec = dist_array_point_key_pair.second;
      *reinterpret_cast<int32_t*>(cursor) = dist_array_id;
      cursor += sizeof(int32_t);
      *reinterpret_cast<size_t*>(cursor) = point_key_vec.size();
      cursor += sizeof(size_t);
      memcpy(cursor, point_key_vec.data(), point_key_vec.size() * sizeof(int64_t));
      cursor += sizeof(int64_t) * point_key_vec.size();
    }
  }
  point_prefetch_dist_array_map_.clear();
}

void
AbstractExecForLoop::SerializeAndClearGlobalPartitionedDistArrays() {
  SerializeAndClearDistArrayBuffers();
  SerializeAndClearDistArrayCaches();
}

void
AbstractExecForLoop::SerializeAndClearDistArrayBuffers() {
  std::unordered_map<int32_t, size_t> server_buffer_accum_size;
  std::unordered_map<int32_t, size_t> server_num_buffers;
  std::unordered_map<int32_t, ExecutorDataBufferMap> dist_array_buffer_data_buffer_map;
  std::unordered_map<int32_t, uint8_t*> server_cursor_map;
  for (auto &dist_array_pair : dist_array_buffers_) {
    int32_t dist_array_buffer_id = dist_array_pair.first;
    auto *buffer_partition = dist_array_pair.second->GetBufferPartition();
    auto iter_pair = dist_array_buffer_data_buffer_map.emplace(dist_array_buffer_id, ExecutorDataBufferMap());
    auto iter = iter_pair.first;
    auto &data_buffer_map = iter->second;
    buffer_partition->HashSerialize(&data_buffer_map);
    buffer_partition->Clear();
    for (auto &data_buff_pair : data_buffer_map) {
      int32_t server_id = data_buff_pair.first;
      auto &data_buff = data_buff_pair.second;
      auto server_iter = server_buffer_accum_size.find(server_id);
      if (server_iter == server_buffer_accum_size.end()) {
        server_buffer_accum_size[server_id] = data_buff.size() + sizeof(int32_t);
        server_num_buffers[server_id] = 1;
      } else {
        server_buffer_accum_size[server_id] += data_buff.size() + sizeof(int32_t);
        server_num_buffers[server_id] += 1;
      }
    }
  }

  for (auto &buffer_accum_size_pair : server_buffer_accum_size) {
    int32_t server_id = buffer_accum_size_pair.first;
    size_t buffer_accum_size = buffer_accum_size_pair.second + sizeof(size_t);
    uint8_t* buff = new uint8_t[buffer_accum_size];
    buffer_send_buffer_map_.emplace(server_id, std::make_pair(buff, buffer_accum_size));
    uint8_t* cursor = buff;
    *reinterpret_cast<size_t*>(cursor) = server_num_buffers[server_id];
    cursor += sizeof(size_t);
    server_cursor_map.emplace(server_id, cursor);
  }

  for (auto &data_buff_pair : dist_array_buffer_data_buffer_map) {
    int32_t dist_array_id = data_buff_pair.first;
    auto &data_buffer_map = data_buff_pair.second;
    for (auto &data_buff_pair : data_buffer_map) {
      int32_t server_id = data_buff_pair.first;
      auto &data_buff = data_buff_pair.second;
      *reinterpret_cast<int32_t*>(server_cursor_map[server_id]) = dist_array_id;
      server_cursor_map[server_id] += sizeof(int32_t);
      memcpy(server_cursor_map[server_id], data_buff.data(), data_buff.size());
      server_cursor_map[server_id] += data_buff.size();
    }
  }
}

void
AbstractExecForLoop::SerializeAndClearDistArrayCaches() {
  std::unordered_map<int32_t, size_t> server_cache_accum_size;
  std::unordered_map<int32_t, size_t> server_num_caches;
  std::unordered_map<int32_t, ExecutorDataBufferMap> dist_array_cache_data_buffer_map;
  std::unordered_map<int32_t, uint8_t*> server_cursor_map;
  for (auto &dist_array_pair : dist_array_cache_) {
    auto dist_array_id = dist_array_pair.first;
    auto *cache_partition = dist_array_pair.second.get();
    if (written_dist_array_ids_.count(dist_array_id) == 0) {
      cache_partition->Clear();
      continue;
    }
    auto iter_pair = dist_array_cache_data_buffer_map.emplace(dist_array_id, ExecutorDataBufferMap());
    auto iter = iter_pair.first;
    auto &data_buffer_map = iter->second;
    cache_partition->HashSerialize(&data_buffer_map);
    cache_partition->Clear();
    for (auto &data_buff_pair : data_buffer_map) {
      int32_t server_id = data_buff_pair.first;
      auto &data_buff = data_buff_pair.second;
      auto server_iter = server_cache_accum_size.find(server_id);
      if (server_iter == server_cache_accum_size.end()) {
        server_cache_accum_size[server_id] = data_buff.size() + sizeof(int32_t);
        server_num_caches[server_id] = 1;
      } else {
        server_cache_accum_size[server_id] += data_buff.size() + sizeof(int32_t);
        server_num_caches[server_id] += 1;
      }
    }
  }

  for (auto &cache_accum_size_pair : server_cache_accum_size) {
    int32_t server_id = cache_accum_size_pair.first;
    size_t cache_accum_size = cache_accum_size_pair.second + sizeof(size_t);
    uint8_t* buff = new uint8_t[cache_accum_size];
    cache_send_buffer_map_.emplace(server_id, std::make_pair(buff, cache_accum_size));
    uint8_t* cursor = buff;
    *reinterpret_cast<size_t*>(cursor) = server_num_caches[server_id];
    cursor += sizeof(size_t);
    server_cursor_map.emplace(server_id, cursor);
  }

  for (auto &data_buff_pair : dist_array_cache_data_buffer_map) {
    int32_t dist_array_id = data_buff_pair.first;
    auto &data_buffer_map = data_buff_pair.second;
    for (auto &data_buff_pair : data_buffer_map) {
      int32_t server_id = data_buff_pair.first;
      auto &data_buff = data_buff_pair.second;
      *reinterpret_cast<int32_t*>(server_cursor_map[server_id]) = dist_array_id;
      server_cursor_map[server_id] += sizeof(int32_t);
      memcpy(server_cursor_map[server_id], data_buff.data(), data_buff.size());
      server_cursor_map[server_id] += data_buff.size();
    }
  }
}

void
AbstractExecForLoop::GetAndClearDistArrayBufferSendMap(
    ExecutorSendBufferMap *buffer_send_buffer_map) {
  *buffer_send_buffer_map = buffer_send_buffer_map_;
  buffer_send_buffer_map_.clear();
}

void
AbstractExecForLoop::GetAndClearDistArrayCacheSendMap(
    ExecutorSendBufferMap *cache_send_buffer_map) {
  *cache_send_buffer_map = cache_send_buffer_map_;
  cache_send_buffer_map_.clear();
}

void
AbstractExecForLoop::SerializeAndClearPipelinedTimePartitions() {
  time_partitions_serialized_bytes_ = nullptr;
  time_partitions_serialized_size_ = 0;
  int32_t time_partition_id_to_send = GetTimePartitionIdToSend();
  if (time_partition_id_to_send < 0) return;
  std::unordered_map<int32_t, SendDataBuffer> data_buffers;
  for (auto &dist_array_pair : time_partitioned_dist_arrays_) {
    int32_t dist_array_id = dist_array_pair.first;
    auto *dist_array = dist_array_pair.second;
    auto *dist_array_partition = dist_array->GetLocalPartition(
        time_partition_id_to_send);
    if (dist_array_partition == nullptr) continue;
    auto data_buff = dist_array_partition->Serialize();
    if (data_buff.second > 0) {
      CHECK(data_buff.first != nullptr);
      data_buffers.emplace(dist_array_id, data_buff);
    }
    dist_array->DeletePartition(time_partition_id_to_send);
    dist_array->GcPartitions();
  }

  if (data_buffers.empty()) return;
  size_t total_size = sizeof(size_t) + sizeof(int32_t);
  for (auto &data_buff : data_buffers) {
    total_size += sizeof(int32_t) + data_buff.second.second;
  }
  auto* buffer_bytes = new uint8_t[total_size];
  uint8_t* cursor = buffer_bytes;
  *reinterpret_cast<size_t*>(cursor) = data_buffers.size();
  cursor += sizeof(size_t);
  *reinterpret_cast<int32_t*>(cursor) = time_partition_id_to_send;
  cursor += sizeof(int32_t);
  for (auto &data_buff_pair : data_buffers) {
    int32_t dist_array_id = data_buff_pair.first;
    auto &data_buff = data_buff_pair.second;
    *reinterpret_cast<int32_t*>(cursor) = dist_array_id;
    cursor += sizeof(int32_t);
    memcpy(cursor, data_buff.first, data_buff.second);
    cursor += data_buff.second;
    delete[] data_buff.first;
  }
  time_partitions_serialized_bytes_ = buffer_bytes;
  time_partitions_serialized_size_ = total_size;
}

void
AbstractExecForLoop::DeserializePipelinedTimePartitions(const uint8_t* bytes) {
  const uint8_t *cursor = bytes;
  size_t num_data_buffers = *reinterpret_cast<const size_t*>(cursor);
  cursor += sizeof(size_t);
  int32_t time_partition_id = *reinterpret_cast<const int32_t*>(cursor);
  cursor += sizeof(int32_t);

  for (size_t i = 0; i < num_data_buffers; i++) {
    int32_t dist_array_id = *reinterpret_cast<const int32_t*>(cursor);
    cursor += sizeof(int32_t);
    auto iter = time_partitioned_dist_arrays_.find(dist_array_id);
    CHECK(iter != time_partitioned_dist_arrays_.end())
        << "id = " << dist_array_id;
    auto *dist_array = iter->second;
    auto create_pair = dist_array->GetAndCreateLocalPartition(time_partition_id);
    CHECK(create_pair.second);
    auto *partition = create_pair.first;
    cursor = partition->Deserialize(cursor);
    partition->BuildIndex();
  }
}

void
AbstractExecForLoop::DeserializePipelinedTimePartitionsBuffVec(
    PeerRecvPipelinedTimePartitionsBuffer** buff_vec,
    size_t num_buffs) {
  for (size_t i = 0; i < num_buffs; i++) {
    auto *buff = buff_vec[i];
    auto &byte_buff = buff->byte_buff;
    uint64_t pred_notice = buff->pred_notice;
    ApplyPredecessorNotice(pred_notice);
    DeserializePipelinedTimePartitions(byte_buff.GetBytes());
    delete buff;
  }
  delete[] buff_vec;
}

SendDataBuffer
AbstractExecForLoop::GetAndResetSerializedTimePartitions() {
  auto send_data_buffer = std::make_pair(time_partitions_serialized_bytes_,
                                          time_partitions_serialized_size_);
  time_partitions_serialized_bytes_ = nullptr;
  time_partitions_serialized_size_ = 0;
  return send_data_buffer;
}

void
AbstractExecForLoop::CachePrefetchDistArrayValues(
    PeerRecvGlobalIndexedDistArrayDataBuffer **buff_vec,
    size_t num_buffs) {
  CHECK(num_pending_prefetch_requests_ > 0);
  for (size_t i = 0; i < num_buffs; i++) {
    auto *buff = buff_vec[i];
    const auto *bytes = buff->byte_buff.data();
    const auto *cursor = bytes;
    size_t num_dist_arrays = *reinterpret_cast<const size_t*>(cursor);
    cursor += sizeof(size_t);
    for (size_t j = 0; j < num_dist_arrays; j++) {
      int32_t dist_array_id = *reinterpret_cast<const int32_t*>(cursor);
      LOG(INFO) << __func__ << " dist_array_id = " << dist_array_id;
      cursor += sizeof(int32_t);
      auto iter = dist_array_cache_.find(dist_array_id);
      CHECK(iter != dist_array_cache_.end()) << " dist_array_id = " << dist_array_id;
      auto *cache_partition = iter->second.get();
      cursor = cache_partition->DeserializeAndAppend(cursor);
    }
    delete buff;
  }
  delete[] buff_vec;
  num_pending_prefetch_requests_ -= num_buffs;

  if (num_pending_prefetch_requests_ == 0) {
    prefetch_status_ = PrefetchStatus::kPrefetchRecved;
  }
}

void
AbstractExecForLoop::PrepareDistArrayCachePartitions() {
  if (dist_array_cache_prepared_) return;
  for (auto& dist_array_pair : global_indexed_dist_arrays_) {
    auto dist_array_id = dist_array_pair.first;
    auto *cache_partition = dist_array_cache_.at(dist_array_id).get();
    cache_partition->CreateCacheAccessor();
  }
  dist_array_cache_prepared_ = true;
}

}
}
