#include <orion/bosen/abstract_exec_for_loop.hpp>
#include <orion/bosen/julia_evaluator.hpp>
#include <orion/bosen/julia_module.hpp>
#include <orion/bosen/util.hpp>
#include <string.h>
#include <vector>
#include <memory>

namespace orion {
namespace bosen {

// DistArray Prefetch
// 1) if a DistArray is written to, it is prefetch at the beginning of each
// partitoin to ensure dependence is preserved;
// 2) if a DistArray has one or multiple DistArrayBuffers applied to it, it is prefetch
// when each DistArrayBuffer is applied

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
    std::unordered_map<int32_t, DistArray> *dist_array_buffers,
    const std::unordered_map<int32_t, DistArrayBufferInfo> &dist_array_buffer_info_map,
    bool is_repeated):
    prefetch_status_(strlen(prefetch_batch_func_name) == 0 ? PrefetchStatus::kSkipPrefetch : PrefetchStatus::kNotPrefetched),
    kExecutorId(executor_id),
    kNumExecutors(num_executors),
    kNumServers(num_servers),
    kLoopBatchFuncName(loop_batch_func_name),
    kPrefetchBatchFuncName(prefetch_batch_func_name),
    kDistArrayBufferInfoMap(dist_array_buffer_info_map),
    kIsRepeated(is_repeated) {
  LOG(INFO) << "strlen(prefetch_batch_func_name) = " << strlen(prefetch_batch_func_name);
  auto iter = dist_arrays->find(iteration_space_id);
  CHECK(iter != dist_arrays->end());
  iteration_space_ = &(iter->second);

  std::set<int32_t> accessed_dist_array_id_set;
  for (size_t i = 0; i < num_accessed_dist_arrays; i++) {
    int32_t dist_array_id = accessed_dist_array_ids[i];
    accessed_dist_array_id_set.emplace(dist_array_id);
  }

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
    if (accessed_dist_array_id_set.count(id) == 1) {
      accessed_global_indexed_dist_arrays_.emplace(std::make_pair(id, dist_array_ptr));
      dist_array_cache_buffer_to_create_accessor_.emplace(std::make_pair(id, true));
    }
  }

  for (size_t i = 0; i < num_dist_array_buffers; i++) {
    int32_t dist_array_buffer_id = dist_array_buffer_ids[i];
    auto iter = dist_array_buffers->find(dist_array_buffer_id);
    CHECK(iter != dist_array_buffers->end());
    auto *dist_array_buffer = &(iter->second);
    dist_array_buffers_.emplace(dist_array_buffer_id, dist_array_buffer);
    accessed_dist_array_buffer_syms_.emplace_back(dist_array_buffer->GetMeta().GetSymbol());
    dist_array_cache_buffer_to_create_accessor_.emplace(
        std::make_pair(dist_array_buffer_id, true));
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

  global_read_only_var_vals_.resize(num_global_read_only_var_vals);
  for (size_t i = 0; i < num_global_read_only_var_vals; i++) {
    global_read_only_var_vals_[i] = *global_read_only_var_vals[i];
  }

  for (size_t i = 0; i < num_accumulator_var_syms; i++) {
    accumulator_var_syms_.emplace_back(*accumulator_var_syms[i]);
  }
}

AbstractExecForLoop::~AbstractExecForLoop() { }

void
AbstractExecForLoop::ResetGlobalReadOnlyVarVals(const std::string * const *global_read_only_var_vals,
                                                size_t num_global_read_only_var_vals) {
  for (size_t i = 0; i < num_global_read_only_var_vals; i++) {
    global_read_only_var_vals_[i] = *global_read_only_var_vals[i];
  }
}

void
AbstractExecForLoop::InitOnCreation() {
  accessed_dist_arrays_.resize(accessed_dist_array_syms_.size());
  jl_value_t *dist_array = nullptr;
  for (size_t i = 0; i < accessed_dist_array_syms_.size(); i++) {
    JuliaEvaluator::GetVarJlValue(accessed_dist_array_syms_[i], &dist_array);
    accessed_dist_arrays_[i] = dist_array;
  }

  accessed_dist_array_buffers_.resize(accessed_dist_array_buffer_syms_.size());
  jl_value_t *dist_array_buffer = nullptr;
  for (size_t i = 0; i < accessed_dist_array_buffer_syms_.size(); i++) {
    JuliaEvaluator::GetVarJlValue(accessed_dist_array_buffer_syms_[i], &dist_array_buffer);
    accessed_dist_array_buffers_[i] = dist_array_buffer;
  }
  InitExecInterval();
  InitEachExecution(true);
}

void
AbstractExecForLoop::InitEachExecution(bool is_first_time) {
  LOG(INFO) << __func__ << " is_first_time = " << is_first_time_
            << " prefetch status = " << static_cast<int>(prefetch_status_);
  for (auto global_indexed_dist_array_pair : accessed_global_indexed_dist_arrays_) {
    int32_t id = global_indexed_dist_array_pair.first;
    auto* dist_array_ptr = global_indexed_dist_array_pair.second;
    dist_array_cache_.emplace(id,
                              std::unique_ptr<AbstractDistArrayPartition>(dist_array_ptr->CreatePartition()));
  }

  jl_value_t *num_global_read_only_var_vals_jl = nullptr;
  jl_value_t *index_jl = nullptr;
  jl_value_t *serialized_value_array = nullptr;
  jl_value_t *serialized_value_array_type = nullptr;

  JL_GC_PUSH4(&num_global_read_only_var_vals_jl,
              &index_jl,
              &serialized_value_array,
              &serialized_value_array_type);

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

  for (auto &to_create_accessor_pair : dist_array_cache_buffer_to_create_accessor_) {
    to_create_accessor_pair.second = true;
  }
  server_prefetch_info_list_iter_ = server_prefetch_info_list_.begin();
  is_first_time_ = is_first_time;
  num_elements_to_exec_ = num_elements_per_exec_;
  skipped_time_partitioned_dist_array_id_map_.clear();
  prefetch_status_ = kPrefetchBatchFuncName.empty() ?
                     PrefetchStatus::kSkipPrefetch :
                     PrefetchStatus::kNotPrefetched;
  InitClocks();
  ComputePartitionIdsAndFindPartitionToExecute();
}

void
AbstractExecForLoop::Clear() {
  jl_module_t *orion_worker_module = GetJlModule(JuliaModule::kOrionWorker);
  jl_function_t *clear_func = JuliaEvaluator::GetFunction(
      orion_worker_module,
      "clear_global_read_only_var_buff");
  jl_call0(clear_func);
  dist_array_cache_.clear();
  JuliaEvaluator::AbortIfException();
}

void
AbstractExecForLoop::InitExecInterval() {
  size_t num_elements_per_exec = 0;
  std::set<int32_t> global_indexed_dist_array_ids_with_buffer_applied;
  for (auto &dist_array_buffer_pair : dist_array_buffers_) {
    int32_t dist_array_buffer_id = dist_array_buffer_pair.first;
    auto buffer_info_iter = kDistArrayBufferInfoMap.find(dist_array_buffer_id);
    if (buffer_info_iter == kDistArrayBufferInfoMap.end()) continue;
    const auto &buffer_info = buffer_info_iter->second;
    auto delay_mode = buffer_info.kDelayMode;
    auto max_delay = buffer_info.kMaxDelay;
    if (delay_mode == DistArrayBufferDelayMode::kMaxDelay) {
      if (num_elements_per_exec == 0) {
        num_elements_per_exec = max_delay;
      } else {
        num_elements_per_exec = Gcd(max_delay, num_elements_per_exec);
      }

    }
    dist_array_buffer_delay_info_.emplace(std::make_pair(dist_array_buffer_id,
                                                         DistArrayBufferDelayInfo(max_delay,
                                                                                  delay_mode)));
    int32_t dist_array_id = buffer_info.kDistArrayId;
    global_indexed_dist_array_ids_with_buffer_applied.emplace(dist_array_id);
    for (auto helper_dist_array_id : buffer_info.kHelperDistArrayIds) {
      global_indexed_dist_array_ids_with_buffer_applied.emplace(helper_dist_array_id);
    }

    for (auto helper_dist_array_buffer_id : buffer_info.kHelperDistArrayBufferIds) {
      dist_array_buffer_delay_info_.emplace(std::make_pair(helper_dist_array_buffer_id,
                                                           DistArrayBufferDelayInfo(max_delay,
                                                                                    delay_mode)));
    }
  }

  for (auto &dist_array_pair : global_indexed_dist_arrays_) {
    int32_t dist_array_id = dist_array_pair.first;
    if ((global_indexed_dist_array_ids_with_buffer_applied.count(dist_array_id) == 0) &&
        (written_dist_array_ids_.count(dist_array_id) == 0)) {
      read_only_global_indexed_dist_array_ids_.emplace(dist_array_id);
    }
  }

  num_elements_per_exec_ = num_elements_per_exec;
  num_elements_to_exec_ = num_elements_per_exec;
}

AbstractExecForLoop::RunnableStatus
AbstractExecForLoop::GetRunnableStatus() {
  if (skipped_ ||
      (curr_partition_num_elements_executed_ >= curr_partition_length_)) {
    AdvanceClock();
    ComputePartitionIdsAndFindPartitionToExecute();
  }

  if (IsCompleted())
    return AbstractExecForLoop::RunnableStatus::kCompleted;

  if (SkipTimePartition())
    return AbstractExecForLoop::RunnableStatus::kSkip;

  int32_t time_partition_id = GetCurrTimePartitionId();
  if (curr_partition_ == nullptr ||
      curr_partition_->GetLength() == 0) {
    if (!HasRecvedAllTimePartitionedDistArrays(time_partition_id))
      return AbstractExecForLoop::RunnableStatus::kAwaitPredecessor;
    return AbstractExecForLoop::RunnableStatus::kSkip;
  }

  if (!accessed_global_indexed_dist_arrays_.empty()) {
    if (AwaitPredecessorForGlobalIndexedDistArrays()) {
      return AbstractExecForLoop::RunnableStatus::kAwaitPredecessor;
    }
    if (!SkipPrefetch()) {
      if (!HasSentAllPrefetchRequests())
        return AbstractExecForLoop::RunnableStatus::kPrefetchGlobalIndexedDistArrays;
      if (!HasRecvedAllPrefetches())
        return AbstractExecForLoop::RunnableStatus::kAwaitGlobalIndexedDistArrays;
    }
  }

  if (!HasRecvedAllTimePartitionedDistArrays(time_partition_id))
      return AbstractExecForLoop::RunnableStatus::kAwaitPredecessor;
  return AbstractExecForLoop::RunnableStatus::kRunnable;
}

void
AbstractExecForLoop::SentAllPrefetchRequests() {
  prefetch_status_ = PrefetchStatus::kPrefetchSent;
}

void
AbstractExecForLoop::ToSkipPrefetch() {
  prefetch_status_ = PrefetchStatus::kSkipPrefetch;
}

bool
AbstractExecForLoop::SkipPrefetch() const {
  return (prefetch_status_ == PrefetchStatus::kSkipPrefetch);
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
    if (partition == nullptr) {
      auto skipped_iter = skipped_time_partitioned_dist_array_id_map_.find(
          time_partition_id);
      if (skipped_iter == skipped_time_partitioned_dist_array_id_map_.end()) {
        return false;
      }
      auto &skipped_set = skipped_iter->second;
      if (skipped_set.count(dist_array_pair.first) == 0) {
        return false;
      }
    }
  }
  return true;
}

void
AbstractExecForLoop::ComputePrefetchIndices() {
  if (server_prefetch_info_list_iter_
      != server_prefetch_info_list_.end()) {
    server_prefetch_key_map_ptr_ = &(server_prefetch_info_list_iter_->kKeyMap);
    const auto& re_prefetch_dist_array_ids = server_prefetch_info_list_iter_->kRePrefetchDistArrayIds;
    server_prefetch_info_list_iter_++;
    for (auto dist_array_id : re_prefetch_dist_array_ids) {
      auto& to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_id);
      CHECK(!to_create_accessor);
      to_create_accessor = true;
      auto *cache_partition = dist_array_cache_.at(dist_array_id).get();
      cache_partition->ClearCacheAccessor();
    }
  } else {
    CHECK(is_first_time_);
    std::vector<int32_t> dist_array_ids_vec;
    for (const auto &dist_array_pair : accessed_global_indexed_dist_arrays_) {
      dist_array_ids_vec.push_back(dist_array_pair.first);
    }

    if (num_elements_per_exec_ == 0) {
      curr_partition_->ComputePrefetchIndices(
          kPrefetchBatchFuncName,
          dist_array_ids_vec,
          accessed_global_indexed_dist_arrays_,
          global_read_only_var_jl_vals_,
          accumulator_var_syms_,
          &point_prefetch_dist_array_map_,
          0,
          curr_partition_length_);
    } else {
      size_t num_elements_left_in_curr_partition
          = curr_partition_length_ - curr_partition_num_elements_executed_;
      size_t num_elements_to_exec
          = std::min(num_elements_to_exec_, num_elements_left_in_curr_partition);
      curr_partition_->ComputePrefetchIndices(
          kPrefetchBatchFuncName,
          dist_array_ids_vec,
          accessed_global_indexed_dist_arrays_,
          global_read_only_var_jl_vals_,
          accumulator_var_syms_,
          &point_prefetch_dist_array_map_,
          curr_partition_num_elements_executed_,
          num_elements_to_exec);
    }
    std::vector<int64_t> diff_keys;
    std::vector<int32_t> re_prefetch_dist_array_ids;
    for (auto &to_create_accessor_pair : dist_array_cache_buffer_to_create_accessor_) {
      if (to_create_accessor_pair.second) continue;
      auto dist_array_id = to_create_accessor_pair.first;
      auto iter = point_prefetch_dist_array_map_.find(dist_array_id);
      if (iter == point_prefetch_dist_array_map_.end()) continue;

      auto cache_iter = dist_array_cache_.find(dist_array_id);
      if (cache_iter == dist_array_cache_.end()) continue;

      auto *cache_partition = cache_iter->second.get();
      cache_partition->ComputeKeyDiffs(iter->second, &diff_keys);
      if (diff_keys.empty()) {
        point_prefetch_dist_array_map_.erase(iter);
      } else {
        cache_partition->ClearCacheAccessor();
        iter->second = diff_keys;
        re_prefetch_dist_array_ids.emplace_back(dist_array_id);
        to_create_accessor_pair.second = true;
      }
    }

    for (const auto &dist_array_key_pair : point_prefetch_dist_array_map_) {
      int32_t dist_array_id = dist_array_key_pair.first;
      const auto &point_key_vec = dist_array_key_pair.second;
      for (auto key : point_key_vec) {
        int32_t server_id = key % kNumServers;
        server_prefetch_key_map_[server_id][dist_array_id].push_back(key);
      }
    }
    point_prefetch_dist_array_map_.clear();
    if (kIsRepeated) {
      server_prefetch_info_list_.emplace_back(server_prefetch_key_map_,
                                              std::move(re_prefetch_dist_array_ids));
    }
    server_prefetch_key_map_ptr_ = &server_prefetch_key_map_;
  }
}

void
AbstractExecForLoop::ExecuteForLoopPartition() {
  if (curr_partition_num_elements_executed_ == 0) {
    PrepareSpaceDistArrayPartitions();
    PrepareTimeDistArrayPartitions();
  }
  PrepareDistArrayCacheBufferPartitions();
  size_t num_elements_executed = 0;
  bool end_of_partition = false;
  if (num_elements_per_exec_ == 0) {
    curr_partition_->Execute(kLoopBatchFuncName,
                             accessed_dist_arrays_,
                             accessed_dist_array_buffers_,
                             global_read_only_var_jl_vals_,
                             accumulator_var_syms_,
                             0,
                             curr_partition_length_);
    num_elements_executed = curr_partition_length_;
    end_of_partition = true;
    curr_partition_num_elements_executed_ += num_elements_executed;
  } else {
    size_t num_elements_left_in_curr_partition
        = curr_partition_length_ - curr_partition_num_elements_executed_;
    size_t num_elements_to_exec_this_partition
        = std::min(num_elements_to_exec_, num_elements_left_in_curr_partition);
    curr_partition_->Execute(kLoopBatchFuncName,
                             accessed_dist_arrays_,
                             accessed_dist_array_buffers_,
                             global_read_only_var_jl_vals_,
                             accumulator_var_syms_,
                             curr_partition_num_elements_executed_,
                             num_elements_to_exec_this_partition);
    num_elements_executed = num_elements_to_exec_this_partition;
    num_elements_to_exec_ -= num_elements_executed;
    if (num_elements_to_exec_ == 0)
      num_elements_to_exec_ = num_elements_per_exec_;
    curr_partition_num_elements_executed_ += num_elements_executed;
    end_of_partition = (curr_partition_num_elements_executed_ == curr_partition_length_);
  }
  UpdateDistArrayBufferDelayInfo(num_elements_executed, end_of_partition,
                                 LastPartition());
  if (end_of_partition) {
    ClearSpaceDistArrayPartitions();
    ClearTimeDistArrayPartitions();
  }
  ClearDistArrayCacheBufferPartitions();
}

void
AbstractExecForLoop::Skip() {
  skipped_ = true;
  if (LastPartition())
    UpdateDistArrayBufferDelayInfoWhenSkipLastPartition();
  ClearTimeDistArrayPartitions();
  if (LastPartition())
    ClearDistArrayCacheBufferPartitions();
}

void
AbstractExecForLoop::SerializeAndClearPrefetchIds(ExecutorSendBufferMap *send_buffer_map) {
  num_pending_prefetch_requests_ = server_prefetch_key_map_ptr_->size();
  for (const auto &server_point_key_pair : *server_prefetch_key_map_ptr_) {
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
  server_prefetch_key_map_.clear();
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
    auto to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_buffer_id);
    if (!to_create_accessor) continue;

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
    auto to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(
        dist_array_id);
    if (!to_create_accessor) continue;
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
  LOG(INFO) << __func__ << "Start";
  CHECK(time_partitions_cleared_);
  time_partitions_serialized_bytes_ = nullptr;
  time_partitions_serialized_size_ = 0;
  int32_t time_partition_id_to_send = GetTimePartitionIdToSend();
  if (time_partition_id_to_send < 0) return;
  std::unordered_map<int32_t, SendDataBuffer> data_buffers;
  std::vector<int32_t> skipped_dist_array_id_vec;
  for (auto &dist_array_pair : time_partitioned_dist_arrays_) {
    int32_t dist_array_id = dist_array_pair.first;
    auto *dist_array = dist_array_pair.second;
    auto *dist_array_partition = dist_array->GetLocalPartition(
        time_partition_id_to_send);
    if (dist_array_partition == nullptr) {
      skipped_dist_array_id_vec.push_back(dist_array_id);
      continue;
    }
    auto data_buff = dist_array_partition->Serialize();
    if (data_buff.second > 0) {
      data_buffers.emplace(dist_array_id, data_buff);
    } else {
      skipped_dist_array_id_vec.push_back(dist_array_id);
    }
    dist_array->DeletePartition(time_partition_id_to_send);
  }
  skipped_time_partitioned_dist_array_id_map_.erase(time_partition_id_to_send);
  if (data_buffers.empty() && skipped_dist_array_id_vec.empty()) return;
  size_t total_size = sizeof(size_t) + sizeof(size_t) + sizeof(int32_t);
  for (auto &data_buff : data_buffers) {
    total_size += sizeof(int32_t) + data_buff.second.second;
  }
  total_size += sizeof(int32_t) * skipped_dist_array_id_vec.size();
  auto* buffer_bytes = new uint8_t[total_size];
  uint8_t* cursor = buffer_bytes;
  *reinterpret_cast<size_t*>(cursor) = data_buffers.size();
  cursor += sizeof(size_t);
  *reinterpret_cast<size_t*>(cursor) = skipped_dist_array_id_vec.size();
  cursor += sizeof(size_t);
  *reinterpret_cast<int32_t*>(cursor) = time_partition_id_to_send;
  cursor += sizeof(int32_t);
  for (auto &data_buff_pair : data_buffers) {
    int32_t dist_array_id = data_buff_pair.first;
    auto &data_buff = data_buff_pair.second;
    *reinterpret_cast<int32_t*>(cursor) = dist_array_id;
    cursor += sizeof(int32_t);
    if (data_buff.second > 0) {
      memcpy(cursor, data_buff.first, data_buff.second);
      cursor += data_buff.second;
      delete[] data_buff.first;
    }
  }
  memcpy(cursor, skipped_dist_array_id_vec.data(),
         sizeof(int32_t) * skipped_dist_array_id_vec.size());
  cursor += sizeof(int32_t) * skipped_dist_array_id_vec.size();
  time_partitions_serialized_bytes_ = buffer_bytes;
  time_partitions_serialized_size_ = total_size;
  LOG(INFO) << __func__ << "End";
}

void
AbstractExecForLoop::DeserializePipelinedTimePartitions(uint8_t* bytes) {
  LOG(INFO) << __func__ << "Start";
  uint8_t *cursor = bytes;
  size_t num_data_buffers = *reinterpret_cast<const size_t*>(cursor);
  cursor += sizeof(size_t);
  size_t num_skipped_dist_arrays = *reinterpret_cast<const size_t*>(cursor);
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

  if (num_skipped_dist_arrays > 0) {
     auto skipped_set_pair = skipped_time_partitioned_dist_array_id_map_.emplace(
        time_partition_id, std::set<int32_t>());
    CHECK(skipped_set_pair.second);
    auto &skipped_set = skipped_set_pair.first->second;
    for (size_t i = 0; i < num_skipped_dist_arrays; i++) {
      int32_t dist_array_id = *reinterpret_cast<const int32_t*>(cursor);
      cursor += sizeof(int32_t);
      skipped_set.insert(dist_array_id);
    }
  }
  LOG(INFO) << __func__ << "End";
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
    auto *bytes = buff->byte_buff.data();
    auto *cursor = bytes;
    size_t num_dist_arrays = *reinterpret_cast<const size_t*>(cursor);
    cursor += sizeof(size_t);
    for (size_t j = 0; j < num_dist_arrays; j++) {
      int32_t dist_array_id = *reinterpret_cast<const int32_t*>(cursor);
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
AbstractExecForLoop::InitPartitionToExec() {
  skipped_ = false;
  curr_partition_num_elements_executed_ = 0;
  if (curr_partition_ != nullptr) {
    curr_partition_length_ = curr_partition_->GetLength();
  } else {
    curr_partition_length_ = 0;
  }
}

void
AbstractExecForLoop::UpdateDistArrayBufferDelayInfo(size_t num_elements_executed,
                                                    bool end_of_partition,
                                                    bool last_partition) {
  for (auto &delay_info_pair : dist_array_buffer_delay_info_) {
    auto dist_array_buffer_id = delay_info_pair.first;
    auto &delay_info = delay_info_pair.second;
    if (delay_info.kDelayMode == DistArrayBufferDelayMode::kMaxDelay) {
      delay_info.delay += num_elements_executed;
    }

    if (
            (delay_info.kDelayMode == DistArrayBufferDelayMode::kMaxDelay &&
             (delay_info.delay >= delay_info.kMaxDelay ||
              (end_of_partition && last_partition)
              )
         ) ||
            (delay_info.kDelayMode == DistArrayBufferDelayMode::kDefault
             && end_of_partition)
        ) {
      auto& to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_buffer_id);
      to_create_accessor = true;
      auto& dist_array_buffer_info = kDistArrayBufferInfoMap.at(dist_array_buffer_id);
      auto dist_array_id = dist_array_buffer_info.kDistArrayId;
      auto dist_array_to_create_accessor_iter = dist_array_cache_buffer_to_create_accessor_.find(dist_array_id);
      if (dist_array_to_create_accessor_iter != dist_array_cache_buffer_to_create_accessor_.end()) {
        auto& dist_array_to_create_accessor = dist_array_to_create_accessor_iter->second;
        dist_array_to_create_accessor = true;
      }
      for (auto curr_id : dist_array_buffer_info.kHelperDistArrayIds) {
        auto curr_to_create_accessor_iter = dist_array_cache_buffer_to_create_accessor_.find(curr_id);
        if (curr_to_create_accessor_iter == dist_array_cache_buffer_to_create_accessor_.end()) continue;
        auto& curr_to_create_accessor = curr_to_create_accessor_iter->second;
        curr_to_create_accessor = true;
      }
      for (auto curr_id : dist_array_buffer_info.kHelperDistArrayBufferIds) {
        auto& curr_to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(curr_id);
        curr_to_create_accessor = true;
      }
    }
  }

  if (end_of_partition) {
    for (auto& to_create_accessor_pair : dist_array_cache_buffer_to_create_accessor_) {
      auto dist_array_id = to_create_accessor_pair.first;
      if (written_dist_array_ids_.count(dist_array_id) == 1) {
        to_create_accessor_pair.second = true;
      }
    }

    for (auto& dist_array_id : read_only_global_indexed_dist_array_ids_) {
      auto& to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_id);
      to_create_accessor = true;
    }
  }

  for (auto &delay_info_pair : dist_array_buffer_delay_info_) {
    auto dist_array_buffer_id = delay_info_pair.first;
    auto& dist_array_to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_buffer_id);
    if (dist_array_to_create_accessor) {
      delay_info_pair.second.delay = 0;
    }
  }
}

void
AbstractExecForLoop::UpdateDistArrayBufferDelayInfoWhenSkipLastPartition() {
  for (auto& to_create_accessor_pair : dist_array_cache_buffer_to_create_accessor_) {
    auto dist_array_id = to_create_accessor_pair.first;
    if (written_dist_array_ids_.count(dist_array_id) == 1) {
      to_create_accessor_pair.second = false;
    }
  }

  for (auto& dist_array_id : read_only_global_indexed_dist_array_ids_) {
    auto& to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_id);
    to_create_accessor = false;
  }

  for (auto &delay_info_pair : dist_array_buffer_delay_info_) {
    auto dist_array_buffer_id = delay_info_pair.first;
    auto &delay_info = delay_info_pair.second;
    if (delay_info.kDelayMode == DistArrayBufferDelayMode::kMaxDelay) {
      if (delay_info.delay > 0) {
        auto& to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_buffer_id);
        to_create_accessor = true;
        auto& dist_array_buffer_info = kDistArrayBufferInfoMap.at(dist_array_buffer_id);
        auto dist_array_id = dist_array_buffer_info.kDistArrayId;
        auto dist_array_to_create_accessor_iter = dist_array_cache_buffer_to_create_accessor_.find(dist_array_id);
        if (dist_array_to_create_accessor_iter != dist_array_cache_buffer_to_create_accessor_.end()) {
          auto& dist_array_to_create_accessor = dist_array_to_create_accessor_iter->second;
          dist_array_to_create_accessor = true;
        }
        for (auto curr_id : dist_array_buffer_info.kHelperDistArrayIds) {
          auto curr_to_create_accessor_iter = dist_array_cache_buffer_to_create_accessor_.find(curr_id);
          if (curr_to_create_accessor_iter == dist_array_cache_buffer_to_create_accessor_.end()) continue;
          auto& curr_to_create_accessor = curr_to_create_accessor_iter->second;
          curr_to_create_accessor = true;
        }
        for (auto curr_id : dist_array_buffer_info.kHelperDistArrayBufferIds) {
          auto& curr_to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(curr_id);
          curr_to_create_accessor = true;
        }
      } else {
        auto& to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_buffer_id);
        to_create_accessor = false;
        auto& dist_array_buffer_info = kDistArrayBufferInfoMap.at(dist_array_buffer_id);
        auto dist_array_id = dist_array_buffer_info.kDistArrayId;
        auto dist_array_to_create_accessor_iter = dist_array_cache_buffer_to_create_accessor_.find(dist_array_id);
        if (dist_array_to_create_accessor_iter != dist_array_cache_buffer_to_create_accessor_.end()) {
          auto& dist_array_to_create_accessor = dist_array_to_create_accessor_iter->second;
          dist_array_to_create_accessor = false;
        }
        for (auto curr_id : dist_array_buffer_info.kHelperDistArrayIds) {
          auto curr_to_create_accessor_iter = dist_array_cache_buffer_to_create_accessor_.find(curr_id);
          if (curr_to_create_accessor_iter == dist_array_cache_buffer_to_create_accessor_.end()) continue;
          auto& curr_to_create_accessor = curr_to_create_accessor_iter->second;
          curr_to_create_accessor = false;
        }
        for (auto curr_id : dist_array_buffer_info.kHelperDistArrayBufferIds) {
          auto& curr_to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(curr_id);
          curr_to_create_accessor = false;
        }
      }
    } else {
      auto& to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_buffer_id);
      to_create_accessor = false;
    }
  }

  for (auto &delay_info_pair : dist_array_buffer_delay_info_) {
    auto dist_array_buffer_id = delay_info_pair.first;
    auto& dist_array_to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_buffer_id);
    if (dist_array_to_create_accessor) {
      delay_info_pair.second.delay = 0;
    }
  }
}

void
AbstractExecForLoop::PrepareSpaceDistArrayPartitions() {
  int32_t space_partition_id = GetCurrSpacePartitionId();
  for (auto& dist_array_pair : space_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    auto *access_partition = dist_array->GetLocalPartition(space_partition_id);
    access_partition->CreateAccessor();
  }
}

void
AbstractExecForLoop::PrepareTimeDistArrayPartitions() {
  if (time_partitions_cleared_) {
    int32_t time_partition_id = GetCurrTimePartitionId();
    for (auto& dist_array_pair : time_partitioned_dist_arrays_) {
      auto* dist_array = dist_array_pair.second;
      auto *access_partition = dist_array->GetLocalPartition(time_partition_id);
      CHECK(access_partition != nullptr);
      access_partition->CreateAccessor();
    }
    time_partitions_cleared_ = false;
  }
}

void
AbstractExecForLoop::PrepareDistArrayCacheBufferPartitions() {
  for (auto& dist_array_pair : accessed_global_indexed_dist_arrays_) {
    auto dist_array_id = dist_array_pair.first;
    auto& to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_id);
    if (!to_create_accessor) continue;
    to_create_accessor = false;

    auto *cache_partition = dist_array_cache_.at(dist_array_id).get();
    cache_partition->CreateCacheAccessor();
  }
  for (auto& buffer_pair : dist_array_buffers_) {
    auto buffer_id = buffer_pair.first;
    auto &to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(buffer_id);
    if (!to_create_accessor) continue;
    to_create_accessor = false;

    auto* dist_array_buffer = buffer_pair.second;
    auto *buffer_partition = dist_array_buffer->GetBufferPartition();
    buffer_partition->CreateBufferAccessor();
  }
}

void
AbstractExecForLoop::ClearSpaceDistArrayPartitions() {
  int32_t space_partition_id = GetCurrSpacePartitionId();
  for (auto& dist_array_pair : space_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    auto *access_partition = dist_array->GetLocalPartition(space_partition_id);
    access_partition->ClearAccessor();
  }
}

void
AbstractExecForLoop::ClearTimeDistArrayPartitions() {
  if (ToClearTimePartition() && !time_partitions_cleared_) {
    int32_t time_partition_id = GetCurrTimePartitionId();
    for (auto& dist_array_pair : time_partitioned_dist_arrays_) {
      auto* dist_array = dist_array_pair.second;
      auto *access_partition = dist_array->GetLocalPartition(time_partition_id);
      access_partition->ClearAccessor();
    }
    time_partitions_cleared_ = true;
  }
}

void
AbstractExecForLoop::ClearDistArrayCacheBufferPartitions() {
  for (auto& buffer_pair : dist_array_buffers_) {
    auto dist_array_id = buffer_pair.first;
    auto &to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_id);
    if (!to_create_accessor) continue;
    auto* dist_array_buffer = buffer_pair.second;
    auto *buffer_partition = dist_array_buffer->GetBufferPartition();
    buffer_partition->ClearBufferAccessor();
  }

  for (auto& dist_array_pair : accessed_global_indexed_dist_arrays_) {
    auto dist_array_id = dist_array_pair.first;
    auto& to_create_accessor = dist_array_cache_buffer_to_create_accessor_.at(dist_array_id);
    if (!to_create_accessor) continue;
    auto *cache_partition = dist_array_cache_.at(dist_array_id).get();
    cache_partition->ClearCacheAccessor();
  }

  prefetch_status_ = kPrefetchBatchFuncName.empty() ?
                     PrefetchStatus::kSkipPrefetch :
                     PrefetchStatus::kNotPrefetched;
}

}
}
