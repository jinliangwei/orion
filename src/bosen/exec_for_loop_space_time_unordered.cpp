#include <orion/bosen/exec_for_loop_space_time_unordered.hpp>

namespace orion {
namespace bosen {

ExecForLoopSpaceTimeUnordered::ExecForLoopSpaceTimeUnordered(
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
    AbstractExecForLoop(
        executor_id,
        num_executors,
        num_servers,
        iteration_space_id,
        space_partitioned_dist_array_ids,
        num_space_partitioned_dist_arrays,
        time_partitioned_dist_array_ids,
        num_time_partitioned_dist_arrays,
        global_indexed_dist_array_ids,
        num_global_indexed_dist_arrays,
        dist_array_buffer_ids,
        num_dist_array_buffers,
        written_dist_array_ids,
        num_written_dist_array_ids,
        accessed_dist_array_ids,
        num_accessed_dist_arrays,
        global_read_only_var_vals,
        num_global_read_only_var_vals,
        accumulator_var_syms,
        num_accumulator_var_syms,
        loop_batch_func_name,
        prefetch_batch_func_name,
        dist_arrays,
        dist_array_buffers),
    curr_space_partition_id_ (executor_id) {
  auto &meta = iteration_space_->GetMeta();
  auto &max_ids = meta.GetMaxPartitionIds();
  CHECK(max_ids.size() == 2) << "max_ids.size() = " << max_ids.size();
  kMaxSpacePartitionId = max_ids[0];
  kMaxTimePartitionId = max_ids[1];
  kNumClocks = kNumExecutors;
  kNumSpaceSubClocks = (kMaxSpacePartitionId + kNumExecutors) / kNumExecutors;
  kNumTimeSubClocks = (kMaxTimePartitionId + kNumExecutors) / kNumExecutors;

  LOG(INFO) << "max_space_partition_id = " << kMaxSpacePartitionId
            << " max_time_partition_id = " << kMaxTimePartitionId
            << " num_space_sub_clocks = " << kNumSpaceSubClocks
            << " num_time_sub_clocks = " << kNumTimeSubClocks;

  clock_ = 0;
  space_sub_clock_ = 0;
  time_sub_clock_ = 0;

  ComputePartitionIdsAndFindPartitionToExecute();
}

ExecForLoopSpaceTimeUnordered::~ExecForLoopSpaceTimeUnordered() { }

AbstractExecForLoop::RunnableStatus
ExecForLoopSpaceTimeUnordered::GetCurrPartitionRunnableStatus() {
  if (clock_ == kNumClocks) return AbstractExecForLoop::RunnableStatus::kCompleted;

  if (curr_time_partition_id_ > kMaxTimePartitionId)
    return AbstractExecForLoop::RunnableStatus::kSkip;

  if (curr_partition_ == nullptr) {
    if (!HasRecvedAllTimePartitionedDistArrays(curr_time_partition_id_))
      return AbstractExecForLoop::RunnableStatus::kAwaitPredecessor;
    return AbstractExecForLoop::RunnableStatus::kSkip;
  }

  if (!global_indexed_dist_arrays_.empty()) {
    if (!(
            (clock_ <= pred_clock_) ||
            ((clock_ == pred_clock_ + 1) && (time_sub_clock_ <= pred_time_sub_clock_))
          )) return AbstractExecForLoop::RunnableStatus::kAwaitPredecessor;
    if (!SkipPrefetch()) {
      if (!HasSentAllPrefetchRequests()) return AbstractExecForLoop::RunnableStatus::kPrefetchGlobalIndexedDistArrays;
      if (!HasRecvedAllPrefetches()) return AbstractExecForLoop::RunnableStatus::kAwaitGlobalIndexedDistArrays;
    }
  }

  if (!HasRecvedAllTimePartitionedDistArrays(curr_time_partition_id_))
      return AbstractExecForLoop::RunnableStatus::kAwaitPredecessor;

  return AbstractExecForLoop::RunnableStatus::kRunnable;
}

void
ExecForLoopSpaceTimeUnordered::FindNextToExecPartition() {
  if (clock_ == kNumClocks) return;
  space_sub_clock_++;
  if (space_sub_clock_ == kNumSpaceSubClocks) {
    space_sub_clock_ = 0;
    time_sub_clock_++;
    if (time_sub_clock_ == kNumTimeSubClocks) {
      time_sub_clock_ = 0;
      clock_++;
    }
  }
  if (clock_ == kNumClocks) return;
  ComputePartitionIdsAndFindPartitionToExecute();
}

int32_t
ExecForLoopSpaceTimeUnordered::GetTimePartitionIdToSend() {
  if (space_sub_clock_ == kNumSpaceSubClocks - 1) {
    return curr_time_partition_id_;
  }
  return -1;
}

void
ExecForLoopSpaceTimeUnordered::ApplyPredecessorNotice(uint64_t clock) {
  pred_clock_ = static_cast<int32_t>(clock >> 32);
  pred_time_sub_clock_ = static_cast<int32_t>(clock &
                                              static_cast<uint64_t>(std::numeric_limits<int>::max()));
}

void
ExecForLoopSpaceTimeUnordered::ComputePartitionIdsAndFindPartitionToExecute() {
  curr_time_partition_id_ = time_sub_clock_ * kNumExecutors
                            + (clock_ + kExecutorId) % kNumClocks;
  curr_space_partition_id_ = space_sub_clock_ * kNumExecutors + kExecutorId;
  curr_partition_ = iteration_space_->GetLocalPartition(curr_space_partition_id_,
                                                        curr_time_partition_id_);
}

void
ExecForLoopSpaceTimeUnordered::PrepareToExecCurrPartition() {
  if (curr_partition_prepared_) return;
  LOG(INFO) << __func__ << " space = " << curr_space_partition_id_
            << " time = " << curr_time_partition_id_;
  for (auto& dist_array_pair : space_partitioned_dist_arrays_) {
    LOG(INFO) << __func__ << " CreateAccessor for sp " << dist_array_pair.first;
    auto* dist_array = dist_array_pair.second;
    auto *access_partition = dist_array->GetLocalPartition(curr_space_partition_id_);
    access_partition->CreateAccessor();
  }

  for (auto& dist_array_pair : time_partitioned_dist_arrays_) {
    LOG(INFO) << __func__ << " CreateAccessor for tp " << dist_array_pair.first;
    auto* dist_array = dist_array_pair.second;
    auto *access_partition = dist_array->GetLocalPartition(curr_time_partition_id_);
    access_partition->CreateAccessor();
  }

  for (auto& buffer_pair : dist_array_buffers_) {
    LOG(INFO) << __func__ << " CreateBufferAccessor for " << buffer_pair.first;
    auto* dist_array_buffer = buffer_pair.second;
    auto *buffer_partition = dist_array_buffer->GetBufferPartition();
    buffer_partition->CreateBufferAccessor();
  }
  curr_partition_prepared_ = true;
}

void
ExecForLoopSpaceTimeUnordered::ClearCurrPartition() {
  if (!curr_partition_prepared_) return;
  for (auto& dist_array_pair : space_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    auto *access_partition = dist_array->GetLocalPartition(curr_space_partition_id_);
    access_partition->ClearAccessor();
  }

  for (auto& dist_array_pair : time_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    auto *access_partition = dist_array->GetLocalPartition(curr_time_partition_id_);
    access_partition->ClearAccessor();
  }

  for (auto& dist_array_pair : global_indexed_dist_arrays_) {
    auto dist_array_id = dist_array_pair.first;
    auto *cache_partition = dist_array_cache_.at(dist_array_id).get();
    cache_partition->ClearCacheAccessor();
  }

  for (auto& buffer_pair : dist_array_buffers_) {
    auto* dist_array_buffer = buffer_pair.second;
    auto *buffer_partition = dist_array_buffer->GetBufferPartition();
    buffer_partition->ClearBufferAccessor();
  }
  curr_partition_prepared_ = false;
  dist_array_cache_prepared_ = false;
  prefetch_status_ = SkipPrefetch() ? PrefetchStatus::kSkipPrefetch : PrefetchStatus::kNotPrefetched;
}

}
}
