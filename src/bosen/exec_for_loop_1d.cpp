#include <orion/bosen/exec_for_loop_1d.hpp>

namespace orion {
namespace bosen {

ExecForLoop1D::ExecForLoop1D(
      int32_t executor_id,
      size_t num_executors,
      size_t num_servers,
      int32_t iteration_space_id,
      const int32_t *space_partitioned_dist_array_ids,
      size_t num_space_partitioned_dist_arrays,
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
        nullptr,
        0,
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
        dist_array_buffers) {
  auto &meta = iteration_space_->GetMeta();
  auto &max_ids = meta.GetMaxPartitionIds();
  CHECK(max_ids.size() == 1) << "max_ids.size() = " << max_ids.size();
  kMaxPartitionId = max_ids[0];
  kNumClocks = (kMaxPartitionId + kNumExecutors) / kNumExecutors;

  clock_ = 0;
  ComputePartitionIdsAndFindPartitionToExecute();
}

ExecForLoop1D::~ExecForLoop1D() { }

AbstractExecForLoop::RunnableStatus
ExecForLoop1D::GetCurrPartitionRunnableStatus() {
  if (clock_ == kNumClocks) return AbstractExecForLoop::RunnableStatus::kCompleted;
  if (curr_partition_ == nullptr) return AbstractExecForLoop::RunnableStatus::kSkip;
  if (!global_indexed_dist_arrays_.empty() && !SkipPrefetch()) {
    if (!HasSentAllPrefetchRequests()) return AbstractExecForLoop::RunnableStatus::kPrefetchGlobalIndexedDistArrays;
    if (!HasRecvedAllPrefetches()) return AbstractExecForLoop::RunnableStatus::kAwaitGlobalIndexedDistArrays;
  }
  return AbstractExecForLoop::RunnableStatus::kRunnable;
}

void
ExecForLoop1D::FindNextToExecPartition() {
  if (clock_ == kNumClocks) return;
  clock_++;
  if (clock_ == kNumClocks) return;
  ComputePartitionIdsAndFindPartitionToExecute();
}

void
ExecForLoop1D::ComputePartitionIdsAndFindPartitionToExecute() {
  curr_partition_id_ = clock_ * kNumExecutors + kExecutorId;
  curr_partition_ = iteration_space_->GetLocalPartition(curr_partition_id_);
  //LOG(INFO) << __func__ << " curr_partition_id = " << curr_partition_id_
  //<< " curr_partition_ = " << (void*) curr_partition_;
}

void
ExecForLoop1D::PrepareToExecCurrPartition() {
  if (curr_partition_prepared_) return;
  LOG(INFO) << __func__;
  for (auto& dist_array_pair : space_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    auto *access_partition = dist_array->GetLocalPartition(curr_partition_id_);
    access_partition->CreateAccessor();
  }

  for (auto& buffer_pair : dist_array_buffers_) {
    LOG(INFO) << "prepare buffer " << buffer_pair.first;
    auto* dist_array_buffer = buffer_pair.second;
    auto *buffer_partition = dist_array_buffer->GetBufferPartition();
    buffer_partition->CreateBufferAccessor();
  }
  curr_partition_prepared_ = true;
}

void
ExecForLoop1D::ClearCurrPartition() {
  if (!curr_partition_prepared_) return;
  LOG(INFO) << __func__;
  for (auto& dist_array_pair : space_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    auto *access_partition = dist_array->GetLocalPartition(curr_partition_id_);
    access_partition->ClearAccessor();
    LOG(INFO) << "cleared dist_array " << dist_array_pair.first;
  }

  for (auto& dist_array_pair : global_indexed_dist_arrays_) {
    auto dist_array_id = dist_array_pair.first;
    auto *cache_partition = dist_array_cache_.at(dist_array_id).get();
    cache_partition->ClearCacheAccessor();
    LOG(INFO) << "cleared dist_array cache " << dist_array_pair.first;
  }

  for (auto& buffer_pair : dist_array_buffers_) {
    auto* dist_array_buffer = buffer_pair.second;
    auto *buffer_partition = dist_array_buffer->GetBufferPartition();
    buffer_partition->ClearBufferAccessor();
    LOG(INFO) << "cleared dist_array buffer " << buffer_pair.first;
  }
  curr_partition_prepared_ = false;
  dist_array_cache_prepared_ = false;
  prefetch_status_ = SkipPrefetch() ? PrefetchStatus::kSkipPrefetch : PrefetchStatus::kNotPrefetched;
}

}
}
