#include <orion/bosen/exec_for_loop_space_time_ordered.hpp>

namespace orion {
namespace bosen {

ExecForLoopSpaceTimeOrdered::ExecForLoopSpaceTimeOrdered(
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
        dist_array_buffers,
        dist_array_buffer_info_map,
        is_repeated),
    curr_space_partition_id_ (executor_id) {
  auto &meta = iteration_space_->GetMeta();
  auto &max_ids = meta.GetMaxPartitionIds();
  CHECK(max_ids.size() == 2) << "max_ids.size() = " << max_ids.size();
  kMaxSpacePartitionId = max_ids[0];
  kMaxTimePartitionId = max_ids[1];
  kMaxSpaceTimePartitionId = std::max(kMaxSpacePartitionId, kMaxTimePartitionId);
  kNumClocks = 2 * (kMaxSpaceTimePartitionId + 1);
  kNumSpaceSubClocks = (kMaxSpaceTimePartitionId + kNumExecutors) / kNumExecutors;

  LOG(INFO) << __func__ << " kMaxSpacePartitionId = " << kMaxSpacePartitionId
            << " kMaxTimePartitionId = " << kMaxTimePartitionId
            << " kNumClocks = " << kNumClocks
            << " kNumSpaceSubClocks = " << kNumSpaceSubClocks;
}

ExecForLoopSpaceTimeOrdered::~ExecForLoopSpaceTimeOrdered() { }

void
ExecForLoopSpaceTimeOrdered::InitClocks() {
  clock_ = 0;
  space_sub_clock_ = 0;
}

int32_t
ExecForLoopSpaceTimeOrdered::GetTimePartitionIdToSend() {
  if ((curr_time_partition_id_ <= kMaxTimePartitionId) &&
      (curr_time_partition_id_ >= 0)) {
    return curr_time_partition_id_;
  }
  return -1;
}

void
ExecForLoopSpaceTimeOrdered::AdvanceClock() {
  if (clock_ == kNumClocks) return;
  space_sub_clock_++;
  if (space_sub_clock_ == kNumSpaceSubClocks) {
    space_sub_clock_ = 0;
    clock_++;
  }
  //LOG(INFO) << __func__ << " executor_id = " << kExecutorId << " clock = " << clock_
  //          << " space_sub_clock = " << space_sub_clock_;
}

bool
ExecForLoopSpaceTimeOrdered::LastPartition() {
  return (space_sub_clock_ == (kNumSpaceSubClocks - 1)) &&
      (clock_ == (kNumClocks - 1));
}

void
ExecForLoopSpaceTimeOrdered::ComputePartitionIdsAndFindPartitionToExecute() {
  if (clock_ == kNumClocks) return;
  curr_space_partition_id_ = space_sub_clock_ * kNumExecutors + kExecutorId;
  curr_time_partition_id_ = ((kMaxSpaceTimePartitionId + 1) - curr_space_partition_id_
                             + clock_) % (kMaxSpaceTimePartitionId + 1);
  if (clock_ < curr_space_partition_id_ || clock_ > curr_space_partition_id_ + kMaxTimePartitionId) {
    curr_partition_ = nullptr;
  } else {
    curr_partition_ = iteration_space_->GetLocalPartition(curr_space_partition_id_,
                                                          curr_time_partition_id_);
  }
  InitPartitionToExec();
  //LOG(INFO) << __func__ << " executor_id = " << kExecutorId << " curr_space_partition_id = " << curr_space_partition_id_
  //          << " curr_time_partition_id = " << curr_time_partition_id_;
}

void
ExecForLoopSpaceTimeOrdered::ApplyPredecessorNotice(uint64_t clock) {
  pred_clock_ = static_cast<int32_t>(clock >> 32);
  pred_space_sub_clock_ = static_cast<int32_t>(
      clock &
      static_cast<uint64_t>(std::numeric_limits<int>::max()));
}

}
}
