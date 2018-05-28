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
  kNumClocks = kNumExecutors;
  kNumSpaceSubClocks = (kMaxSpacePartitionId + kNumExecutors) / kNumExecutors;
  kNumTimeSubClocks = (kMaxTimePartitionId + kNumExecutors) / kNumExecutors;

  LOG(INFO) << "max_space_partition_id = " << kMaxSpacePartitionId
            << " max_time_partition_id = " << kMaxTimePartitionId
            << " num_space_sub_clocks = " << kNumSpaceSubClocks
            << " num_time_sub_clocks = " << kNumTimeSubClocks;
}

ExecForLoopSpaceTimeUnordered::~ExecForLoopSpaceTimeUnordered() { }

void
ExecForLoopSpaceTimeUnordered::ApplyPredecessorNotice(uint64_t clock) {
  pred_clock_ = static_cast<int32_t>(clock >> 32);
  pred_time_sub_clock_ = static_cast<int32_t>(clock &
                                              static_cast<uint64_t>(std::numeric_limits<int>::max()));
}

int32_t
ExecForLoopSpaceTimeUnordered::GetTimePartitionIdToSend() {
  if ((curr_partition_num_elements_executed_ == curr_partition_length_)
      && (space_sub_clock_ == (kNumSpaceSubClocks - 1))) {
    return curr_time_partition_id_;
  }
  return -1;
}

void
ExecForLoopSpaceTimeUnordered::InitClocks() {
  clock_ = 0;
  space_sub_clock_ = 0;
  time_sub_clock_ = 0;
}

void
ExecForLoopSpaceTimeUnordered::AdvanceClock() {
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
}

bool
ExecForLoopSpaceTimeUnordered::LastPartition() {
  return (space_sub_clock_ == (kNumSpaceSubClocks - 1)) &&
      (time_sub_clock_ == (kNumTimeSubClocks - 1)) &&
      (clock_ == (kNumClocks - 1));
}

void
ExecForLoopSpaceTimeUnordered::ComputePartitionIdsAndFindPartitionToExecute() {
  if (clock_ == kNumClocks) return;
  curr_time_partition_id_ = time_sub_clock_ * kNumExecutors
                            + (clock_ + kExecutorId) % kNumClocks;
  curr_space_partition_id_ = space_sub_clock_ * kNumExecutors + kExecutorId;
  curr_partition_ = iteration_space_->GetLocalPartition(curr_space_partition_id_,
                                                        curr_time_partition_id_);
  InitPartitionToExec();
}

}
}
