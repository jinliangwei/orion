#pragma once

#include <orion/bosen/abstract_exec_for_loop.hpp>
#include <glog/logging.h>
#include <algorithm>

namespace orion {
namespace bosen {

class ExecForLoopSpaceTimeOrdered : public AbstractExecForLoop {
 private:
  int32_t kMaxSpacePartitionId {0};
  int32_t kMaxTimePartitionId {0};
  int32_t kNumClocks {0};
  int32_t kNumSpaceSubClocks {0};
  int32_t clock_ {0};
  int32_t space_sub_clock_ {0};
  int32_t curr_space_partition_id_ {0};
  int32_t curr_time_partition_id_ {0};
  int32_t pred_clock_ {0};
  int32_t pred_space_sub_clock_ {0};

 public:
  ExecForLoopSpaceTimeOrdered(
      int32_t executor_id,
      size_t num_executors,
      size_t num_servers,
      int32_t iteration_space_id,
      const int32_t *space_partitioned_dist_array_ids,
      size_t num_space_partitioned_dist_arrays,
      const int32_t *time_partitioned_dist_array_ids,
      size_t num_time_partitioned_dist_arrays,
      const int32_t *global_indexed_dist_array_ids,
      size_t num_gloabl_indexed_dist_arrays,
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
      bool is_repeated);
  virtual ~ExecForLoopSpaceTimeOrdered();

  int32_t GetPredecessor() {
    return (kExecutorId + kNumExecutors - 1) % kNumExecutors;
  }
  int32_t GetSuccessorToNotify() { return (kExecutorId + 1) % kNumExecutors; }
  uint64_t GetNoticeToSuccessor() {
    return (static_cast<uint64_t>(clock_) << 32) | space_sub_clock_;
  }

  int32_t GetTimePartitionIdToSend();

 private:
  void InitClocks();
  bool IsCompleted() { return clock_ == kNumClocks; }
  void AdvanceClock();
  bool LastPartition();
  bool ToClearTimePartition() { return true; }
  void ComputePartitionIdsAndFindPartitionToExecute();
  void ApplyPredecessorNotice(uint64_t clock);
  int32_t GetCurrSpacePartitionId() { return curr_space_partition_id_; }
  int32_t GetCurrTimePartitionId() { return curr_time_partition_id_; }
  bool SkipTimePartition() {
    return curr_time_partition_id_ < 0;
  }
  bool AwaitPredecessorForGlobalIndexedDistArrays() {
    return !(
        (clock_ <= pred_clock_) ||
        (
            (clock_ == pred_clock_ + 1) &&
            (space_sub_clock_ <= pred_space_sub_clock_)
         )
             );
  }
};

}
}
