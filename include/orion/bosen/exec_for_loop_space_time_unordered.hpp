#pragma once

#include <orion/bosen/abstract_exec_for_loop.hpp>
#include <glog/logging.h>
#include <algorithm>

namespace orion {
namespace bosen {

// 1. The iteration space is padded so that both dimensions are multiple
// of kNumExecutors, say, M by N;
// 2. Divide the iteration space into M * N blocks, each of which is
// kNumExecutors * kNumExecutors
// 3. Within each clock, each executor executes one partition from
// each block, space dimension first, then time dimension, so that
// time partitions can be pipelined

class ExecForLoopSpaceTimeUnordered : public AbstractExecForLoop {
 private:
  int32_t kMaxSpacePartitionId {0};
  int32_t kMaxTimePartitionId {0};
  int32_t kNumClocks {0};
  int32_t kNumSpaceSubClocks {0};
  int32_t kNumTimeSubClocks {0};
  int32_t clock_ {0};
  int32_t space_sub_clock_ {0};
  int32_t time_sub_clock_ {0};
  int32_t curr_space_partition_id_ {0};
  int32_t curr_time_partition_id_ {0};
  int32_t pred_clock_ {0};
  int32_t pred_time_sub_clock_ {0};

 public:
  ExecForLoopSpaceTimeUnordered(
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
      std::unordered_map<int32_t, DistArray> *dist_array_buffers);
  virtual ~ExecForLoopSpaceTimeUnordered();

  void FindNextToExecPartition();
  AbstractExecForLoop::RunnableStatus GetCurrPartitionRunnableStatus();
  int32_t GetTimePartitionIdToSend();
  void ApplyPredecessorNotice(uint64_t clock);
  int32_t GetPredecessor() { return (kExecutorId + 1) % kNumExecutors; }
  int32_t GetSuccessorToNotify() { return (kExecutorId + kNumExecutors - 1) % kNumExecutors; }
  uint64_t GetNoticeToSuccessor() {
    return (static_cast<uint64_t>(clock_) << 32) || time_sub_clock_; }
  void PrepareToExecCurrPartition();
  void ClearCurrPartition();

 private:
  void ComputePartitionIdsAndFindPartitionToExecute();
};

}
}
