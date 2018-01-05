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
      const int32_t *buffered_dist_array_ids,
      size_t num_buffered_dist_arrays,
      const int32_t *dist_array_buffer_ids,
      const size_t *num_buffers_each_dist_array,
      const char* loop_batch_func_name,
      const char *prefetch_batch_func_name,
      std::unordered_map<int32_t, DistArray> *dist_arrays,
      std::unordered_map<int32_t, DistArray> *dist_array_buffers);
  virtual ~ExecForLoopSpaceTimeOrdered();

  void FindNextToExecPartition();
  AbstractExecForLoop::RunnableStatus GetCurrPartitionRunnableStatus();
  int32_t GetTimePartitionIdToSend();
  void ApplyPredecessorNotice(uint64_t clock);
  int32_t GetSuccessorToNotify() { return (kExecutorId + 1) % kNumExecutors; }
  uint64_t GetNoticeToSuccessor() {
    return (static_cast<uint64_t>(clock_) << 32) || space_sub_clock_; }
  void PrepareToExecCurrPartition();

 private:
  void ComputePartitionIdsAndFindPartitionToExecute();
};

}
}
