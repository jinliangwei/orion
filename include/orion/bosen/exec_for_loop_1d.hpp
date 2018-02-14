#pragma once

#include <orion/bosen/abstract_exec_for_loop.hpp>

namespace orion {
namespace bosen {

class ExecForLoop1D : public AbstractExecForLoop {
 private:
  int32_t kMaxPartitionId {0};
  int32_t kNumClocks {0};
  int32_t clock_ {0};
  int32_t curr_partition_id_ {0};

 public:
  ExecForLoop1D(
      int32_t executor_id,
      size_t num_executors,
      size_t num_servers,
      int32_t iteration_space_id,
      const int32_t *space_partitioned_dist_array_ids,
      size_t num_space_partitioned_dist_arrays,
      const int32_t *global_indexed_dist_array_ids,
      size_t num_gloabl_indexed_dist_arrays,
      const int32_t *dist_array_buffer_ids,
      size_t num_dist_array_buffers,
      const int32_t *written_dist_array_ids,
      size_t num_written_dist_array_ids,
      const char* loop_batch_func_name,
      const char *prefetch_batch_func_name,
      std::unordered_map<int32_t, DistArray> *dist_arrays,
      std::unordered_map<int32_t, DistArray> *dist_array_buffers);
  virtual ~ExecForLoop1D();

  void FindNextToExecPartition();
  AbstractExecForLoop::RunnableStatus GetCurrPartitionRunnableStatus();
  int32_t GetTimePartitionIdToSend() { return -1; }
  void ApplyPredecessorNotice(uint64_t clock) { }
  int32_t GetPredecessor() { return -1; }
  int32_t GetSuccessorToNotify() { return -1; }
  uint64_t GetNoticeToSuccessor() { return 0; }
  void PrepareToExecCurrPartition();
  void ClearCurrPartition();

 private:
  void ComputePartitionIdsAndFindPartitionToExecute();
};

}
}
