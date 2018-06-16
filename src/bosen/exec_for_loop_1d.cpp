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
        dist_array_buffers,
        dist_array_buffer_info_map,
        is_repeated) {
  auto &meta = iteration_space_->GetMeta();
  auto &max_ids = meta.GetMaxPartitionIds();
  CHECK(max_ids.size() == 1) << "max_ids.size() = " << max_ids.size()
                             << " iteration_space_id = " << iteration_space_id;
  kMaxPartitionId = max_ids[0];
  kNumClocks = (kMaxPartitionId + kNumExecutors) / kNumExecutors;
}

ExecForLoop1D::~ExecForLoop1D() { }

void
ExecForLoop1D::InitClocks() {
  clock_ = 0;
}

void
ExecForLoop1D::AdvanceClock() {
  if (clock_ == kNumClocks) return;
  clock_++;
}

bool
ExecForLoop1D::LastPartition() {
  return (clock_ == (kNumClocks - 1));
}

void
ExecForLoop1D::ComputePartitionIdsAndFindPartitionToExecute() {
  if (clock_ == kNumClocks) return;
  curr_partition_id_ = clock_ * kNumExecutors + kExecutorId;
  curr_partition_ = iteration_space_->GetLocalPartition(curr_partition_id_);
  InitPartitionToExec();
}

}
}
