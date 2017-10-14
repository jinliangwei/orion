#include <orion/bosen/executor.hpp>
#include <orion/bosen/exec_for_loop_1d.hpp>
#include <orion/bosen/exec_for_loop_space_time.hpp>
#include <orion/bosen/dist_array_meta.hpp>

namespace orion {
namespace bosen {

void
Executor::ExecForLoop() {
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());
  task::ExecForLoop exec_for_loop_task;
  exec_for_loop_task.ParseFromString(task_str);
  int32_t iteration_space_id = exec_for_loop_task.iteration_space_id();
  ForLoopParallelScheme parallel_scheme
      = static_cast<ForLoopParallelScheme>(exec_for_loop_task.parallel_scheme());
  const int32_t *space_partitioned_dist_array_ids
     = exec_for_loop_task.space_partitioned_dist_array_ids().data();
  size_t num_space_partitioned_dist_arrays
      = exec_for_loop_task.space_partitioned_dist_array_ids_size();
  const int32_t *time_partitioned_dist_array_ids
     = exec_for_loop_task.space_partitioned_dist_array_ids().data();
  size_t num_time_partitioned_dist_arrays
      = exec_for_loop_task.time_partitioned_dist_array_ids_size();
  const int32_t *global_indexed_dist_array_ids
      = exec_for_loop_task.global_indexed_dist_array_ids().data();
  size_t num_global_indexed_dist_arrays
      = exec_for_loop_task.global_indexed_dist_array_ids_size();

  std::string loop_batch_func_name = exec_for_loop_task.loop_batch_func_name();
  bool is_ordered = exec_for_loop_task.is_ordered();
  CHECK(exec_for_loop_ == nullptr);

  switch (parallel_scheme) {
    case ForLoopParallelScheme::k1D:
      {
        exec_for_loop_ = new ExecForLoop1D();
        break;
      }
    case ForLoopParallelScheme::kSpaceTime:
      {
        exec_for_loop_ = new ExecForLoopSpaceTime(
            iteration_space_id,
            space_partitioned_dist_array_ids,
            num_space_partitioned_dist_arrays,
            time_partitioned_dist_array_ids,
            num_time_partitioned_dist_arrays,
            global_indexed_dist_array_ids,
            num_global_indexed_dist_arrays,
            loop_batch_func_name.c_str(),
            is_ordered,
            dist_arrays_);
        break;
      }
    default:
      LOG(FATAL) << "unknown parallel_scheme = "
                 << static_cast<int32_t>(parallel_scheme);
  }

}

}
}
