#include <orion/bosen/executor.hpp>
#include <orion/bosen/exec_for_loop_1d.hpp>
#include <orion/bosen/exec_for_loop_space_time_unordered.hpp>
#include <orion/bosen/exec_for_loop_space_time_ordered.hpp>
#include <orion/bosen/dist_array_meta.hpp>

namespace orion {
namespace bosen {

void
Executor::CreateExecForLoop() {
  LOG(INFO) << __func__;
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
     = exec_for_loop_task.time_partitioned_dist_array_ids().data();
  size_t num_time_partitioned_dist_arrays
      = exec_for_loop_task.time_partitioned_dist_array_ids_size();
  const int32_t *global_indexed_dist_array_ids
      = exec_for_loop_task.global_indexed_dist_array_ids().data();
  size_t num_global_indexed_dist_arrays
      = exec_for_loop_task.global_indexed_dist_array_ids_size();
  const int32_t *buffered_dist_array_ids
      = exec_for_loop_task.buffered_dist_array_ids().data();
  size_t num_buffered_dist_arrays
      = exec_for_loop_task.buffered_dist_array_ids_size();
  const int32_t *dist_array_buffer_ids
      = exec_for_loop_task.dist_array_buffer_ids().data();
  const size_t *num_buffers_each_dist_array
      = exec_for_loop_task.num_buffers_each_dist_array().data();

  std::string loop_batch_func_name = exec_for_loop_task.loop_batch_func_name();
  std::string prefetch_batch_func_name = exec_for_loop_task.prefetch_batch_func_name();
  bool is_ordered = exec_for_loop_task.is_ordered();
  CHECK(exec_for_loop_.get() == nullptr);
  switch (parallel_scheme) {
    case ForLoopParallelScheme::k1D:
      {
        exec_for_loop_.reset(
            new ExecForLoop1D(
                kExecutorId,
                kNumExecutors,
                kNumServers,
                iteration_space_id,
                space_partitioned_dist_array_ids,
                num_space_partitioned_dist_arrays,
                global_indexed_dist_array_ids,
                num_global_indexed_dist_arrays,
                buffered_dist_array_ids,
                num_buffered_dist_arrays,
                dist_array_buffer_ids,
                num_buffers_each_dist_array,
                loop_batch_func_name.c_str(),
                prefetch_batch_func_name.c_str(),
                &dist_arrays_,
                &dist_array_buffers_));
        break;
      }
    case ForLoopParallelScheme::kSpaceTime:
      {
        if (!is_ordered) {
          exec_for_loop_.reset(
              new ExecForLoopSpaceTimeUnordered(
                kExecutorId,
                kNumExecutors,
                kNumServers,
                iteration_space_id,
                space_partitioned_dist_array_ids,
                num_space_partitioned_dist_arrays,
                time_partitioned_dist_array_ids,
                num_time_partitioned_dist_arrays,
                global_indexed_dist_array_ids,
                num_global_indexed_dist_arrays,
                buffered_dist_array_ids,
                num_buffered_dist_arrays,
                dist_array_buffer_ids,
                num_buffers_each_dist_array,
                loop_batch_func_name.c_str(),
                prefetch_batch_func_name.c_str(),
                &dist_arrays_,
                &dist_array_buffers_));
        } else {
          exec_for_loop_.reset(
              new ExecForLoopSpaceTimeOrdered(
                kExecutorId,
                kNumExecutors,
                kNumServers,
                iteration_space_id,
                space_partitioned_dist_array_ids,
                num_space_partitioned_dist_arrays,
                time_partitioned_dist_array_ids,
                num_time_partitioned_dist_arrays,
                global_indexed_dist_array_ids,
                num_global_indexed_dist_arrays,
                buffered_dist_array_ids,
                num_buffered_dist_arrays,
                dist_array_buffer_ids,
                num_buffers_each_dist_array,
                loop_batch_func_name.c_str(),
                prefetch_batch_func_name.c_str(),
                &dist_arrays_,
                &dist_array_buffers_));
        }
        break;
      }
    default:
      LOG(FATAL) << "unknown parallel_scheme = "
                 << static_cast<int32_t>(parallel_scheme);
  }
}

void
Executor::CheckAndExecuteForLoop(bool next_partition) {
  if (next_partition) exec_for_loop_->FindNextToExecPartition();
  bool stop_search = false;
  while (true) {
    auto runnable_status = exec_for_loop_->GetCurrPartitionRunnableStatus();
    switch (runnable_status) {
      case AbstractExecForLoop::RunnableStatus::kRunnable:
        {
          auto* exec_for_loop_ptr = exec_for_loop_.get();
          auto cpp_func = std::bind(
              &AbstractExecForLoop::ExecuteForLoopPartition,
              exec_for_loop_ptr);
          exec_cpp_func_task_.func = cpp_func;
          exec_cpp_func_task_.label = TaskLabel::kExecForLoopPartition;
          julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
          stop_search = true;
        }
        break;
      case AbstractExecForLoop::RunnableStatus::kPrefetchGlobalIndexedDistArrays:
        {
          stop_search = true;
        }
        break;
      case AbstractExecForLoop::RunnableStatus::kAwaitGlobalIndexedDistArrays:
        {
          event_handler_.SetToReadOnly(&prt_poll_conn_);
          RequestExecForLoopGlobalIndexedDistArrays();
          stop_search = true;
        }
        break;
      case AbstractExecForLoop::RunnableStatus::kAwaitPredecessor:
        {
          event_handler_.SetToReadOnly(&prt_poll_conn_);
          RequestExecForLoopPipelinedTimePartitions();
          stop_search = true;
        }
        break;
      case AbstractExecForLoop::RunnableStatus::kCompleted:
        {
          LOG(INFO) << "completed!";
          stop_search = true;
          if (kNumExecutors > 1)
            SendPredCompletion();
          bool requested = CheckAndRequestExecForLoopPredecesorCompletion();
          if (requested) {
            event_handler_.SetToReadOnly(&prt_poll_conn_);
          } else {
            ExecForLoopAck();
          }
        }
        break;
      case AbstractExecForLoop::RunnableStatus::kSkip:
        {
          stop_search = true;
          bool serialize_dist_array_time_partitions
              = CheckAndSerializeDistArrayTimePartitions();
          if (!serialize_dist_array_time_partitions) {
            stop_search = false;
          }
        }
        break;
      default:
        LOG(FATAL) << "unknown status";
    }
    if (stop_search) break;
    exec_for_loop_->FindNextToExecPartition();
  }
}

bool
Executor::CheckAndSerializeGlobalIndexedDistArrays() {
  bool send_global_indexed_dist_arrays
      = exec_for_loop_->SendGlobalIndexedDistArrays();
  if (!send_global_indexed_dist_arrays) return false;

  auto* exec_for_loop_ptr = exec_for_loop_.get();
  auto cpp_func = std::bind(
      &AbstractExecForLoop::SerializeAndClearGlobalPartitionedDistArrays,
      exec_for_loop_ptr);
  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kSerializeGlobalIndexedDistArrays;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
  return true;
}

bool
Executor::CheckAndSerializeDistArrayTimePartitions() {
  int32_t successor_to_send
      = exec_for_loop_->GetSuccessorToNotify();
  int32_t time_partition_id_to_send
      = exec_for_loop_->GetTimePartitionIdToSend();

  if (successor_to_send < 0 || successor_to_send == kExecutorId
      || time_partition_id_to_send < 0) return false;

  auto* exec_for_loop_ptr = exec_for_loop_.get();
  auto cpp_func = std::bind(
      &AbstractExecForLoop::SerializeAndClearPipelinedTimePartitions,
      exec_for_loop_ptr);
  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kSerializeDistArrayTimePartitions;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
  return true;
}

void
Executor::SendGlobalIndexedDistArrays() {
}

void
Executor::SendPipelinedTimePartitions() {
  int32_t successor_to_send
      = exec_for_loop_->GetSuccessorToNotify();
  auto send_data_buff
      = exec_for_loop_->GetAndResetSerializedTimePartitions();
  if (send_data_buff.second == 0) return;
  uint64_t notice = exec_for_loop_->GetNoticeToSuccessor();
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgPipelinedTimePartitions>(
      &send_buff_, send_data_buff.second, notice);
  send_buff_.set_next_to_send(send_data_buff.first, send_data_buff.second, true);
  Send(&executor_conn_[successor_to_send], executor_[successor_to_send].get());
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
}

void
Executor::SendPredCompletion() {
  int32_t successor_to_send
      = exec_for_loop_->GetSuccessorToNotify();
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgPredCompletion>(
      &send_buff_);
  Send(&executor_conn_[successor_to_send], executor_[successor_to_send].get());
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
}

void
Executor::ExecForLoopAck() {
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgExecForLoopAck>(
      &send_buff_);
  Send(&master_poll_conn_, &master_);
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
  exec_for_loop_.reset();
}

void
Executor::RequestExecForLoopGlobalIndexedDistArrays() {
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRequestExecForLoopGlobalIndexedDistArrays>(
      &send_buff_);
  Send(&prt_poll_conn_, prt_pipe_conn_.get());
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
}

void
Executor::RequestExecForLoopPipelinedTimePartitions() {
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRequestExecForLoopPipelinedTimePartitions>(
      &send_buff_);
  Send(&prt_poll_conn_, prt_pipe_conn_.get());
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
}

bool
Executor::CheckAndRequestExecForLoopPredecesorCompletion() {
  if (kNumExecutors == 1) return false;
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRequestExecForLoopPredecessorCompletion>(
      &send_buff_);
  Send(&prt_poll_conn_, prt_pipe_conn_.get());
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
  return true;
}

}
}
