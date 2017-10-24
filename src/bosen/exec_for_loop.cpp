#include <orion/bosen/executor.hpp>
#include <orion/bosen/exec_for_loop_1d.hpp>
#include <orion/bosen/exec_for_loop_space_time_unordered.hpp>
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

  std::string loop_batch_func_name = exec_for_loop_task.loop_batch_func_name();
  bool is_ordered = exec_for_loop_task.is_ordered();
  CHECK(exec_for_loop_.get() == nullptr);

  switch (parallel_scheme) {
    case ForLoopParallelScheme::k1D:
      {
        exec_for_loop_.reset(new ExecForLoop1D());
        break;
      }
    case ForLoopParallelScheme::kSpaceTime:
      {
        if (!is_ordered) {
          exec_for_loop_.reset(new ExecForLoopSpaceTimeUnordered(
              kId,
              kNumExecutors,
              iteration_space_id,
              space_partitioned_dist_array_ids,
              num_space_partitioned_dist_arrays,
              time_partitioned_dist_array_ids,
              num_time_partitioned_dist_arrays,
              global_indexed_dist_array_ids,
              num_global_indexed_dist_arrays,
              loop_batch_func_name.c_str(),
              dist_arrays_));
        } else {
          LOG(FATAL) << "unsupported!";
        }
        break;
      }
    default:
      LOG(FATAL) << "unknown parallel_scheme = "
                 << static_cast<int32_t>(parallel_scheme);
  }
}

bool
Executor::CheckAndExecuteForLoopUntilNotRunnable() {
  bool exec_is_scheduled = false;
  bool waiting_for_peers = false;
  bool completed = false;
  while (!exec_is_scheduled
         && !waiting_for_peers
         && !completed) {
    exec_is_scheduled = CheckAndExecuteForLoop(&waiting_for_peers,
                                               &completed);
  }

  if (waiting_for_peers) {
    event_handler_.SetToReadOnly(&prt_poll_conn_);
    RequestDistArrayData();
  }
  if (completed) {
    exec_for_loop_.reset();
  }
  return completed;
}

// return true if it some work has been launched else
// set waiting_for_peers to true if some deps is missing from peers
// return false when 1) some deps is missing; or 2) no work is left to do in this clock, so I advanced to the next clock;
// or 3) I completed all works in this loop
bool
Executor::CheckAndExecuteForLoop(bool *waiting_for_peers, bool *completed) {
  bool exec_is_scheduled = false;
  *waiting_for_peers = false;
  *completed = false;

  if (dynamic_cast<ExecForLoopSpaceTimeUnordered*>(exec_for_loop_.get()) == nullptr) {
    LOG(FATAL) << "I don't yet support this";
    return exec_is_scheduled;
  }
  auto *exec_for_loop_space_time
      = dynamic_cast<ExecForLoopSpaceTimeUnordered*>(exec_for_loop_.get());
  int32_t space_id = 0, time_id = 0;
  AbstractDistArrayPartition* partition_to_exec = exec_for_loop_space_time->GetNextPartitionToExec(
      &space_id, &time_id);
  const auto &completed_time_partitions = exec_for_loop_space_time->GetCompletedTimePartitions();
  if (!completed_time_partitions.empty()) {
    for (auto time_partition_to_send : completed_time_partitions) {
      ExecForLoopSendResults(time_partition_to_send);
    }
    exec_for_loop_space_time->ClearCompletedTimePartitions();
  }
  exec_for_loop_space_time->SetCurrRunningTimePartitionId(time_id);

  if (partition_to_exec == nullptr) {
    *completed = true;
    return exec_is_scheduled;
  }

  auto runnable_status = exec_for_loop_space_time->GetRunnableStatus(time_id);
  switch (runnable_status) {
    case ExecForLoopSpaceTimeUnordered::RunnableStatus::kRunnable:
      {
        exec_for_loop_space_time->PrepareToExecCurrentTile(time_id);
        if (!exec_for_loop_space_time->IsCompleted()) {
          auto &loop_func_name = exec_for_loop_space_time->GetExecLoopFuncName();
          ExecuteForLoopTile(partition_to_exec, loop_func_name);
          exec_is_scheduled = true;
        }
      }
      break;
    case ExecForLoopSpaceTimeUnordered::RunnableStatus::kPrefretchGlobalIndexedDep:
      {
        LOG(FATAL) << "I can't yet deal with prefetching global dep";
        *waiting_for_peers = true;
      }
      break;
    case ExecForLoopSpaceTimeUnordered::RunnableStatus::kMissingDep:
      *waiting_for_peers = true;
      break;
    default:
      LOG(FATAL) << "unknown status code";
  }
  return exec_is_scheduled;
}

void
Executor::ExecuteForLoopTile(AbstractDistArrayPartition* partition_to_exec,
                             const std::string &loop_batch_func_name) {
  auto cpp_func = std::bind(
      JuliaEvaluator::StaticExecForLoopTile,
      std::placeholders::_1,
      partition_to_exec,
      loop_batch_func_name);

  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kExecForLoopTile;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
}

void
Executor::ExecForLoopSendResults(int32_t time_partition_id_to_send) {
  if (dynamic_cast<ExecForLoopSpaceTimeUnordered*>(exec_for_loop_.get()) == nullptr) {
    LOG(FATAL) << "unsupported!";
  }

  auto *exec_for_loop_space_time = dynamic_cast<ExecForLoopSpaceTimeUnordered*>(
      exec_for_loop_.get());
  int32_t dest_executor_id = exec_for_loop_space_time->GetDestExecutorId();
  std::unordered_map<int32_t, AbstractDistArrayPartition*> time_partitions_to_send;
  exec_for_loop_space_time->GetDistArrayTimePartitionsToSend(&time_partitions_to_send);
  for (auto &partition_pair : time_partitions_to_send) {
    auto dist_array_id = partition_pair.first;
    auto *dist_array_partition = partition_pair.second;
    auto buff_pair = dist_array_partition->Serialize();
    auto* buff = buff_pair.first;
    auto buff_size = buff_pair.second;
    message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgPipelineTimePartition>(
        &send_buff_, dist_array_id, time_partition_id_to_send, buff_size);
    send_buff_.set_next_to_send(buff, buff_size, true);
    Send(&peer_conn_[dest_executor_id], peer_[dest_executor_id].get());
    send_buff_.clear_to_send();
    send_buff_.reset_sent_sizes();
    auto &dist_array = dist_arrays_.at(dist_array_id);
    dist_array.DeletePartition(time_partition_id_to_send);
  }
}

void
Executor::RequestDistArrayData() {
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRequestExecForLoopDistArrayData>(
      &send_buff_);
  Send(&prt_poll_conn_, prt_pipe_conn_.get());
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
}

}
}
