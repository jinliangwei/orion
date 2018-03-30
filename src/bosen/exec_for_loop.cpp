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

  const int32_t *dist_array_buffer_ids
      = exec_for_loop_task.dist_array_buffer_ids().data();
  size_t num_dist_array_buffers
      = exec_for_loop_task.dist_array_buffer_ids_size();

  const int32_t *written_dist_array_ids
      = exec_for_loop_task.written_dist_array_ids().data();
  size_t num_written_dist_array_ids
      = exec_for_loop_task.written_dist_array_ids_size();

  const int32_t *accessed_dist_array_ids
      = exec_for_loop_task.accessed_dist_array_ids().data();
  size_t num_accessed_dist_arrays = exec_for_loop_task.accessed_dist_array_ids_size();

  const std::string * const *global_read_only_var_vals
      = exec_for_loop_task.global_read_only_var_vals().data();
  size_t num_global_read_only_var_vals
      = exec_for_loop_task.global_read_only_var_vals_size();

  const std::string * const *accumulator_var_syms
      = exec_for_loop_task.accumulator_var_syms().data();
  size_t num_accumulator_var_syms
      = exec_for_loop_task.accumulator_var_syms_size();

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

  auto cpp_func = std::bind(
      &AbstractExecForLoop::Init,
      exec_for_loop_.get());

  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kExecForLoopInit;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
}

void
Executor::CreateServerExecForLoop() {
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());
  task::ExecForLoop exec_for_loop_task;
  exec_for_loop_task.ParseFromString(task_str);
  const int32_t *global_indexed_dist_array_ids
      = exec_for_loop_task.global_indexed_dist_array_ids().data();
  size_t num_global_indexed_dist_arrays
      = exec_for_loop_task.global_indexed_dist_array_ids_size();
  const int32_t *dist_array_buffer_ids
      = exec_for_loop_task.dist_array_buffer_ids().data();
  size_t num_dist_array_buffers
      = exec_for_loop_task.dist_array_buffer_ids_size();

  server_exec_for_loop_.reset(
      new ServerExecForLoop(
          kServerId,
          kNumExecutors,
          &dist_arrays_,
          &dist_array_buffers_,
          dist_array_buffer_info_map_,
          global_indexed_dist_array_ids,
          num_global_indexed_dist_arrays,
          dist_array_buffer_ids,
          num_dist_array_buffers));
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
          auto* exec_for_loop_ptr = exec_for_loop_.get();
          auto cpp_func = std::bind(
              &AbstractExecForLoop::ComputePrefetchIndinces,
              exec_for_loop_ptr);
          exec_cpp_func_task_.func = cpp_func;
          exec_cpp_func_task_.label = TaskLabel::kComputePrefetchIndices;
          julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
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
          stop_search = true;
          if (kNumExecutors > 1)
            SendPredCompletion();
          bool requested = CheckAndRequestExecForLoopPredecesorCompletion();
          if (requested) {
            event_handler_.SetToReadOnly(&prt_poll_conn_);
          } else {
            ExecForLoopClear();
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
  ExecutorSendBufferMap cache_send_buffer_map;
  exec_for_loop_->GetAndClearDistArrayCacheSendMap(&cache_send_buffer_map);
  for (auto &buffer_pair : cache_send_buffer_map) {
    int32_t server_id = buffer_pair.first;
    auto &send_data_buffer = buffer_pair.second;
    auto *data_bytes = send_data_buffer.first;
    size_t num_bytes = send_data_buffer.second;

    message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgExecForLoopDistArrayCacheData>(
        &send_buff_, num_bytes);
    send_buff_.set_next_to_send(data_bytes, num_bytes, true);
    Send(&server_conn_[server_id], server_[server_id].get());
    send_buff_.clear_to_send();
    send_buff_.reset_sent_sizes();
  }

  ExecutorSendBufferMap buffer_send_buffer_map;
  exec_for_loop_->GetAndClearDistArrayBufferSendMap(&buffer_send_buffer_map);

  for (auto &buffer_pair : buffer_send_buffer_map) {
    int32_t server_id = buffer_pair.first;
    auto &send_data_buffer = buffer_pair.second;
    auto *data_bytes = send_data_buffer.first;
    size_t num_bytes = send_data_buffer.second;

    message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgExecForLoopDistArrayBufferData>(
        &send_buff_, num_bytes);
    send_buff_.set_next_to_send(data_bytes, num_bytes, true);
    Send(&server_conn_[server_id], server_[server_id].get());
    send_buff_.clear_to_send();
    send_buff_.reset_sent_sizes();
  }
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
  if (successor_to_send < 0) return;
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgReplyExecForLoopPredecessorCompletion>(
      &send_buff_, nullptr, 0);
  Send(&executor_conn_[successor_to_send], executor_[successor_to_send].get());
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
}

void
Executor::ExecForLoopClear() {
  auto cpp_func = std::bind(
      &AbstractExecForLoop::Clear,
      exec_for_loop_.get());

  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kExecForLoopClear;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
}

void
Executor::ExecForLoopAck() {
  LOG(INFO) << __func__;

  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgExecForLoopDone>(
      &send_buff_);
  for (int32_t server_id = 0; server_id < kNumServers; server_id++) {
    Send(&server_conn_[server_id], server_[server_id].get());
  }
  send_buff_.reset_sent_sizes();
  send_buff_.clear_to_send();

  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgExecForLoopAck>(
      &send_buff_);
  Send(&master_poll_conn_, &master_);
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
  exec_for_loop_.reset();
}

void
Executor::ServerExecForLoopAck() {
  LOG(INFO) << __func__;
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgExecForLoopAck>(
      &send_buff_);
  Send(&master_poll_conn_, &master_);
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
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
  if (exec_for_loop_->GetPredecessor() < 0) return false;
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRequestExecForLoopPredecessorCompletion>(
      &send_buff_);
  Send(&prt_poll_conn_, prt_pipe_conn_.get());
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
  return true;
}

void
Executor::SerializeAndSendExecForLoopPrefetchRequests() {
  auto* exec_for_loop_ptr = exec_for_loop_.get();
  ExecutorSendBufferMap send_buffer_map;
  exec_for_loop_ptr->SerializeAndClearPrefetchIds(&send_buffer_map);
  for (auto &executor_send_buffer_pair : send_buffer_map) {
    int32_t server_id = executor_send_buffer_pair.first;
    auto &send_data_buffer = executor_send_buffer_pair.second;
    auto *data_bytes = send_data_buffer.first;
    size_t num_bytes = send_data_buffer.second;
    message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRequestDistArrayValues>(
        &send_buff_, num_bytes, kExecutorId, true);
    send_buff_.set_next_to_send(data_bytes, num_bytes, true);
    Send(&server_conn_[server_id], server_[server_id].get());
    send_buff_.clear_to_send();
    send_buff_.reset_sent_sizes();
  }
  exec_for_loop_->SentAllPrefetchRequests();
}

void
Executor::ReplyDistArrayValues() {
  auto &bytes_buff = exec_cpp_func_task_.result_buff;
  size_t num_bytes = bytes_buff.size();
  auto *buff_bytes = bytes_buff.data();
  auto *bytes = new uint8_t[num_bytes];
  memcpy(bytes, buff_bytes, num_bytes);
  bytes_buff.clear();
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgReplyDistArrayValues>(
      &send_buff_, num_bytes);
  send_buff_.set_next_to_send(bytes, num_bytes, true);
  int32_t recv_id = dist_array_value_request_meta_.requester_id;
  bool is_requester_executor = dist_array_value_request_meta_.is_requester_executor;

  auto &recv_conn = is_requester_executor ? executor_conn_ : server_conn_;
  auto &receiver = is_requester_executor ? executor_ : server_;

  Send(&recv_conn[recv_id], receiver[recv_id].get());
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
}

void
Executor::CacheGlobalIndexedDistArrayValues(
    PeerRecvGlobalIndexedDistArrayDataBuffer **buff_vec,
    size_t num_buffs) {
  auto runnable_status = exec_for_loop_->GetCurrPartitionRunnableStatus();
  if (runnable_status == AbstractExecForLoop::RunnableStatus::kAwaitGlobalIndexedDistArrays) {
    auto cpp_func = std::bind(
        &AbstractExecForLoop::CachePrefetchDistArrayValues,
        exec_for_loop_.get(),
        buff_vec,
        num_buffs);
    exec_cpp_func_task_.func = cpp_func;
    exec_cpp_func_task_.label = TaskLabel::kCachePrefetchDistArrayValues;
    julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
  } else {
    CHECK(runnable_status == AbstractExecForLoop::RunnableStatus::kRunnable);
    CHECK(num_buffs = 1);
    auto *buff = buff_vec[0];
    const auto *bytes = buff->byte_buff.data();
    size_t num_bytes = buff->byte_buff.size();
    julia_requester_->ReplyDistArrayData(bytes, num_bytes);
    delete buff;
    delete buff_vec;
  }
}

}
}
