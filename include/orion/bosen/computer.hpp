#pragma once

#include <orion/bosen/message.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/constants.hpp>
#include <climits>
#include <stdint.h>
#include <orion/noncopyable.hpp>
#include <orion/bosen/worker_runtime.hpp>
#include <orion/bosen/worker_task.hpp>
#include <orion/bosen/operator/max_table_key_operator.hpp>
#include <gflags/gflags.h>
#include <queue>

namespace orion {
namespace bosen {

class Computer {
 private:
  conn::Poll poll_;
  std::unique_ptr<uint8_t[]> send_mem_;
  conn::SendBuffer send_buff_;
  std::unique_ptr<uint8_t[]> recv_mem_;
  const size_t control_buff_capacity_;
  const size_t data_buff_capacity_;
  conn::PipeConn master_;
  conn::PipeConn server_;
  conn::PipeConn client_;
  bool stop_ {false};
  WorkerRuntime *runtime_;
  std::queue<WorkerTask> tasks_;
  bool curr_task_complete_ {true};

 public:
  Computer(conn::Pipe master, conn::Pipe server, conn::Pipe client,
           size_t control_buff_capacity,
           size_t data_buff_capacity,
           WorkerRuntime *runtime);
  void operator() ();
  DISALLOW_COPY(Computer);
 private:
  conn::RecvBuffer &HandleReadEvent(conn::PipeConn *pipe_conn_ptr);
};

Computer::Computer(
    conn::Pipe master, conn::Pipe server, conn::Pipe client,
    size_t control_buff_capacity,
    size_t data_buff_capacity,
    WorkerRuntime *runtime):
    send_mem_(std::make_unique<uint8_t[]>(data_buff_capacity)),
    send_buff_(send_mem_.get(), control_buff_capacity),
    recv_mem_(std::make_unique<uint8_t[]>(control_buff_capacity * 3)),
    control_buff_capacity_(control_buff_capacity),
    data_buff_capacity_(data_buff_capacity),
    master_(master, recv_mem_.get(), control_buff_capacity),
    server_(server, recv_mem_.get() + control_buff_capacity, control_buff_capacity),
    client_(client, recv_mem_.get() + control_buff_capacity * 2, control_buff_capacity),
  runtime_(runtime) {
  int ret = poll_.Init();
  CHECK(ret == 0) << "poll init failed";
  poll_.Add(master_.pipe.get_read_fd(), &master_);
  poll_.Add(server_.pipe.get_read_fd(), &server_);
  poll_.Add(client_.pipe.get_read_fd(), &client_);
}

conn::RecvBuffer &
Computer::HandleReadEvent(conn::PipeConn *pipe_conn_ptr) {
  auto &recv_buff = pipe_conn_ptr->recv_buff;
  auto &pipe = pipe_conn_ptr->pipe;

  if (!recv_buff.ReceivedFullMsg()) {
    bool recv = pipe.Recv(&recv_buff);
    if (!recv) return recv_buff;
  }

  CHECK (!recv_buff.is_error()) << "server error during receiving " << errno;
  CHECK (!recv_buff.EOFAtIncompleteMsg()) << "server error : early EOF";

  // maybe EOF but not received anything
  if (!recv_buff.ReceivedFullMsg()) return recv_buff;

  auto msg_type = message::Helper::get_type(
      recv_buff.get_payload_mem());
  switch(msg_type) {
    case message::Type::kExecutorStop:
      {
        stop_ = true;
        LOG(INFO) << "executor stop!";
      }
      break;
    case message::Type::kExecutorCreateTable:
      {
        auto in_msg = (message::ExecutorCreateTable*)
                      recv_buff.get_payload_mem();
        runtime_->CreateTable(in_msg->table_id,
                              TableConfig(in_msg->get_table_config_mem()));
        message::Helper::CreateMsg<
          message::ExecutorCreateTableAck>(&send_buff_, in_msg->task_id);
        master_.pipe.Send(&send_buff_);
      }
      break;
    case message::Type::kExecutorRangePartitionTable:
      {
        auto in_msg = (message::ExecutorRangePartitionTable*)
                      recv_buff.get_payload_mem();
        //tasks_.emplace(WorkerTask::Operator::kRangePartitionTable,
        //               in_msg->table_id, 0);
        //curr_task_complete_ = false;
        runtime_->CreateOutOperator<MaxTableKeyOperator>();
        auto max_key = runtime_->GetTableMaxLocalKey(in_msg->table_id, in_msg->key_index);
        LOG(INFO) << "max local key = " << max_key;
        auto out_msg = message::Helper::CreateMsg<message::TableMaxKey>(&send_buff_);
        out_msg->max_key = max_key;
        server_.pipe.Send(&send_buff_);
      }
      break;
    default:
      LOG(FATAL) << "who is this?";
      break;
  }

  recv_buff.ClearOneMsg();

  return recv_buff;
}

void
Computer::operator() () {
  static constexpr size_t kNumEvents = 100;
  epoll_event es[kNumEvents];
  while (1) {
    int num_events = poll_.Wait(es, kNumEvents);
    CHECK(num_events > 0);
    for (int i = 0; i < num_events; ++i) {
      auto pipe_conn_ptr = conn::Poll::EventConn<conn::PipeConn>(es, i);
      if (es[i].events & EPOLLIN) {
        auto &recv_buff = HandleReadEvent(pipe_conn_ptr);
        while (recv_buff.ReceivedFullMsg()) {
          HandleReadEvent(pipe_conn_ptr);
        }
        if (recv_buff.is_eof()) {
          LOG(INFO) << "someone has closed";
        }
      }
    }
    if (stop_) break;
  }
}

}
}
