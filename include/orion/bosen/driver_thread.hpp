#pragma once

#include <vector>
#include <pthread.h>
#include <errno.h>

#include <orion/bosen/message.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/noncopyable.hpp>
#include <orion/bosen/driver_runtime.hpp>
#include <orion/helper.hpp>
#include <gflags/gflags.h>
#include <memory>
#include <iostream>
#include <orion/bosen/block_queue.hpp>

namespace orion {
namespace bosen {

DECLARE_string(driver_ip);
DECLARE_int32(driver_port);
DECLARE_int32(driver_num_executors);
DECLARE_uint64(comm_buff_capacity);

const char *kDriverReadyString = "Driver is ready!";

class DriverThread {
 private:
  struct PollConn {
    enum class ConnType {
      listen = 0,
        executor = 1
    };

    void *conn;
    ConnType type;
  };

  enum class State {
    kInitConn = 0,
    kStartExecutors = 1,
    kExecutorsStarting = 2,
    kExecutorConnectToPeers = 3,
    kExecutorPeerConnecting = 4,
    kRunning = 5,
    kShutDown = 6,
    kExit = 7
  };
  const size_t num_executors_;
  conn::Poll poll_;
  conn::Socket listen_;
  std::unique_ptr<std::unique_ptr<conn::SocketConn>[]> executors_;
  std::unique_ptr<PollConn[]> poll_conns_;
  std::unique_ptr<HostInfo[]> host_info_;
  std::unique_ptr<uint8_t[]> recv_mem_;
  std::unique_ptr<uint8_t[]> send_mem_;
  conn::SendBuffer send_buff_;
  BlockQueue<DriverTask*> task_queue_;
  BlockQueue<DriverTask*> complete_queue_;
  DriverTask* curr_task_;

  size_t num_accepted_executors_ {0};
  size_t num_identified_executors_ {0};
  size_t num_started_executors_ {0};
  size_t num_closed_conns_ {0};
  State state_ {State::kInitConn};
  DriverRuntime runtime_;
  bool next_task_ {false};
  size_t num_peer_connected_executors_ {0};
 public:
  DriverThread(const DriverRuntimeConfig &runtime_config);
  ~DriverThread();

  DISALLOW_COPY(DriverThread);

  void operator() ();
  void ScheduleTask(DriverTask* task);
  DriverTask* GetCompletedTask();
 private:
  void InitListener();
  void HandleConnection(PollConn* poll_conn_ptr);
  void HandleClosedConnection(PollConn *poll_conn_ptr);
  conn::RecvBuffer& HandleReadEvent(PollConn* poll_conn_ptr);
  bool HandleExecutorMsg(PollConn *poll_conn_ptr);

  // driver state transitions
  void StartExecutors();
  void ExecutorConnectToPeers();
  void NextTask();
  void ShutDown();

  // helpers
  void BroadcastAllExecutors();

  uint8_t* AllocRecvMem();
};

DriverThread::DriverThread(const DriverRuntimeConfig &runtime_config):
    num_executors_(FLAGS_driver_num_executors),
    executors_(std::make_unique<std::unique_ptr<conn::SocketConn>[]>(
        num_executors_)),
    poll_conns_(std::make_unique<PollConn[]>(num_executors_)),
    host_info_(std::make_unique<HostInfo[]>(num_executors_)),
    recv_mem_(std::make_unique<uint8_t[]>(
        FLAGS_driver_num_executors
        * FLAGS_comm_buff_capacity)),
    send_mem_(std::make_unique<uint8_t[]>(FLAGS_comm_buff_capacity)),
  send_buff_(send_mem_.get(),
             FLAGS_comm_buff_capacity),
  runtime_(runtime_config, num_executors_) { }

DriverThread::~DriverThread() { }

void
DriverThread::InitListener () {
  std::string driver_listen_ip(FLAGS_driver_ip);
  listen_.Bind(driver_listen_ip.c_str(), (uint16_t) FLAGS_driver_port);
  listen_.Listen(num_executors_);
}

void
DriverThread::HandleConnection(PollConn *poll_conn_ptr) {
  conn::Socket accepted;
  listen_.Accept(&accepted);
  LOG(INFO) << "DriverThread " << __func__ << " socket fd = "
	    << accepted.get_fd();

  uint8_t *recv_mem = AllocRecvMem();
  auto *sock_conn = new conn::SocketConn(
      accepted, recv_mem,
      (size_t) FLAGS_comm_buff_capacity);

  auto &curr_poll_conn
      = poll_conns_[num_accepted_executors_];

  curr_poll_conn.conn = sock_conn;
  curr_poll_conn.type = PollConn::ConnType::executor;

  poll_.Add(accepted.get_fd(), &curr_poll_conn);
  num_accepted_executors_++;
}

void
DriverThread::HandleClosedConnection(PollConn *poll_conn_ptr) {
  num_closed_conns_++;
  auto &sock_conn = *reinterpret_cast<conn::SocketConn*>(
      poll_conn_ptr->conn);
  auto &sock = sock_conn.sock;
  poll_.Remove(sock.get_fd());
  sock.Close();
}

bool
DriverThread::HandleExecutorMsg(PollConn *poll_conn_ptr) {
  auto &sock_conn = *reinterpret_cast<conn::SocketConn*>(
      poll_conn_ptr->conn);
  auto &sock = sock_conn.sock;
  auto &recv_buff = sock_conn.recv_buff;

  if (!recv_buff.ReceivedFullMsg()) {
    bool recv = sock.Recv(&recv_buff);
    if (!recv) return false;
  }

  CHECK (!recv_buff.is_error()) << "driver error during receiving " << errno;
  CHECK (!recv_buff.EOFAtIncompleteMsg()) << "driver error : early EOF";
  // maybe EOF but not received anything
  if (!recv_buff.ReceivedFullMsg()) return false;

  auto msg_type = message::Helper::get_type(recv_buff);
  switch (msg_type) {
    case message::Type::kExecutorHostInfo:
      {
        auto *msg = message::Helper::get_msg<message::ExecutorHostInfo>(recv_buff);
        host_info_[msg->executor_id] = msg->host_info;
        executors_[msg->executor_id].reset(&sock_conn);
        num_identified_executors_++;
        if (state_ == State::kInitConn
            && (num_identified_executors_ ==
                num_executors_)) {
          state_ = State::kStartExecutors;
        }
      }
      break;
    case message::Type::kExecutorReady:
      {
        num_started_executors_++;
        if (state_ == State::kExecutorsStarting
            && num_started_executors_ == num_executors_) {
          state_ = State::kExecutorConnectToPeers;
        }
      }
      break;
    case message::Type::kExecutorConnectToPeersAck:
      {
        num_peer_connected_executors_++;
        if (num_peer_connected_executors_
            == num_executors_) {
          state_ = State::kRunning;
          next_task_ = true;
        }
      }
      break;
    default:
      {
        LOG(FATAL) << "Unknown message type " << static_cast<int>(msg_type)
                   << " from " << sock.get_fd();
      }
  }
  return true;
}

conn::RecvBuffer &
DriverThread::HandleReadEvent(PollConn *poll_conn_ptr) {
  bool next_message = HandleExecutorMsg(poll_conn_ptr);
  auto &recv_buff = reinterpret_cast<conn::SocketConn*>(
          poll_conn_ptr->conn)->recv_buff;
  if (next_message) {
    recv_buff.ClearOneMsg();
  }
  return recv_buff;
}

void
DriverThread::operator() () {
  static const size_t kNumEvents = 100;
  LOG(INFO) << "DriverThread is started";
  InitListener();
  int ret = poll_.Init();
  CHECK(ret == 0) << "poll init failed";
  PollConn listen_poll_conn = {&listen_, PollConn::ConnType::listen};
  poll_.Add(listen_.get_fd(), &listen_poll_conn);

  std::cout << kDriverReadyString << std::endl;
  epoll_event es[kNumEvents];
  while (1) {
    int num_events = poll_.Wait(es, kNumEvents);
    CHECK(num_events > 0);
    for (int i = 0; i < num_events; ++i) {
      PollConn *poll_conn_ptr = conn::Poll::EventConn<PollConn>(es, i);
      if (es[i].events & EPOLLIN) {
        if (poll_conn_ptr->type == PollConn::ConnType::listen) {
          HandleConnection(poll_conn_ptr);
        } else {
          auto &recv_buff = HandleReadEvent(poll_conn_ptr);
          // repeat until receive buffer is exhausted
          while (recv_buff.ReceivedFullMsg()) {
            LOG(INFO) << "continue handling read from unexhausted buffer";
            HandleReadEvent(poll_conn_ptr);
          }
          if (recv_buff.is_eof()) {
            LOG(INFO) << "someone has closed";
            HandleClosedConnection(poll_conn_ptr);
          }
        }
      } else {
        LOG(WARNING) << "unknown event happened happend: " << es[i].events;
      }
    }
    switch(state_) {
      case State::kStartExecutors:
        {
          StartExecutors();
          state_ = State::kExecutorsStarting;
        }
        break;
      case State::kExecutorConnectToPeers:
        {
          ExecutorConnectToPeers();
          state_ = State::kExecutorPeerConnecting;
        }
        break;
      case State::kRunning:
        {
          LOG(INFO) << "num_peer_connected_executors_ = " << num_peer_connected_executors_;
          if (next_task_) {
            NextTask();
            next_task_ = false;
          }
        }
        break;
      case State::kShutDown:
        {
          if (num_closed_conns_ == num_executors_) {
            listen_.Close();
            state_ = State::kExit;
          }
        }
        break;
      default:
        break;
    }
    if (state_ == State::kExit) break;
  }

  if (state_ == State::kExit) {
    complete_queue_.Push(curr_task_);
  }

  LOG(INFO) << "driver exiting";
}

void
DriverThread::StartExecutors() {
  LOG(INFO) << "DriverThread " << __func__;

  size_t num_hostinfos_per_msg
      = message::Helper::get_msg_payload_capacity<message::ExecutorInfo>(send_buff_) / sizeof(HostInfo);
  size_t hostinfos_offset = 0;
  size_t hosts_left_to_send = num_executors_;
  while (hosts_left_to_send > 0) {
    size_t num_hosts_to_send = std::min(hosts_left_to_send, num_hostinfos_per_msg);
    auto host_info_msg
        = message::Helper::CreateMsg<
          message::ExecutorInfo>(&send_buff_, num_executors_,
                                 num_hosts_to_send);

    memcpy(host_info_msg->get_host_info_mem(),
           reinterpret_cast<uint8_t*>(host_info_.get()) + hostinfos_offset,
           sizeof(HostInfo) * num_hosts_to_send);
    hostinfos_offset += sizeof(HostInfo) * num_hosts_to_send;
    BroadcastAllExecutors();
    hosts_left_to_send -= num_hosts_to_send;
  }
}

void
DriverThread::ExecutorConnectToPeers() {
  message::Helper::CreateMsg<message::ExecutorConnectToPeers>(&send_buff_);
  BroadcastAllExecutors();
}

void
DriverThread::ScheduleTask(DriverTask* task) {
  task_queue_.Push(task);
}

DriverTask*
DriverThread::GetCompletedTask() {
  return complete_queue_.Pop();
}

void
DriverThread::NextTask() {
  curr_task_ = task_queue_.Pop();
  if (curr_task_->inst == DriverTask::Inst::kStall) {
    ShutDown();
    state_ = State::kShutDown;
    return;
  }
}

void
DriverThread::ShutDown() {
  message::Helper::CreateMsg<message::ExecutorStop>(&send_buff_);
  for (int i = 0; i < num_executors_; ++i) {
    size_t nsent = executors_[i]->sock.Send(&send_buff_);
    CHECK(conn::CheckSendSize(send_buff_, nsent));
  }
}

void
DriverThread::BroadcastAllExecutors() {
  for (int i = 0; i < num_executors_; ++i) {
    size_t nsent = executors_[i]->sock.Send(&send_buff_);
    CHECK(conn::CheckSendSize(send_buff_, nsent)) << "send only " << nsent;
  }
}

uint8_t*
DriverThread::AllocRecvMem() {
  return recv_mem_.get()
      + num_accepted_executors_
      * FLAGS_comm_buff_capacity;
}

}
}
