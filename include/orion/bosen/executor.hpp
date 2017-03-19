#pragma once

#include <memory>
#include <iostream>
#include <vector>

#include <orion/bosen/config.hpp>
#include <orion/noncopyable.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/util.hpp>
#include <orion/bosen/message.hpp>
#include <orion/bosen/execute_message.hpp>
#include <orion/bosen/host_info.hpp>
#include <orion/bosen/server_thread.hpp>
#include <orion/bosen/client_thread.hpp>
#include <orion/bosen/event_handler.hpp>
#include <orion/bosen/thread_pool.hpp>
#include <orion/bosen/julia_eval_thread.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/byte_buffer.hpp>
#include <orion/bosen/task.pb.h>
#include <orion/bosen/recv_arbitrary_bytes.hpp>

namespace orion {
namespace bosen {

class Executor {
 private:
  struct PollConn {
    enum class ConnType {
      listen = 0,
        master = 1,
        compute = 2,
        peer = 3
    };
    void* conn;
    ConnType type;

    bool Receive() {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.Recv(&(sock_conn->recv_buff));
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.Recv(&(pipe_conn->recv_buff));
      }
    }

    bool Send() {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.Send(&(sock_conn->send_buff));
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.Send(&(pipe_conn->send_buff));
      }
    }

    conn::RecvBuffer& get_recv_buff() {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer) {
        return reinterpret_cast<conn::SocketConn*>(conn)->recv_buff;
      } else {
        return reinterpret_cast<conn::PipeConn*>(conn)->recv_buff;
      }
    }

    conn::SendBuffer& get_send_buff() {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer) {
        return reinterpret_cast<conn::SocketConn*>(conn)->send_buff;
      } else {
        return reinterpret_cast<conn::PipeConn*>(conn)->send_buff;
      }
    }

    bool is_connect_event() const {
      return type == ConnType::listen;
    }

    int get_read_fd() const {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.get_fd();
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.get_read_fd();
      }
    }

    int get_write_fd() const {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.get_fd();
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.get_write_fd();
      }
    }
  };

  enum class Action {
    kNone = 0,
      kExit = 1,
      kConnectToPeers = 2,
      kAckConnectToPeers = 3,
      kExecuteCode = 4
  };

  static const int32_t kPortSpan = 100;
  const size_t kCommBuffCapacity;
  const size_t kNumExecutors;
  const size_t kNumLocalExecutors;
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  const std::string kListenIp;
  const uint16_t kListenPort;
  const size_t kThreadPoolSize;
  const int32_t kId;

  EventHandler<PollConn> event_handler_;
  Blob master_send_mem_;
  Blob master_recv_mem_;
  conn::SocketConn master_;
  PollConn master_poll_conn_;
  Blob listen_send_mem_;
  Blob listen_recv_mem_;
  PollConn listen_poll_conn_;
  conn::SocketConn listen_;
  Blob send_mem_;
  conn::SendBuffer send_buff_;
  Action action_ {Action::kNone};
  Blob peer_send_mem_;
  Blob peer_recv_mem_;
  std::vector<std::unique_ptr<conn::SocketConn>> peer_;
  std::vector<PollConn> peer_conn_;
  Blob thread_pool_recv_mem_;
  Blob thread_pool_send_mem_;
  std::vector<std::unique_ptr<conn::PipeConn>> compute_;
  std::vector<PollConn> compute_conn_;
  ThreadPool thread_pool_;

  Blob julia_eval_recv_mem_;
  Blob julia_eval_send_mem_;
  std::unique_ptr<conn::PipeConn> julia_eval_;
  PollConn julia_eval_conn_;
  JuliaEvalThread julia_eval_thread_;

  ByteBuffer byte_buff_;

  size_t num_connected_peers_ {0};
  size_t num_identified_peers_ {0};
  std::vector<HostInfo> host_info_;
  size_t host_info_recved_size_ {0};

 public:
  Executor(const Config& config, int32_t id);
  ~Executor();
  DISALLOW_COPY(Executor);
  void operator() ();
 private:
  void InitListener();
  void HandleConnection(PollConn* poll_conn_ptr);
  int HandleClosedConnection(PollConn *poll_conn_ptr);
  void ConnectToMaster();
  int HandleMsg(PollConn* poll_conn_ptr);
  int HandleMasterMsg();
  int HandlePeerMsg(PollConn* poll_conn_ptr);
  int HandlePipeMsg(PollConn* poll_conn_ptr);
  int HandleExecuteMsg();
  void ConnectToPeers();
  void SetUpThreadPool();
  void ClearBeforeExit();
  void Send(PollConn* poll_conn_ptr, conn::SocketConn* sock_conn);
  void Send(PollConn* poll_conn_ptr, conn::PipeConn* pipe_conn);
  void ExecuteCode();
};

Executor::Executor(const Config& config, int32_t index):
    kCommBuffCapacity(config.kCommBuffCapacity),
    kNumExecutors(config.kNumExecutors),
    kNumLocalExecutors(config.kNumExecutorsPerWorker),
    kMasterIp(config.kMasterIp),
    kMasterPort(config.kMasterPort),
    kListenIp(config.kWorkerIp),
    kListenPort(config.kWorkerPort + index * kPortSpan),
    kThreadPoolSize(config.kExecutorThreadPoolSize),
    kId(config.kWorkerId*config.kNumExecutorsPerWorker + index),
    master_send_mem_(kCommBuffCapacity),
    master_recv_mem_(kCommBuffCapacity),
    master_(conn::Socket(),
            master_recv_mem_.data(),
            master_send_mem_.data(),
            kCommBuffCapacity),
    listen_send_mem_(kCommBuffCapacity),
    listen_recv_mem_(kCommBuffCapacity),
    listen_(conn::Socket(),
            listen_recv_mem_.data(),
            listen_send_mem_.data(),
            kCommBuffCapacity),
    send_mem_(kCommBuffCapacity),
    send_buff_(send_mem_.data(), kCommBuffCapacity),
    peer_send_mem_(config.kCommBuffCapacity*config.kNumExecutors),
    peer_recv_mem_(config.kCommBuffCapacity*config.kNumExecutors),
    peer_(config.kNumExecutors),
    peer_conn_(config.kNumExecutors),
    thread_pool_recv_mem_(kCommBuffCapacity*kThreadPoolSize),
    thread_pool_send_mem_(kCommBuffCapacity*kThreadPoolSize),
    compute_(kThreadPoolSize),
    compute_conn_(kThreadPoolSize),
    thread_pool_(config.kExecutorThreadPoolSize, config.kCommBuffCapacity),
    julia_eval_recv_mem_(kCommBuffCapacity),
    julia_eval_send_mem_(kCommBuffCapacity),
    julia_eval_thread_(kCommBuffCapacity),
    host_info_(kNumExecutors) { }

Executor::~Executor() { }

void
Executor::operator() () {
  InitListener();

  listen_poll_conn_ = {&listen_, PollConn::ConnType::listen};
  int ret = event_handler_.SetToReadOnly(&listen_poll_conn_);
  CHECK_EQ(ret, 0);
  ConnectToMaster();

  master_poll_conn_ = {&master_, PollConn::ConnType::master};
  ret = event_handler_.SetToReadOnly(&master_poll_conn_);
  CHECK_EQ(ret, 0);

  {
    HostInfo host_info;
    ret = GetIPFromStr(kListenIp.c_str(), &host_info.ip);
    CHECK_NE(ret, 0);
    host_info.port = kListenPort;

    message::Helper::CreateMsg<
      message::ExecutorIdentity>(&send_buff_, kId, host_info);
    Send(&master_poll_conn_, &master_);
  }

  event_handler_.SetConnectEventHandler(
      std::bind(&Executor::HandleConnection, this,
                std::placeholders::_1));

  event_handler_.SetClosedConnectionHandler(
      std::bind(&Executor::HandleClosedConnection, this,
                std::placeholders::_1));

  event_handler_.SetReadEventHandler(
      std::bind(&Executor::HandleMsg, this, std::placeholders::_1));

  event_handler_.SetDefaultWriteEventHandler();
  SetUpThreadPool();

  while (true) {
    event_handler_.WaitAndHandleEvent();
    if (action_ == Action::kExit) break;
  }
  ClearBeforeExit();
}

void
Executor::InitListener () {
  uint32_t ip;
  int ret = GetIPFromStr(kListenIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = listen_.sock.Bind(ip, kListenPort);
  CHECK_EQ(ret, 0);
  ret = listen_.sock.Listen(kNumExecutors);
  CHECK_EQ(ret, 0);
}

void
Executor::HandleConnection(PollConn* poll_conn_ptr) {
  conn::Socket accepted;
  listen_.sock.Accept(&accepted);

  uint8_t *recv_mem = peer_recv_mem_.data()
                      + kCommBuffCapacity*num_connected_peers_;

  uint8_t *send_mem = peer_send_mem_.data()
                      + kCommBuffCapacity*(num_connected_peers_ + 1);

  auto *sock_conn = new conn::SocketConn(
      accepted, recv_mem, send_mem, kCommBuffCapacity);

  auto &curr_poll_conn = peer_conn_[num_connected_peers_];
  curr_poll_conn.conn = sock_conn;
  curr_poll_conn.type = PollConn::ConnType::peer;
  int ret = event_handler_.SetToReadOnly(&curr_poll_conn);
  CHECK_EQ(ret, 0);
  num_connected_peers_++;
}

int
Executor::HandleClosedConnection(PollConn *poll_conn_ptr) {
  int ret = EventHandler<PollConn>::kNoAction;
  auto type = poll_conn_ptr->type;
  if (type == PollConn::ConnType::listen
      || type == PollConn::ConnType::compute
      || type == PollConn::ConnType::peer) {
    int ret = event_handler_.Remove(poll_conn_ptr);
    CHECK_EQ(ret, 0);
  } else {
    int ret = event_handler_.Remove(poll_conn_ptr);
    CHECK_EQ(ret, 0);
    auto* conn = reinterpret_cast<conn::SocketConn*>(poll_conn_ptr->conn);
    conn->sock.Close();
    action_ = Action::kExit;
    ret |= EventHandler<PollConn>::kExit;
  }
  return ret;
}

void
Executor::ConnectToMaster() {
  uint32_t ip;
  int ret = GetIPFromStr(kMasterIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = master_.sock.Connect(ip, kMasterPort);
  CHECK(ret == 0) << "executor failed connecting to master";
}

int
Executor::HandleMsg(PollConn* poll_conn_ptr) {
  int ret = 0;
  if (poll_conn_ptr->type == PollConn::ConnType::master) {
    ret = HandleMasterMsg();
  } else if (poll_conn_ptr->type == PollConn::ConnType::peer) {
    ret = HandlePeerMsg(poll_conn_ptr);
  } else {
    ret = HandlePipeMsg(poll_conn_ptr);
  }

  while (action_ != Action::kNone
         && action_ != Action::kExit) {
    switch(action_) {
      case Action::kConnectToPeers:
        {
          ConnectToPeers();
          if (kId == 0) action_ = Action::kAckConnectToPeers;
          else action_ = Action::kNone;
        }
        break;
      case Action::kAckConnectToPeers:
        {
          message::Helper::CreateMsg<message::ExecutorConnectToPeersAck>(&send_buff_);
          Send(&master_poll_conn_, &master_);
          action_ = Action::kNone;
        }
        break;
      case Action::kExecuteCode:
        {
          ExecuteCode();
          action_ = Action::kNone;
        }
        break;
      case Action::kExit:
        break;
      default:
        LOG(FATAL) << "unknown";
    }
  }
  return ret;
}

int
Executor::HandleMasterMsg() {
  auto &sock = master_.sock;
  auto &recv_buff = master_.recv_buff;

  auto msg_type = message::Helper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kNoAction;
  switch (msg_type) {
    case message::Type::kExecutorConnectToPeers:
      {
        auto *msg = message::Helper::get_msg<message::ExecutorConnectToPeers>(
              recv_buff);
        size_t expected_size = msg->num_executors*sizeof(HostInfo);
        bool received_next_msg =
            ReceiveArbitraryBytes(sock, &recv_buff,
                                  reinterpret_cast<uint8_t*>(host_info_.data()), &host_info_recved_size_,
                                  expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kConnectToPeers;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::Type::kExecutorStop:
      {
        action_ = Action::kExit;
        ret = EventHandler<PollConn>::kClearOneMsg | EventHandler<PollConn>::kExit;
      }
      break;
    case message::Type::kExecuteMsg:
      {
        ret = HandleExecuteMsg();
      }
      break;
    default:
      {
        LOG(FATAL) << "unknown message type " << static_cast<int>(msg_type);
      }
      break;
  }
  return ret;
}

int
Executor::HandlePeerMsg(PollConn* poll_conn_ptr) {
  auto &recv_buff = poll_conn_ptr->get_recv_buff();

  auto msg_type = message::Helper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kClearOneMsg;
  switch (msg_type) {
    case message::Type::kExecutorIdentity:
      {
        auto *msg = message::Helper::get_msg<message::ExecutorIdentity>(recv_buff);
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(poll_conn_ptr->conn);
        peer_[msg->executor_id].reset(sock_conn);
        num_identified_peers_++;
        if (num_identified_peers_ == kId) {
          action_ = Action::kAckConnectToPeers;
        }
        ret = EventHandler<PollConn>::kClearOneMsg;
      }
      break;
    default:
      {
        LOG(FATAL) << "unknown message type " << static_cast<int>(msg_type);
      }
      break;
  }
  return ret;
}

int
Executor::HandlePipeMsg(PollConn* poll_conn_ptr) {
  return EventHandler<PollConn>::kClearOneMsg;
}

int
Executor::HandleExecuteMsg() {
  auto &recv_buff = master_.recv_buff;
  auto msg_type = message::ExecuteMsgHelper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kClearOneMsg;
  switch (msg_type) {
    case message::ExecuteMsgType::kExecuteCode:
      {
        auto* msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgExecuteCode>(recv_buff);
        size_t expected_size = msg->task_size;
        bool received_next_msg
            = ReceiveArbitraryBytes(master_.sock, &recv_buff, &byte_buff_,
                                    expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kExecuteCode;
        } else {
          ret = EventHandler<PollConn>::kNoAction;
          action_ = Action::kNone;
        }
      }
      break;
    default:
      {
        LOG(FATAL) << "unknown message type " << static_cast<int>(msg_type);
      }
      break;
  }

  return ret;
}

void
Executor::ConnectToPeers() {
  HostInfo host_info;
  int ret = GetIPFromStr(kListenIp.c_str(), &host_info.ip);
  CHECK_NE(ret, 0);
  host_info.port = kListenPort;
  message::Helper::CreateMsg<
    message::ExecutorIdentity>(&send_buff_, kId, host_info);

  for (int i = kId + 1; i < kNumExecutors; i++) {
    uint32_t ip = host_info_[i].ip;
    uint16_t port = host_info_[i].port;
    conn::Socket peer_sock;
    ret = peer_sock.Connect(ip, port);
    CHECK(ret == 0) << "executor failed connecting to peer " << i
                    << " ip = " << ip << " port = " << port;
    peer_[i].reset(new conn::SocketConn(peer_sock,
                                        peer_recv_mem_.data() + kCommBuffCapacity*i,
                                        peer_send_mem_.data() + kCommBuffCapacity*i,
                                        kCommBuffCapacity));
    peer_conn_[i].conn = peer_[i].get();
    peer_conn_[i].type = PollConn::ConnType::peer;
    Send(&peer_conn_[i], peer_[i].get());
    int ret = event_handler_.SetToReadOnly(&peer_conn_[i]);
    CHECK_EQ(ret, 0);
  }
}

void
Executor::SetUpThreadPool() {
  auto read_pipe = julia_eval_thread_.get_read_pipe();
  julia_eval_ = std::make_unique<conn::PipeConn>(
      read_pipe, julia_eval_recv_mem_.data(),
      julia_eval_send_mem_.data(), kCommBuffCapacity);
  julia_eval_conn_.type = PollConn::ConnType::compute;
  julia_eval_conn_.conn = julia_eval_.get();
  int ret = event_handler_.SetToReadOnly(&julia_eval_conn_);
  CHECK_EQ(ret, 0);
  julia_eval_thread_.Start();
  for (int i = 0; i < kThreadPoolSize; ++i) {
    auto read_pipe = thread_pool_.get_read_pipe(i);
    compute_[i] = std::make_unique<conn::PipeConn>(
        read_pipe, thread_pool_recv_mem_.data() + kCommBuffCapacity*i,
        thread_pool_send_mem_.data() + kCommBuffCapacity*i,
        kCommBuffCapacity);
    compute_conn_[i].type = PollConn::ConnType::compute;
    compute_conn_[i].conn = compute_[i].get();
    int ret = event_handler_.SetToReadOnly(&compute_conn_[i]);
    CHECK_EQ(ret, 0);
  }
  thread_pool_.Start();
}

void
Executor::ClearBeforeExit() {
  julia_eval_thread_.Stop();
  thread_pool_.StopAll();
}

void
Executor::Send(PollConn* poll_conn_ptr, conn::SocketConn* sock_conn) {
  auto& send_buff = poll_conn_ptr->get_send_buff();
  if (send_buff.get_remaining_to_send_size() > 0
      || send_buff.get_remaining_next_to_send_size() > 0) {
    bool sent = sock_conn->sock.Send(&send_buff);
    while (!sent) {
      sent = sock_conn->sock.Send(&send_buff);
    }
    send_buff.clear_to_send();
  }
  bool sent = sock_conn->sock.Send(&send_buff_);
  if (!sent) {
    send_buff.Copy(send_buff_);
    event_handler_.SetToReadWrite(poll_conn_ptr);
  }
  send_buff_.reset_sent_sizes();
}

void
Executor::Send(PollConn* poll_conn_ptr, conn::PipeConn* pipe_conn) {
  auto& send_buff = poll_conn_ptr->get_send_buff();
  if (send_buff.get_remaining_to_send_size() > 0
      || send_buff.get_remaining_next_to_send_size() > 0) {
    bool sent = pipe_conn->pipe.Send(&send_buff);
    while (!sent) {
      sent = pipe_conn->pipe.Send(&send_buff);
    }
    send_buff.clear_to_send();
  }
  bool sent = pipe_conn->pipe.Send(&send_buff_);
  if (!sent) {
    send_buff.Copy(send_buff_);
    event_handler_.SetToReadWrite(poll_conn_ptr);
  }
  send_buff_.reset_sent_sizes();
}

void
Executor::ExecuteCode() {
  LOG(INFO) << __func__;
  std::string task_str(reinterpret_cast<const char*>(byte_buff_.GetBytes()),
                       byte_buff_.GetSize());
  task::ExecuteCode execute;
  execute.ParseFromString(task_str);
  LOG(INFO) << execute.code();

  //Task julia_task;
  //julia_task.code = execute.code();
  //julia_eval_thread_.SchedTask(julia_task);
}

}
}
