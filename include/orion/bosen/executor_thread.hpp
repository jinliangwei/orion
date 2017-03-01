#pragma once

#include <memory>
#include <iostream>
#include <vector>

#include <orion/bosen/config.hpp>
#include <orion/noncopyable.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/util.hpp>
#include <orion/bosen/message.hpp>
#include <orion/bosen/host_info.hpp>
#include <orion/bosen/server_thread.hpp>
#include <orion/bosen/client_thread.hpp>
#include <orion/bosen/event_handler.hpp>

namespace orion {
namespace bosen {

class ExecutorThread {
 private:
  struct PollConn {
    enum class ConnType {
      listen = 0,
        master = 1,
        pipe = 2
    };
    void* conn;
    ConnType type;

    bool Receive() {
      if (type == ConnType::listen
          || type == ConnType::master) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.Recv(&(sock_conn->recv_buff));
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.Recv(&(pipe_conn->recv_buff));
      }
    }

    conn::RecvBuffer& get_recv_buff() {
      if (type == ConnType::listen
          || type == ConnType::master) {
        return reinterpret_cast<conn::SocketConn*>(conn)->recv_buff;
      } else {
        return reinterpret_cast<conn::PipeConn*>(conn)->recv_buff;
      }
    }
    bool is_noread_event() const {
      return type == ConnType::listen;
    }
  };
  static const int32_t kPortSpan = 100;
  const size_t kCommBuffCapacity;
  const size_t kNumExecutors;
  const size_t kNumLocalExecutors;
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  const std::string kListenIp;
  const uint16_t kListenPort;
  const int32_t kId;

  EventHandler<PollConn> event_handler_;
  std::vector<uint8_t> master_recv_mem_;
  conn::SocketConn master_;
  std::vector<uint8_t> listen_recv_mem_;
  conn::SocketConn listen_;
  std::vector<uint8_t> send_mem_;
  conn::SendBuffer send_buff_;
  //State state_;

  std::vector<conn::Socket> clients_;
  size_t num_connected_clients_ {0};
  std::vector<HostInfo> host_info_;
  size_t host_info_recved_ {0};
  size_t num_executors_ {0};
  size_t num_executors_recved_ {0};
  // 0: executor <> server :1
  conn::Pipe pipe_ese_[2];
  // 0: executor <> client :1
  conn::Pipe pipe_ecl_[2];
  // 0: server <> client :1
  conn::Pipe pipe_scl_[2];

  std::vector<uint8_t> pipe_recv_mem_;
  std::unique_ptr<ServerThread> server_;
  std::thread server_thread_;
  std::unique_ptr<conn::PipeConn> server_pipe_conn_;
  PollConn server_poll_conn_;

  std::unique_ptr<ClientThread> client_;
  std::thread client_thread_;
  std::unique_ptr<conn::PipeConn> client_pipe_conn_;
  PollConn client_poll_conn_;

  size_t num_peer_connected_threads_ {0};

 public:
  ExecutorThread(const Config& config, int32_t id);
  ~ExecutorThread();
  DISALLOW_COPY(ExecutorThread);
  void operator() ();
 private:
  void InitListener();
  void HandleConnection(PollConn* poll_conn_ptr);
  void ConnectToMaster();
  int HandleMsg(PollConn* poll_conn_ptr);
  int HandleMasterMsg();
  int HandlePipeMsg(PollConn* poll_conn_ptr);
  //void SetUpLocalThreads();
};

ExecutorThread::ExecutorThread(const Config& config, int32_t index):
    kCommBuffCapacity(config.kCommBuffCapacity),
    kNumExecutors(config.kNumExecutors),
    kNumLocalExecutors(config.kNumExecutorsPerWorker),
    kMasterIp(config.kMasterIp),
    kMasterPort(config.kMasterPort),
    kListenIp(config.kWorkerIp),
    kListenPort(config.kWorkerPort + index * kPortSpan),
    kId(config.kWorkerId*config.kNumExecutorsPerWorker + index),
    master_recv_mem_(kCommBuffCapacity),
    master_(conn::Socket(), master_recv_mem_.data(),
            kCommBuffCapacity),
    listen_recv_mem_(kCommBuffCapacity),
    listen_(conn::Socket(), listen_recv_mem_.data(),
            kCommBuffCapacity),
    send_mem_(kCommBuffCapacity),
    send_buff_(send_mem_.data(), kCommBuffCapacity),
    clients_(kNumExecutors),
    host_info_(kNumExecutors),
    pipe_recv_mem_(kCommBuffCapacity*3) { }

ExecutorThread::~ExecutorThread() { }

void
ExecutorThread::operator() () {
  InitListener();

  PollConn listen_poll_conn = {&listen_, PollConn::ConnType::listen};
  event_handler_.AddPollConn(listen_.sock.get_fd(), &listen_poll_conn);

  ConnectToMaster();

  PollConn master_poll_conn = {&master_, PollConn::ConnType::master};
  event_handler_.AddPollConn(master_.sock.get_fd(), &master_poll_conn);

  event_handler_.SetNoreadEventHandler(
      std::bind(&ExecutorThread::HandleConnection, this,
                std::placeholders::_1));

  event_handler_.SetReadEventHandler(
      std::bind(&ExecutorThread::HandleMsg, this, std::placeholders::_1));

  while (1) {
    event_handler_.WaitAndHandleEvent();
  }
}

void
ExecutorThread::InitListener () {
  uint32_t ip;
  int ret = GetIPFromStr(kListenIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = listen_.sock.Bind(ip, kListenPort);
  CHECK_EQ(ret, 0);
  ret = listen_.sock.Listen(kNumExecutors);
}

void
ExecutorThread::HandleConnection(PollConn* poll_conn_ptr) {

}

void
ExecutorThread::ConnectToMaster() {
  uint32_t ip;
  int ret = GetIPFromStr(kMasterIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = master_.sock.Connect(ip, kMasterPort);
  CHECK(ret == 0) << "executor has connected";

  HostInfo host_info;
  ret = GetIPFromStr(kListenIp.c_str(), &host_info.ip);
  CHECK_NE(ret, 0);
  host_info.port = kListenPort;
  message::Helper::CreateMsg<
    message::ExecutorIdentity>(&send_buff_, kId, host_info);
  size_t sent_size = master_.sock.Send(&send_buff_);
  CHECK(conn::CheckSendSize(send_buff_, sent_size));
}

int
ExecutorThread::HandleMsg(PollConn* poll_conn_ptr) {
  if (poll_conn_ptr->type == PollConn::ConnType::master) {
    return HandleMasterMsg();
  } else {
    return HandlePipeMsg(poll_conn_ptr);
  }
}

int
ExecutorThread::HandleMasterMsg() {
  auto &sock = master_.sock;
  auto &recv_buff = master_.recv_buff;

  auto msg_type = message::Helper::get_type(recv_buff);
  int ret = 0;
  switch (msg_type) {
    case message::Type::kExecutorConnectToPeers:
      {
        if (recv_buff.IsExepectingNextMsg()) {
          bool recv = sock.Recv(&recv_buff, host_info_.data() + host_info_recved_);
          if (!recv) return 0;
          CHECK (!recv_buff.is_error()) << "driver error during receiving "
                                        << errno;
          CHECK (!recv_buff.EOFAtIncompleteMsg()) << "driver error : early EOF";
          host_info_recved_ = recv_buff.get_next_recved_size();
        } else {
          auto *msg = message::Helper::get_msg<message::ExecutorConnectToPeers>(
              recv_buff);
          recv_buff.set_next_expected_size(msg->num_executors*sizeof(HostInfo));
          if (recv_buff.get_size() > recv_buff.get_expected_size()) {
            size_t size_to_copy = recv_buff.get_size()
                                  - recv_buff.get_expected_size();
            memcpy(host_info_.data(), recv_buff.get_payload_mem(), size_to_copy);
            host_info_recved_ += size_to_copy;
            recv_buff.IncNextRecvedSize(size_to_copy);
          }
        }
        if (recv_buff.ReceivedFullNextMsg()) {
          LOG(INFO) << "got all host info";
          ret = 2;
        } else ret = 0;
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
ExecutorThread::HandlePipeMsg(PollConn* poll_conn_ptr) {
  return 1;
}

}
}
