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
        pipe = 2,
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

    conn::RecvBuffer& get_recv_buff() {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer) {
        return reinterpret_cast<conn::SocketConn*>(conn)->recv_buff;
      } else {
        return reinterpret_cast<conn::PipeConn*>(conn)->recv_buff;
      }
    }
    bool is_noread_event() const {
      return type == ConnType::listen;
    }

    int get_fd() const {
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
  };

  enum class Action {
    kNone = 0,
      kExit = 1,
      kConnectToPeers = 2,
      kAckConnectToPeers = 3
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
  Action action_ {Action::kNone};
  //State state_;

  std::vector<uint8_t> peer_recv_mem_;
  std::vector<std::unique_ptr<conn::SocketConn>> peer_;
  std::vector<PollConn> peer_conn_;
  size_t num_connected_peers_ {0};
  size_t num_identified_peers_ {0};
  std::vector<HostInfo> host_info_;
  size_t host_info_recved_size_ {0};

 public:
  ExecutorThread(const Config& config, int32_t id);
  ~ExecutorThread();
  DISALLOW_COPY(ExecutorThread);
  void operator() ();
 private:
  void InitListener();
  void HandleConnection(PollConn* poll_conn_ptr);
  void HandleClosedConnection(PollConn *poll_conn_ptr);
  void ConnectToMaster();
  int HandleMsg(PollConn* poll_conn_ptr);
  int HandleMasterMsg();
  int HandlePeerMsg(PollConn* poll_conn_ptr);
  int HandlePipeMsg(PollConn* poll_conn_ptr);
  void ConnectToPeers();
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
    peer_recv_mem_(config.kCommBuffCapacity*config.kNumExecutors),
    peer_(config.kNumExecutors),
    peer_conn_(config.kNumExecutors),
    host_info_(kNumExecutors) { }


ExecutorThread::~ExecutorThread() { }

void
ExecutorThread::operator() () {
  InitListener();

  PollConn listen_poll_conn = {&listen_, PollConn::ConnType::listen};
  int ret = event_handler_.AddPollConn(listen_.sock.get_fd(), &listen_poll_conn);
  CHECK_EQ(ret, 0);
  ConnectToMaster();

  PollConn master_poll_conn = {&master_, PollConn::ConnType::master};
  ret = event_handler_.AddPollConn(master_.sock.get_fd(), &master_poll_conn);
  CHECK_EQ(ret, 0);

  event_handler_.SetNoreadEventHandler(
      std::bind(&ExecutorThread::HandleConnection, this,
                std::placeholders::_1));

  event_handler_.SetClosedConnectionHandler(
      std::bind(&ExecutorThread::HandleClosedConnection, this,
                std::placeholders::_1));

  event_handler_.SetReadEventHandler(
      std::bind(&ExecutorThread::HandleMsg, this, std::placeholders::_1));

  while (true) {
    event_handler_.WaitAndHandleEvent();
    if (action_ == Action::kExit) break;
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
  CHECK_EQ(ret, 0);
}

void
ExecutorThread::HandleConnection(PollConn* poll_conn_ptr) {
  conn::Socket accepted;
  listen_.sock.Accept(&accepted);

  uint8_t *recv_mem = peer_recv_mem_.data()
                      + kCommBuffCapacity*num_connected_peers_;

  auto *sock_conn = new conn::SocketConn(
      accepted, recv_mem, kCommBuffCapacity);

  auto &curr_poll_conn = peer_conn_[num_connected_peers_];
  curr_poll_conn.conn = sock_conn;
  curr_poll_conn.type = PollConn::ConnType::peer;
  int ret = event_handler_.AddPollConn(accepted.get_fd(), &curr_poll_conn);
  CHECK_EQ(ret, 0);
  num_connected_peers_++;
}

void
ExecutorThread::HandleClosedConnection(PollConn *poll_conn_ptr) {
  auto type = poll_conn_ptr->type;
  if (type == PollConn::ConnType::listen
      || type == PollConn::ConnType::pipe
      || type == PollConn::ConnType::peer) {
    int fd = poll_conn_ptr->get_fd();
    event_handler_.RemovePollConn(fd);
  } else {
    int fd = poll_conn_ptr->get_fd();
    event_handler_.RemovePollConn(fd);
    auto* conn = reinterpret_cast<conn::SocketConn*>(poll_conn_ptr->conn);
    conn->sock.Close();
    action_ = Action::kExit;
  }
}

void
ExecutorThread::ConnectToMaster() {
  uint32_t ip;
  int ret = GetIPFromStr(kMasterIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = master_.sock.Connect(ip, kMasterPort);
  CHECK(ret == 0) << "executor failed connecting to master";

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
          size_t sent_size = master_.sock.Send(&send_buff_);
          CHECK(conn::CheckSendSize(send_buff_, sent_size));
          action_ = Action::kNone;
        }
      case Action::kExit:
        break;
      default:
        LOG(FATAL) << "unknown";
    }
  }
  return ret;
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
          bool recv = sock.Recv(&recv_buff, host_info_.data() + host_info_recved_size_);
          if (!recv) return 0;
          CHECK (!recv_buff.is_error()) << "driver error during receiving "
                                        << errno;
          CHECK (!recv_buff.EOFAtIncompleteMsg()) << "driver error : early EOF";
          host_info_recved_size_ = recv_buff.get_next_recved_size();
        } else {
          auto *msg = message::Helper::get_msg<message::ExecutorConnectToPeers>(
              recv_buff);
          recv_buff.set_next_expected_size(msg->num_executors*sizeof(HostInfo));
          if (recv_buff.get_size() > recv_buff.get_expected_size()) {
            size_t size_to_copy = recv_buff.get_size()
                                  - recv_buff.get_expected_size();
            memcpy(host_info_.data(),
                   recv_buff.get_mem() + recv_buff.get_expected_size(),
                   size_to_copy);
            host_info_recved_size_ += size_to_copy;
            recv_buff.IncNextRecvedSize(size_to_copy);
          }
        }
        if (recv_buff.ReceivedFullNextMsg()) {
          LOG(INFO) << "got all host info";
          ret = 2;
          action_ = Action::kConnectToPeers;
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
ExecutorThread::HandlePeerMsg(PollConn* poll_conn_ptr) {
  auto &recv_buff = poll_conn_ptr->get_recv_buff();

  auto msg_type = message::Helper::get_type(recv_buff);
  int ret = 1;
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
        ret = 1;
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

void
ExecutorThread::ConnectToPeers() {
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
                                        kCommBuffCapacity));
    peer_conn_[i].conn = peer_[i].get();
    peer_conn_[i].type = PollConn::ConnType::peer;

    size_t sent_size = peer_[i]->sock.Send(&send_buff_);
    CHECK(conn::CheckSendSize(send_buff_, sent_size));
    int ret = event_handler_.AddPollConn(peer_sock.get_fd(), &peer_conn_[i]);
    CHECK_EQ(ret, 0);
  }
}

}
}
