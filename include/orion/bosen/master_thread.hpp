#pragma once

#include <memory>
#include <iostream>
#include <vector>

#include <orion/bosen/config.hpp>
#include <orion/noncopyable.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/host_info.hpp>
#include <orion/bosen/util.hpp>
#include <orion/bosen/message.hpp>

namespace orion {
namespace bosen {

class MasterThread {
 private:
  static constexpr const char *kMasterReadyString = "Master is ready!";
  struct PollConn {
    enum class ConnType {
      listen = 0,
        executor = 1
    };

    void *conn;
    ConnType type;
  };

  enum class State {
    kInitialization = 0,
      kRunning = 1
  };

  enum class Action {
    kNone = 0,
      kExecutorConnectToPeers = 1
  };

  const size_t kNumExecutors;
  const size_t kCommBuffCapacity;
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  conn::Poll poll_;
  conn::Socket listen_;
  conn::Socket client_;
  std::vector<std::unique_ptr<conn::SocketConn>> executors_;
  std::vector<PollConn> poll_conns_;
  std::vector<HostInfo> host_info_;
  std::vector<uint8_t> recv_mem_;
  std::vector<uint8_t> send_mem_;
  conn::SendBuffer send_buff_;

  size_t num_accepted_executors_ {0};
  size_t num_identified_executors_ {0};
  size_t num_started_executors_ {0};
  size_t num_closed_conns_ {0};

  State state_ {State::kInitialization};
  Action action_ {Action::kNone};

 public:
  MasterThread(const Config &config);
  ~MasterThread();
  DISALLOW_COPY(MasterThread);
  void operator() ();
 private:
  void InitListener();
  void HandleConnection(PollConn* poll_conn_ptr);
  void HandleClosedConnection(PollConn *poll_conn_ptr);
  conn::RecvBuffer& HandleReadEvent(PollConn* poll_conn_ptr);
  bool HandleExecutorMsg(PollConn *poll_conn_ptr);
  uint8_t* AllocRecvMem();
};

MasterThread::MasterThread(const Config &config):
    kNumExecutors(config.kNumExecutors),
    kCommBuffCapacity(config.kCommBuffCapacity),
    kMasterIp(config.kMasterIp),
    kMasterPort(config.kMasterPort),
    executors_(kNumExecutors),
    poll_conns_(kNumExecutors),
    host_info_(kNumExecutors),
    recv_mem_((kNumExecutors + 1) * config.kCommBuffCapacity),
    send_mem_(config.kCommBuffCapacity),
    send_buff_(send_mem_.data(), config.kCommBuffCapacity) { }

MasterThread::~MasterThread() { }

void
MasterThread::operator() () {
  static const size_t kNumEvents = 100;
  LOG(INFO) << "MasterThread is started";
  InitListener();
  int ret = poll_.Init();
  CHECK_EQ(ret, 0) << "poll init failed";

  PollConn listen_poll_conn = {&listen_, PollConn::ConnType::listen};
  poll_.Add(listen_.get_fd(), &listen_poll_conn);

  std::cout << kMasterReadyString << std::endl;
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
          while (recv_buff.ReceivedFullMsg()
                 && (!recv_buff.IsExepectingNextMsg())) {
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
  }

}

void
MasterThread::InitListener() {
  uint32_t ip;
  int ret = GetIPFromStr(kMasterIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = listen_.Bind(ip, kMasterPort);
  CHECK_EQ(ret, 0);
  ret = listen_.Listen(kNumExecutors + 1);
  CHECK_EQ(ret, 0);
}

/*
 * For each incoming connection, do the following:
 * 1) accept the connection to get the socket;
 * 2) allocate memory for receive buffer and create socket conn;
 * 3) add the socket conn to poll
 */
void
MasterThread::HandleConnection(PollConn *poll_conn_ptr) {
  conn::Socket accepted;
  listen_.Accept(&accepted);
  LOG(INFO) << "MasterThread " << __func__ << " socket fd = "
	    << accepted.get_fd();

  uint8_t *recv_mem = AllocRecvMem();
  auto *sock_conn = new conn::SocketConn(
      accepted, recv_mem, kCommBuffCapacity);

  auto &curr_poll_conn
      = poll_conns_[num_accepted_executors_];

  curr_poll_conn.conn = sock_conn;
  curr_poll_conn.type = PollConn::ConnType::executor;

  poll_.Add(accepted.get_fd(), &curr_poll_conn);
  num_accepted_executors_++;
}

void
MasterThread::HandleClosedConnection(PollConn *poll_conn_ptr) {
  num_closed_conns_++;
  auto &sock_conn = *reinterpret_cast<conn::SocketConn*>(
      poll_conn_ptr->conn);
  auto &sock = sock_conn.sock;
  poll_.Remove(sock.get_fd());
  sock.Close();
}

conn::RecvBuffer &
MasterThread::HandleReadEvent(PollConn *poll_conn_ptr) {
  bool next_message = HandleExecutorMsg(poll_conn_ptr);
  auto &recv_buff = reinterpret_cast<conn::SocketConn*>(
          poll_conn_ptr->conn)->recv_buff;
  if (next_message) {
    recv_buff.ClearOneMsg();
  }
  return recv_buff;
}

bool
MasterThread::HandleExecutorMsg(PollConn *poll_conn_ptr) {
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
    case message::Type::kExecutorIdentity:
      {

        auto *msg = message::Helper::get_msg<message::ExecutorIdentity>(recv_buff);
        //LOG(INFO) << "ExecutorIdentity from " << msg->executor_id;
        host_info_[msg->executor_id] = msg->host_info;
        executors_[msg->executor_id].reset(&sock_conn);
        num_identified_executors_++;
        if (state_ == State::kInitialization
            && (num_identified_executors_ == kNumExecutors)) {
          action_ = Action::kExecutorConnectToPeers;
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown message type " << static_cast<int>(msg_type)
                   << " from " << sock.get_fd();
  }

  switch (action_) {
    case Action::kExecutorConnectToPeers:
      break;
    case Action::kNone:
      break;
    default:
      LOG(FATAL) << "unknown";
  }
  return true;
}

uint8_t*
MasterThread::AllocRecvMem() {
  return recv_mem_.data()
      + num_accepted_executors_
      * kCommBuffCapacity;
}

} // end namespace bosen
} // end namespace orion
