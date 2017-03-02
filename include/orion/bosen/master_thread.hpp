#pragma once

#include <memory>
#include <iostream>
#include <vector>
#include <functional>

#include <orion/bosen/config.hpp>
#include <orion/noncopyable.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/host_info.hpp>
#include <orion/bosen/util.hpp>
#include <orion/bosen/message.hpp>
#include <orion/bosen/event_handler.hpp>

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

    conn::SocketConn *conn;
    ConnType type;

    bool Receive() {
      return conn->sock.Recv(&(conn->recv_buff));
    }

    conn::RecvBuffer& get_recv_buff() {
      return conn->recv_buff;
    }
    bool is_noread_event() const {
      return type == ConnType::listen;
    }
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
  std::vector<uint8_t> recv_mem_;
  std::vector<uint8_t> send_mem_;
  EventHandler<PollConn> event_handler_;
  conn::SocketConn listen_;
  conn::SocketConn client_;
  std::vector<std::unique_ptr<conn::SocketConn>> executors_;
  std::vector<PollConn> poll_conns_;
  std::vector<HostInfo> host_info_;
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
  int HandleExecutorMsg(PollConn *poll_conn_ptr);

  void BroadcastAllExecutors();
  void BroadcastAllExecutors(void *mem, size_t mem_size);
  uint8_t* AllocRecvMem();
};

MasterThread::MasterThread(const Config &config):
    kNumExecutors(config.kNumExecutors),
    kCommBuffCapacity(config.kCommBuffCapacity),
    kMasterIp(config.kMasterIp),
    kMasterPort(config.kMasterPort),
    recv_mem_((kNumExecutors + 2) * config.kCommBuffCapacity),
    send_mem_(config.kCommBuffCapacity),
    listen_(conn::Socket(), recv_mem_.data(), kCommBuffCapacity),
    client_(conn::Socket(), recv_mem_.data() + kCommBuffCapacity,
            kCommBuffCapacity),
    executors_(kNumExecutors),
    poll_conns_(kNumExecutors),
    host_info_(kNumExecutors),
    send_buff_(send_mem_.data(), config.kCommBuffCapacity) { }

MasterThread::~MasterThread() { }

void
MasterThread::operator() () {
  LOG(INFO) << "MasterThread is started";
  InitListener();
  PollConn listen_poll_conn = {&listen_, PollConn::ConnType::listen};
  event_handler_.AddPollConn(listen_.sock.get_fd(), &listen_poll_conn);
  event_handler_.SetNoreadEventHandler(
      std::bind(&MasterThread::HandleConnection, this,
                std::placeholders::_1));

  event_handler_.SetReadEventHandler(
      std::bind(&MasterThread::HandleExecutorMsg, this,
               std::placeholders::_1));

  event_handler_.SetClosedConnectionHandler(
      std::bind(&MasterThread::HandleClosedConnection, this,
                std::placeholders::_1));

  std::cout << kMasterReadyString << std::endl;
  while (1) {
    event_handler_.WaitAndHandleEvent();
  }
}

void
MasterThread::InitListener() {
  uint32_t ip;
  int ret = GetIPFromStr(kMasterIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = listen_.sock.Bind(ip, kMasterPort);
  CHECK_EQ(ret, 0);
  ret = listen_.sock.Listen(kNumExecutors + 1);
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
  listen_.sock.Accept(&accepted);
  LOG(INFO) << "MasterThread " << __func__ << " socket fd = "
	    << accepted.get_fd();

  uint8_t *recv_mem = AllocRecvMem();
  auto *sock_conn = new conn::SocketConn(
      accepted, recv_mem, kCommBuffCapacity);

  auto &curr_poll_conn
      = poll_conns_[num_accepted_executors_];

  curr_poll_conn.conn = sock_conn;
  curr_poll_conn.type = PollConn::ConnType::executor;

  event_handler_.AddPollConn(accepted.get_fd(), &curr_poll_conn);
  num_accepted_executors_++;
}

void
MasterThread::HandleClosedConnection(PollConn *poll_conn_ptr) {
  num_closed_conns_++;
  auto *sock_conn = poll_conn_ptr->conn;
  auto &sock = sock_conn->sock;
  event_handler_.RemovePollConn(sock.get_fd());
  sock.Close();
}

int
MasterThread::HandleExecutorMsg(PollConn *poll_conn_ptr) {
  auto &recv_buff = poll_conn_ptr->get_recv_buff();

  auto msg_type = message::Helper::get_type(recv_buff);
  switch (msg_type) {
    case message::Type::kExecutorIdentity:
      {
        auto *msg = message::Helper::get_msg<message::ExecutorIdentity>(recv_buff);
        //LOG(INFO) << "ExecutorIdentity from " << msg->executor_id;
        host_info_[msg->executor_id] = msg->host_info;
        auto* sock_conn = poll_conn_ptr->conn;
        executors_[msg->executor_id].reset(sock_conn);
        num_identified_executors_++;
        if (state_ == State::kInitialization
            && (num_identified_executors_ == kNumExecutors)) {
          action_ = Action::kExecutorConnectToPeers;
        }
      }
      break;
    default:
      {
        auto& sock = poll_conn_ptr->conn->sock;
        LOG(FATAL) << "Unknown message type " << static_cast<int>(msg_type)
                   << " from " << sock.get_fd();
      }
  }

  switch (action_) {
    case Action::kExecutorConnectToPeers:
      {
        message::Helper::CreateMsg<message::ExecutorConnectToPeers>(
            &send_buff_, kNumExecutors);
        BroadcastAllExecutors(host_info_.data(),
                              kNumExecutors*sizeof(HostInfo));

      }
      break;
    case Action::kNone:
      break;
    default:
      LOG(FATAL) << "unknown";
  }

  return true;
}

void
MasterThread::BroadcastAllExecutors() {
  for (int i = 0; i < kNumExecutors; ++i) {
    size_t nsent = executors_[i]->sock.Send(&send_buff_);
    CHECK(conn::CheckSendSize(send_buff_, nsent)) << "send only " << nsent;
  }
}

void
MasterThread::BroadcastAllExecutors(void *mem, size_t mem_size) {
  for (int i = 0; i < kNumExecutors; ++i) {
    size_t nsent = executors_[i]->sock.Send(&send_buff_);
    LOG(INFO) << "send size = " << nsent;
    CHECK(conn::CheckSendSize(send_buff_, nsent)) << "send only " << nsent;
    nsent = executors_[i]->sock.Send(mem, mem_size);
    CHECK_EQ(nsent, mem_size);
  }
}

uint8_t*
MasterThread::AllocRecvMem() {
  return recv_mem_.data()
      + (num_accepted_executors_ + 2)
      * kCommBuffCapacity;
}

} // end namespace bosen
} // end namespace orion
