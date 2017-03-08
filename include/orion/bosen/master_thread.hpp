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
        executor = 1,
        driver = 2
    };

    conn::SocketConn *conn;
    ConnType type;

    bool Receive() {
      return conn->sock.Recv(&(conn->recv_buff));
    }

    bool Send() {
      return conn->sock.Send(&(conn->send_buff));
    }

    conn::RecvBuffer& get_recv_buff() {
      return conn->recv_buff;
    }
    conn::SendBuffer& get_send_buff() {
      return conn->send_buff;
    }
    bool is_connect_event() const {
      return type == ConnType::listen;
    }
    int get_read_fd() const {
      return conn->sock.get_fd();
    }
    int get_write_fd() const {
      return conn->sock.get_fd();
    }
  };

  enum class State {
    kInitialization = 0,
      kRunning = 1
  };

  enum class Action {
    kNone = 0,
      kExit = 1,
      kExecutorConnectToPeers = 2
  };

  const size_t kNumExecutors;
  const size_t kCommBuffCapacity;
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  std::vector<uint8_t> recv_mem_;
  std::vector<uint8_t> send_mem_;
  EventHandler<PollConn> event_handler_;
  conn::SocketConn listen_;
  conn::SocketConn driver_;
  std::vector<std::unique_ptr<conn::SocketConn>> executors_;
  std::vector<PollConn> poll_conns_;
  std::vector<HostInfo> host_info_;
  conn::SendBuffer send_buff_;

  size_t num_accepted_executors_ {0};
  size_t num_identified_executors_ {0};
  size_t num_ready_executors_ {0};
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
  int HandleClosedConnection(PollConn *poll_conn_ptr);
  conn::RecvBuffer& HandleReadEvent(PollConn* poll_conn_ptr);
  int HandleMsg(PollConn *poll_conn_ptr);
  int HandleDriverMsg(PollConn *poll_conn_ptr);
  int HandleExecutorMsg(PollConn *poll_conn_ptr);
  void BroadcastAllExecutors();
};

MasterThread::MasterThread(const Config &config):
    kNumExecutors(config.kNumExecutors),
    kCommBuffCapacity(config.kCommBuffCapacity),
    kMasterIp(config.kMasterIp),
    kMasterPort(config.kMasterPort),
    recv_mem_((kNumExecutors + 2) * config.kCommBuffCapacity),
    send_mem_((kNumExecutors + 3) * config.kCommBuffCapacity),
    listen_(conn::Socket(),
            recv_mem_.data(),
            send_mem_.data(),
            kCommBuffCapacity),
    driver_(conn::Socket(),
            recv_mem_.data() + kCommBuffCapacity,
            send_mem_.data() + kCommBuffCapacity,
            kCommBuffCapacity),
    executors_(kNumExecutors),
    poll_conns_(kNumExecutors),
    host_info_(kNumExecutors),
    send_buff_(send_mem_.data() + 2*kCommBuffCapacity,
               config.kCommBuffCapacity) { }

MasterThread::~MasterThread() { }

void
MasterThread::operator() () {
  LOG(INFO) << "MasterThread is started";
  InitListener();
  PollConn listen_poll_conn = {&listen_, PollConn::ConnType::listen};
  event_handler_.SetToReadOnly(&listen_poll_conn);
  event_handler_.SetConnectEventHandler(
      std::bind(&MasterThread::HandleConnection, this,
                std::placeholders::_1));

  event_handler_.SetReadEventHandler(
      std::bind(&MasterThread::HandleMsg, this,
               std::placeholders::_1));

  event_handler_.SetClosedConnectionHandler(
      std::bind(&MasterThread::HandleClosedConnection, this,
                std::placeholders::_1));

  event_handler_.SetDefaultWriteEventHandler();

  std::cout << kMasterReadyString << std::endl;
  while (true) {
    event_handler_.WaitAndHandleEvent();
    if (action_ == Action::kExit) break;
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

  uint8_t *recv_mem = recv_mem_.data()
                      + (num_accepted_executors_ + 2)
                      * kCommBuffCapacity;
  uint8_t *send_mem = send_mem_.data()
                      + (num_accepted_executors_ + 3)
                      * kCommBuffCapacity;

  auto *sock_conn = new conn::SocketConn(
      accepted, recv_mem, send_mem, kCommBuffCapacity);

  auto &curr_poll_conn
      = poll_conns_[num_accepted_executors_];

  curr_poll_conn.conn = sock_conn;
  curr_poll_conn.type = PollConn::ConnType::executor;

  event_handler_.SetToReadOnly(&curr_poll_conn);
  num_accepted_executors_++;
}

int
MasterThread::HandleClosedConnection(PollConn *poll_conn_ptr) {
  num_closed_conns_++;
  auto *sock_conn = poll_conn_ptr->conn;
  auto &sock = sock_conn->sock;
  event_handler_.Remove(poll_conn_ptr);
  sock.Close();
  return EventHandler<PollConn>::kNoAction;
}

int
MasterThread::HandleMsg(PollConn *poll_conn_ptr) {
  int ret = EventHandler<PollConn>::kNoAction;
  if (poll_conn_ptr->type == PollConn::ConnType::executor) {
    ret = HandleExecutorMsg(poll_conn_ptr);
  } else {
    ret = HandleDriverMsg(poll_conn_ptr);
  }

  while (action_ != Action::kNone
         && action_ != Action::kExit) {
    switch (action_) {
      case Action::kExecutorConnectToPeers:
        {
          message::Helper::CreateMsg<message::ExecutorConnectToPeers>(
              &send_buff_, kNumExecutors);
          send_buff_.set_next_to_send(host_info_.data(),
                                      kNumExecutors*sizeof(HostInfo),
                                      nullptr);
          BroadcastAllExecutors();
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
MasterThread::HandleDriverMsg(PollConn *poll_conn_ptr) {
  return EventHandler<PollConn>::kClearOneMsg;
}

int
MasterThread::HandleExecutorMsg(PollConn *poll_conn_ptr) {
  auto &recv_buff = poll_conn_ptr->get_recv_buff();

  auto msg_type = message::Helper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kClearOneMsg;
  switch (msg_type) {
    case message::Type::kExecutorIdentity:
      {
        auto *msg = message::Helper::get_msg<message::ExecutorIdentity>(recv_buff);
        host_info_[msg->executor_id] = msg->host_info;
        auto* sock_conn = poll_conn_ptr->conn;
        executors_[msg->executor_id].reset(sock_conn);
        num_identified_executors_++;
        if (state_ == State::kInitialization
            && (num_identified_executors_ == kNumExecutors)) {
          action_ = Action::kExecutorConnectToPeers;
        }
        ret = EventHandler<PollConn>::kClearOneMsg;
      }
      break;
    case message::Type::kExecutorConnectToPeersAck:
      {
        num_ready_executors_++;
        if (num_ready_executors_ == kNumExecutors) {
          LOG(INFO) << "exit!";
          action_ = Action::kExit;
        }
        ret = EventHandler<PollConn>::kClearOneMsg;
      }
      break;
    default:
      {
        auto& sock = poll_conn_ptr->conn->sock;
        LOG(FATAL) << "Unknown message type " << static_cast<int>(msg_type)
                   << " from " << sock.get_fd();
      }
  }
  return ret;
}

void
MasterThread::BroadcastAllExecutors() {
  for (int i = 0; i < kNumExecutors; ++i) {
    conn::SendBuffer& send_buff = executors_[i]->send_buff;
    if (send_buff.get_remaining_to_send_size() > 0
        || send_buff.get_remaining_next_to_send_size() > 0) {
      bool sent = executors_[i]->sock.Send(&send_buff);
      while (!sent) {
        sent = executors_[i]->sock.Send(&send_buff);
      }
      send_buff.clear_to_send();
    }
    bool sent = executors_[i]->sock.Send(&send_buff_);
    if (!sent) {
      send_buff.Copy(send_buff_);
      event_handler_.SetToReadWrite(&poll_conns_[i]);
    }
    send_buff_.reset_sent_sizes();
  }
}

} // end namespace bosen
} // end namespace orion
