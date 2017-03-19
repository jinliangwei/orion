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
#include <orion/bosen/event_handler.hpp>
#include <orion/bosen/byte_buffer.hpp>

#include <orion/bosen/message.hpp>
#include <orion/bosen/driver_message.hpp>
#include <orion/bosen/execute_message.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/recv_arbitrary_bytes.hpp>

namespace orion {
namespace bosen {

class MasterThread {
 private:
  struct PollConn {
    enum class ConnType {
      listen = 0,
        executor = 1,
        driver = 2,
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
      kExecutorConnectToPeers = 2,
      kExecuteCodeOnOne = 3,
      kExecuteCodeOnAll = 4
  };

  const size_t kNumExecutors;
  const size_t kCommBuffCapacity;
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  EventHandler<PollConn> event_handler_;

  Blob listen_recv_mem_;
  Blob listen_send_mem_;
  conn::SocketConn listen_;
  PollConn listen_poll_conn_;

  Blob driver_recv_mem_;
  Blob driver_send_mem_;
  conn::SocketConn driver_;
  PollConn driver_poll_conn_;

  Blob executor_recv_mem_;
  Blob executor_send_mem_;
  std::vector<std::unique_ptr<conn::SocketConn>> executors_;
  std::vector<PollConn> executor_poll_conn_;

  Blob send_mem_;
  conn::SendBuffer send_buff_;
  std::vector<HostInfo> host_info_;
  ByteBuffer byte_buff_;
  int32_t executor_to_work_ {0};

  size_t num_accepted_executors_ {0};
  size_t num_identified_executors_ {0};
  size_t num_ready_executors_ {0};
  size_t num_closed_conns_ {0};
  bool stopped_all_ {false};

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
  void SendToExecutor(int executor_index);
};

MasterThread::MasterThread(const Config &config):
    kNumExecutors(config.kNumExecutors),
    kCommBuffCapacity(config.kCommBuffCapacity),
    kMasterIp(config.kMasterIp),
    kMasterPort(config.kMasterPort),
    listen_recv_mem_(config.kCommBuffCapacity),
    listen_send_mem_(config.kCommBuffCapacity),
    listen_(conn::Socket(),
            listen_recv_mem_.data(),
            listen_send_mem_.data(),
            config.kCommBuffCapacity),
    driver_recv_mem_(config.kCommBuffCapacity),
    driver_send_mem_(config.kCommBuffCapacity),
    driver_(conn::Socket(),
            driver_recv_mem_.data(),
            driver_send_mem_.data(),
            config.kCommBuffCapacity),
    executor_recv_mem_(config.kCommBuffCapacity*config.kNumExecutors),
    executor_send_mem_(config.kCommBuffCapacity*config.kNumExecutors),
    executors_(config.kNumExecutors),
    executor_poll_conn_(config.kNumExecutors),
    send_mem_(config.kCommBuffCapacity),
    send_buff_(send_mem_.data(), config.kCommBuffCapacity),
    host_info_(config.kNumExecutors) { }

MasterThread::~MasterThread() { }

void
MasterThread::operator() () {
  InitListener();
  listen_poll_conn_ = {&listen_, PollConn::ConnType::listen};
  event_handler_.SetToReadOnly(&listen_poll_conn_);
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

  std::cout << "Master is ready to receive connection from executors!"
            << std::endl;
  while (true) {
    event_handler_.WaitAndHandleEvent();
    if (action_ == Action::kExit) break;
  }
  LOG(INFO) << "master exiting!";
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
  if (num_accepted_executors_ < kNumExecutors) {
    uint8_t *recv_mem = executor_recv_mem_.data()
                        + num_accepted_executors_*kCommBuffCapacity;
    uint8_t *send_mem = executor_send_mem_.data()
                        + num_accepted_executors_*kCommBuffCapacity;
    auto *sock_conn = new conn::SocketConn(
        accepted, recv_mem, send_mem, kCommBuffCapacity);
    auto &curr_poll_conn
        = executor_poll_conn_[num_accepted_executors_];
    curr_poll_conn = {sock_conn, PollConn::ConnType::executor};
    event_handler_.SetToReadOnly(&curr_poll_conn);
    num_accepted_executors_++;
  } else {
    LOG(INFO) << "driver is connected";
    driver_.sock = accepted;
    driver_poll_conn_ = {&driver_, PollConn::ConnType::driver};
    event_handler_.SetToReadOnly(&driver_poll_conn_);
  }
}

int
MasterThread::HandleClosedConnection(PollConn *poll_conn_ptr) {
  int ret = EventHandler<PollConn>::kNoAction;
  if (poll_conn_ptr->type == PollConn::ConnType::driver) {
    LOG(INFO) << "Lost connection to driver";
    if (!stopped_all_) {
      LOG(INFO) << "Command executors to stop";
      message::Helper::CreateMsg<message::ExecutorStop>(&send_buff_);
      BroadcastAllExecutors();
      stopped_all_ = true;
    }
    driver_.sock.Close();
  } else {
    LOG(INFO) << "An executor has disconnected";
    num_closed_conns_++;
    auto *sock_conn = poll_conn_ptr->conn;
    auto &sock = sock_conn->sock;
    event_handler_.Remove(poll_conn_ptr);
    sock.Close();
  }
  if (num_closed_conns_ == kNumExecutors) {
    action_ = Action::kExit;
    ret = EventHandler<PollConn>::kExit;
  }
  return ret;
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
                                      kNumExecutors*sizeof(HostInfo));
          BroadcastAllExecutors();
          action_ = Action::kNone;
        }
        break;
      case Action::kExecuteCodeOnOne:
        {
          message::ExecuteMsgHelper::CreateMsg<
            message::ExecuteMsgExecuteCode>(
                &send_buff_, byte_buff_.GetSize());
          send_buff_.set_next_to_send(byte_buff_.GetBytes(),
                                      byte_buff_.GetSize());
          SendToExecutor(executor_to_work_);
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
  auto &recv_buff = poll_conn_ptr->get_recv_buff();
  auto msg_type = message::Helper::get_type(recv_buff);
  CHECK(msg_type == message::Type::kDriverMsg);

  auto driver_msg_type = message::DriverMsgHelper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kClearOneMsg;
  switch (driver_msg_type) {
    case message::DriverMsgType::kStop:
      {
        message::Helper::CreateMsg<message::ExecutorStop>(&send_buff_);
        BroadcastAllExecutors();
        stopped_all_ = true;
      }
      break;
    case message::DriverMsgType::kExecuteCodeOnOne:
      {
        auto *msg = message::DriverMsgHelper::get_msg<
            message::DriverMsgExecuteCodeOnOne>(recv_buff);
        size_t expected_size = msg->task_size;
        executor_to_work_ = msg->executor_id;
        bool received_next_msg
            = ReceiveArbitraryBytes(driver_.sock, &recv_buff, &byte_buff_,
                                    expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kExecuteCodeOnOne;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    default:
      auto& sock = poll_conn_ptr->conn->sock;
      LOG(FATAL) << "Unknown driver message type " << static_cast<int>(driver_msg_type)
                 << " from " << sock.get_fd();
  }
  return ret;
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
          std::cout << "Your Orion cluster is ready!" << std::endl;
          std::cout << "Connect your client application to "
                    << kMasterIp << ":" << kMasterPort << std::endl;
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
      event_handler_.SetToReadWrite(&executor_poll_conn_[i]);
    }
    send_buff_.reset_sent_sizes();
  }
}

void
MasterThread::SendToExecutor(int executor_index) {
  conn::SocketConn* executor = executors_[executor_index].get();
  conn::SendBuffer& send_buff = executor->send_buff;
  if (send_buff.get_remaining_to_send_size() > 0
      || send_buff.get_remaining_next_to_send_size() > 0) {
    bool sent = executor->sock.Send(&send_buff);
    while (!sent) {
      sent = executor->sock.Send(&send_buff);
    }
    send_buff.clear_to_send();
  }
  bool sent = executor->sock.Send(&send_buff_);
  if (!sent) {
    send_buff.Copy(send_buff_);
    event_handler_.SetToReadWrite(&executor_poll_conn_[executor_index]);
  }
  send_buff_.clear_to_send();
}

} // end namespace bosen
} // end namespace orion
