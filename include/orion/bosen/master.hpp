#pragma once

#include <vector>
#include <stdint.h>
#include <pthread.h>
#include <thread>

#include <orion/noncopyable.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/server.hpp>
#include <orion/bosen/client.hpp>
#include <orion/bosen/worker_runtime.hpp>
#include <orion/bosen/message.hpp>
#include <orion/constants.hpp>
#include <orion/helper.hpp>
#include <climits>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <memory>

namespace orion {
namespace bosen {

class Master {
 private:
  enum class State {
    kConnectToDriver = 0,
      kSetUpLocalThreads = 1,
      kExecutorReady = 2,
      kConnectToPeers = 3,
      kPeerConnecting = 4,
      kRunning = 5,
      kExit = 6
  };

  struct PollConn {
    enum class ConnType {
      listen = 0,
        driver = 1,
        pipe = 2
    };
    void *conn;
    ConnType type;
  };

  const size_t comm_buff_capacity_;
  conn::Poll poll_;
  std::unique_ptr<uint8_t[]> driver_recv_mem_;
  conn::SocketConn driver_;
  conn::Socket listen_;
  std::unique_ptr<uint8_t[]> send_mem_;
  conn::SendBuffer send_buff_;
  State state_;

  const std::string driver_ip_;
  const int32_t driver_port_;
  const std::string listen_ip_;
  const int32_t listen_port_;

  std::unique_ptr<conn::Socket[]> clients_;
  size_t num_connected_clients_ {0};
  std::unique_ptr<HostInfo[]> hosts_;
  const int32_t my_id_ {0};
  size_t num_executors_ {0};
  size_t num_executors_recved_ {0};
  // 0: master <> server :1
  conn::Pipe pipe_mse_[2];
  // 0: master <> client :1
  conn::Pipe pipe_mcl_[2];
  // 0: server <> client :1
  conn::Pipe pipe_scl_[2];

  std::unique_ptr<uint8_t[]> pipe_recv_mem_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<std::thread> server_thread_;
  std::unique_ptr<conn::PipeConn> server_pipe_conn_;
  PollConn server_poll_conn_;

  std::unique_ptr<Client> client_;
  std::unique_ptr<std::thread> client_thread_;
  std::unique_ptr<conn::PipeConn> client_pipe_conn_;
  PollConn client_poll_conn_;

  std::unique_ptr<WorkerRuntime> runtime_;
  const size_t num_local_executors_ {0};
  size_t num_peer_connected_threads_ {0};
 public:
  Master(
      size_t comm_buff_capacity,
      const std::string &driver_ip,
      int32_t driver_port,
      const std::string &master_ip,
      int32_t master_port,
      int32_t executor_id,
      size_t num_local_executors);
  virtual ~Master();
  DISALLOW_COPY(Master);

  void operator() ();

 private:
  void InitListener();
  void HandleConnection(PollConn* poll_conn_ptr);
  void ConnectToDriver();
  conn::RecvBuffer& HandleReadEvent(PollConn *poll_conn_ptr);
  bool HandleDriverMsg();
  bool HandlePipeMsg(PollConn *poll_conn_ptr);
  void SetUpLocalThreads();
  void Stop();
};

Master::Master(
      size_t comm_buff_capacity,
      const std::string &driver_ip,
      int32_t driver_port,
      const std::string &master_ip,
      int32_t master_port,
      int32_t executor_id,
      size_t num_local_executors):
    comm_buff_capacity_(comm_buff_capacity),
    driver_recv_mem_(std::make_unique<uint8_t[]>(comm_buff_capacity)),
    driver_(conn::Socket(),
            driver_recv_mem_.get(),
            comm_buff_capacity),
    send_mem_(std::make_unique<uint8_t[]>(comm_buff_capacity)),
    send_buff_(send_mem_.get(), comm_buff_capacity),
  state_(State::kConnectToDriver),
  driver_ip_(driver_ip),
  driver_port_(driver_port),
  listen_ip_(master_ip),
  listen_port_(master_port),
  my_id_(executor_id),
  num_local_executors_(num_local_executors) { }

Master::~Master() { }

void
Master::InitListener () {
  listen_.Bind(listen_ip_.c_str(), (uint16_t) listen_port_);
  listen_.Listen(64);
}

void
Master::HandleConnection(
    PollConn* poll_conn_ptr) {
  listen_.Accept(&(clients_[num_connected_clients_]));
  num_connected_clients_++;
  if (num_connected_clients_ == num_executors_ - 1) {
    server_ = std::make_unique<Server>(
        clients_.get(), num_executors_, pipe_mse_[1],
        pipe_scl_[0],
        comm_buff_capacity_,
        runtime_.get(),
        my_id_);
    server_thread_ = std::make_unique<std::thread>(
        &Server::operator(),
        server_.get());
  }

  if (num_connected_clients_ == num_executors_ - 1
      && num_peer_connected_threads_ == 2) {
    LOG(INFO) << my_id_ << " Master ConnectToPeers done";
    message::Helper::CreateMsg<message::ExecutorConnectToPeersAck>(
        &send_buff_);
    driver_.sock.Send(&send_buff_);
    state_ = State::kRunning;
  }
}

void
Master::ConnectToDriver() {
  int ret = -1;
  ret = driver_.sock.Connect(driver_ip_.c_str(),
                             (uint16_t) driver_port_);
  CHECK(ret == 0) << "executor has connected";

  auto *host_info_msg
      = message::Helper::CreateMsg<
        message::ExecutorHostInfo>(&send_buff_, my_id_);
  char *msg_ip = host_info_msg->host_info.ip;
  memcpy(msg_ip, listen_ip_.c_str(), listen_ip_.size());
  msg_ip[listen_ip_.size()] = '\0';
  host_info_msg->host_info.port = (uint16_t) listen_port_;
  size_t nsent = driver_.sock.Send(&send_buff_);
  CHECK(conn::CheckSendSize(send_buff_, nsent));
}

void
Master::SetUpLocalThreads() {
  LOG(INFO) << my_id_ << " Master " << __func__;
  runtime_ = std::make_unique<WorkerRuntime>(num_executors_,
                                        num_local_executors_,
                                        my_id_);
  int ret = conn::Pipe::CreateBiPipe(pipe_mse_);
  CHECK(ret == 0);
  ret = conn::Pipe::CreateBiPipe(pipe_mcl_);
  CHECK(ret == 0);
  ret = conn::Pipe::CreateBiPipe(pipe_scl_);
  CHECK(ret == 0);

  pipe_recv_mem_ = std::make_unique<uint8_t[]>(comm_buff_capacity_ * 3);
  server_pipe_conn_ = std::make_unique<conn::PipeConn>(pipe_mse_[0], pipe_recv_mem_.get(),
                                            comm_buff_capacity_);
  client_pipe_conn_ = std::make_unique<conn::PipeConn>(pipe_mcl_[0],
                                            pipe_recv_mem_.get() + comm_buff_capacity_,
                                            comm_buff_capacity_);

  server_poll_conn_ = {server_pipe_conn_.get(), PollConn::ConnType::pipe};
  client_poll_conn_ = {client_pipe_conn_.get(), PollConn::ConnType::pipe};

  poll_.Add(pipe_mse_[0].get_read_fd(), &server_poll_conn_);
  poll_.Add(pipe_mcl_[0].get_read_fd(), &client_poll_conn_);

  client_ = std::make_unique<Client>(
      hosts_.get(), num_executors_, pipe_mcl_[1],
      pipe_scl_[1],
      comm_buff_capacity_,
      runtime_.get(),
      my_id_);

  client_thread_ = std::make_unique<std::thread>(
      &Client::operator(),
      client_.get());
}

bool
Master::HandleDriverMsg() {
  auto &sock = driver_.sock;
  auto &recv_buff = driver_.recv_buff;

  if (!recv_buff.ReceivedFullMsg()) {
    bool recv = sock.Recv(&recv_buff);
    if (!recv) return false;
  }

  CHECK (!recv_buff.is_error()) << "master error during receiving " << errno;
  CHECK (!recv_buff.EOFAtIncompleteMsg()) << "master error : early EOF";
  // maybe EOF but not received anything
  if (!recv_buff.ReceivedFullMsg()) return false;

  auto msg_type = message::Helper::get_type(recv_buff);
  switch (msg_type) {
    case message::Type::kExecutorInfo:
      {
        auto *msg = message::Helper::get_msg<message::ExecutorInfo>(recv_buff);
        if (num_executors_ == 0) {
          num_executors_ = msg->num_total_executors;
          hosts_ = std::make_unique<HostInfo[]>(num_executors_);
          clients_ = std::make_unique<conn::Socket[]>(num_executors_);
        }
        memcpy(reinterpret_cast<uint8_t*>(hosts_.get()) + num_executors_recved_ * sizeof(HostInfo),
               msg->get_host_info_mem(),
               sizeof(HostInfo) * msg->num_executors);
        num_executors_recved_ += msg->num_executors;
        if (num_executors_recved_ == num_executors_) {
          state_ = State::kSetUpLocalThreads;
        }
      }
      break;
    case message::Type::kExecutorConnectToPeers:
      {
        send_buff_.Copy(recv_buff);
        client_pipe_conn_->pipe.Send(&send_buff_);
        state_ = State::kPeerConnecting;
      }
      break;
    case message::Type::kExecutorStop:
      {
        LOG(INFO) << my_id_ << " Master handling Driver message ExecutorStop";
        CHECK(state_ == State::kRunning) << "state_ = " << static_cast<int>(state_);
        state_ = State::kExit;
      }
      break;
    default:
      {
        LOG(FATAL) << "unknown message type " << static_cast<int>(msg_type);
      }
      break;
  }
  return true;
}

bool
Master::HandlePipeMsg(PollConn *poll_conn_ptr) {
  auto &pipe = reinterpret_cast<conn::PipeConn*>(
      poll_conn_ptr->conn)->pipe;
  auto &recv_buff = reinterpret_cast<conn::PipeConn*>(
      poll_conn_ptr->conn)->recv_buff;

  if (!recv_buff.ReceivedFullMsg()) {
    bool recv = pipe.Recv(&recv_buff);
    if (!recv) return false;
  }

  CHECK (!recv_buff.is_error()) << "master error during receiving " << errno;
  CHECK (!recv_buff.EOFAtIncompleteMsg()) << "master error : early EOF";
  // maybe EOF but not received anything
  if (!recv_buff.ReceivedFullMsg()) return false;

  auto msg_type = message::Helper::get_type(recv_buff);
  switch (msg_type) {
    case message::Type::kExecutorConnectToPeersAck:
      {
        num_peer_connected_threads_++;
        if (num_peer_connected_threads_ == 2
            && num_connected_clients_ == num_executors_ - 1) {
          LOG(INFO) << my_id_ << " Master ConnectToPeers done";
          message::Helper::CreateMsg<message::ExecutorConnectToPeersAck>(
              &send_buff_);
          driver_.sock.Send(&send_buff_);
          state_ = State::kRunning;
        }
      }
      break;
    default:
      LOG(FATAL) << "unknown message!";
      break;
  }

  return true;
}

conn::RecvBuffer &
Master::HandleReadEvent(PollConn *poll_conn_ptr) {
  bool next_message = false;
  conn::RecvBuffer *recv_buff_ptr = nullptr;
  switch (poll_conn_ptr->type) {
    case PollConn::ConnType::driver:
      {
        next_message = HandleDriverMsg();
        recv_buff_ptr = &driver_.recv_buff;
      }
      break;
    case PollConn::ConnType::pipe:
      {
        next_message = HandlePipeMsg(poll_conn_ptr);
        recv_buff_ptr = &(reinterpret_cast<conn::PipeConn*>(
            poll_conn_ptr->conn)->recv_buff);
      }
      break;
    default:
      LOG(FATAL) << "unknown ";
  }

  if (next_message) {
    recv_buff_ptr->ClearOneMsg();
  }
  return *recv_buff_ptr;
}

void
Master::Stop() {
  message::Helper::CreateMsg<message::ExecutorStop>(&send_buff_);
  server_pipe_conn_->pipe.Send(&send_buff_);
  client_pipe_conn_->pipe.Send(&send_buff_);
  if (server_thread_.get() != nullptr)
    server_thread_->join();
  client_thread_->join();

  pipe_mse_[0].Close();
  pipe_mcl_[0].Close();

  pipe_mse_[1].Close();
  pipe_mcl_[1].Close();

  pipe_scl_[0].Close();
  pipe_scl_[1].Close();

  driver_.sock.Close();
  listen_.Close();

  for (size_t i = 0; i < num_executors_; i++) {
    if (clients_[i].get_fd() != 0) {
      clients_[i].Close();
    }
  }
  LOG(INFO) << my_id_ << " master exiting";
}

void
Master::operator() () {
  static constexpr size_t kNumEvents = 100;
  InitListener();

  int ret = -1;
  ret = poll_.Init();
  CHECK(ret == 0) << "poll init failed";
  PollConn listen_poll_conn = {&listen_, PollConn::ConnType::listen};
  poll_.Add(listen_.get_fd(), &listen_poll_conn);

  ConnectToDriver();

  PollConn driver_poll_conn = {&driver_, PollConn::ConnType::driver};
  poll_.Add(driver_.sock.get_fd(), &driver_poll_conn);
  epoll_event es[kNumEvents];

  while (1) {
    int num_events = poll_.Wait(es, kNumEvents);
    CHECK(num_events > 0);
    for (int i = 0; i < num_events; ++i) {
      PollConn *poll_conn_ptr = poll_.EventConn<PollConn>(es, i);
      if (es[i].events & EPOLLIN) {
        if (poll_conn_ptr->type == PollConn::ConnType::listen) {
          HandleConnection(poll_conn_ptr);
        } else {
          auto &recv_buff = HandleReadEvent(poll_conn_ptr);
          // repeat until receive buffer is exhausted
          while (recv_buff.ReceivedFullMsg()) {
            HandleReadEvent(poll_conn_ptr);
          }
        }
      } else {
        LOG(WARNING) << "something unknown happend: "
                  << es[i].events;
        CHECK(false);
      }
    }
    switch(state_) {
      case State::kSetUpLocalThreads:
        {
          SetUpLocalThreads();
          message::Helper::CreateMsg<
            message::ExecutorReady>(&send_buff_);
          size_t nsent = driver_.sock.Send(&send_buff_);
          CHECK(conn::CheckSendSize(send_buff_, nsent));
          state_ = State::kExecutorReady;
        }
        break;
      case State::kExit:
        {
          Stop();
        }
        break;
      default:
        break;
    }
    if (state_ == State::kExit) break;
  }
  runtime_->PrintPerfCounts();
}

}
}
