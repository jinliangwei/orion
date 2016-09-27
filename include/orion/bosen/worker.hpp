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

class Worker {
 public:
  enum class Event {
      kConnectedToPeers = 1
  };

 private:
  struct PollConn {
    enum class ConnType {
      listen = 0,
        pipe = 1
    };
    void *conn;
    ConnType type;
  };

  const size_t kCommBuffCapacity_;
  conn::Poll poll_;

  conn::Socket listen_;
  std::unique_ptr<uint8_t[]> send_mem_;
  conn::SendBuffer send_buff_;

  const int32_t kListenIp_;
  const int32_t kListenPort_;
  const int32_t kMyId_;

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
  const size_t kNumLocalExecutors_;
  const size_t kNumExecutors_;

  const std::unique_ptr<conn::Socket[]> kClients_;
  size_t num_connected_clients_ {0};
  const std::unique_ptr<HostInfo[]> kHosts_;

  size_t num_executors_recved_ {0};
  size_t num_peer_connected_threads_ {0};
 public:
  Worker(
      size_t comm_buff_capacity,
      uint64_t master_listen_ip,
      int32_t master_listen_port,
      int32_t executor_id,
      size_t num_local_executors,
      size_t num_total_executors,
      const HostInfo *hosts);
  virtual ~Worker();
  DISALLOW_COPY(Worker);
  void Init();
  void WaitUntilEvent(Event event);
  void Stop();

 private:
  void InitListener();
  void SetUpLocalThreads();
  void HandleConnection(PollConn* poll_conn_ptr);
  conn::RecvBuffer& HandleReadEvent(PollConn *poll_conn_ptr);
};

Worker::Worker(
      size_t comm_buff_capacity,
      uint64_t listen_ip,
      int32_t listen_port,
      int32_t executor_id,
      size_t num_local_executors,
      size_t num_total_executors,
      const HostInfo* hosts):
    kCommBuffCapacity_(comm_buff_capacity),
    send_mem_(std::make_unique<uint8_t[]>(comm_buff_capacity)),
    send_buff_(send_mem_.get(), comm_buff_capacity),
    kListenIp_(listen_ip),
    kListenPort_(listen_port),
    kMyId_(executor_id),
    kNumLocalExecutors_(num_local_executors),
    kNumExecutors_(num_total_executors),
    kClients_(std::make_unique<conn::Socket[]>(kNumExecutors_)),
    kHosts_(std::make_unique<HostInfo[]>(kNumExecutors_))
{
  memcpy(kHosts_.get(), hosts, sizeof(HostInfo)*num_total_executors);
}

Worker::~Worker() { }

void
Worker::Init() {
  static constexpr size_t kNumEvents = 100;
  InitListener();

  int ret = -1;
  ret = poll_.Init();
  CHECK(ret == 0) << "poll init failed";
  PollConn listen_poll_conn = {&listen_, PollConn::ConnType::listen};
  poll_.Add(listen_.get_fd(), &listen_poll_conn);

  PollConn driver_poll_conn = {&driver_, PollConn::ConnType::driver};
  poll_.Add(driver_->pipe.get_read_fd(), &driver_poll_conn);
  epoll_event es[kNumEvents];

  SetUpLocalThreads();
}

void
Worker::WaitUntilEvent(Event event) {
  do {
    switch(event) {
      case Event::kExecutorConnectToPeers:
      if (num_connected_clients_ == kNumExecutors_ - 1
          && num_peer_connected_threads_ == 2) {
        return true;
      }
      default:
        LOG(FATAL) << "unknown event " << static_cast<int>(event);
    }

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
  }
}

//------------ Private Functions -----------------------

void
Worker::InitListener () {
  listen_.Bind(kListenIp_, (uint16_t) kListenPort_);
  listen_.Listen(64);
}

void
Worker::HandleConnection(
    PollConn* poll_conn_ptr) {
  listen_.Accept(&(kClients_[num_connected_clients_]));
  num_connected_clients_++;
  if (num_connected_clients_ == kNumExecutors_ - 1) {
    server_ = std::make_unique<Server>(
        kClients_.get(), kNumExecutors_, pipe_mse_[1],
        pipe_scl_[0],
        kCommBuffCapacity_,
        runtime_.get(),
        kMyId_);
    server_thread_ = std::make_unique<std::thread>(
        &Server::operator(),
        server_.get());
  }

  if (num_connected_clients_ == kNumExecutors_ - 1
      && num_peer_connected_threads_ == 2) {
    LOG(INFO) << kMyId_ << " Worker ConnectToPeers done";
    message::Helper::CreateMsg<
      message::ExecutorConnectToPeersAck,
      message::DefaultPayloadCreator<message::Type::kExecutorConnectToPeersAck>
      >(&send_buff_);
    driver_->pipe.Send(&send_buff_);
    state_ = State::kRunning;
  }
}


void
Worker::SetUpLocalThreads() {
  LOG(INFO) << kMyId_ << " Worker " << __func__;
  runtime_ = std::make_unique<WorkerRuntime>(kNumExecutors_,
                                        kNumLocalExecutors_,
                                        kMyId_);
  int ret = conn::Pipe::CreateBiPipe(pipe_mse_);
  CHECK(ret == 0);
  ret = conn::Pipe::CreateBiPipe(pipe_mcl_);
  CHECK(ret == 0);
  ret = conn::Pipe::CreateBiPipe(pipe_scl_);
  CHECK(ret == 0);

  pipe_recv_mem_ = std::make_unique<uint8_t[]>(kCommBuffCapacity_ * 3);
  server_pipe_conn_ = std::make_unique<conn::PipeConn>(pipe_mse_[0], pipe_recv_mem_.get(),
                                            kCommBuffCapacity_);
  client_pipe_conn_ = std::make_unique<conn::PipeConn>(pipe_mcl_[0],
                                            pipe_recv_mem_.get() + kCommBuffCapacity_,
                                            kCommBuffCapacity_);

  server_poll_conn_ = {server_pipe_conn_.get(), PollConn::ConnType::pipe};
  client_poll_conn_ = {client_pipe_conn_.get(), PollConn::ConnType::pipe};

  poll_.Add(pipe_mse_[0].get_read_fd(), &server_poll_conn_);
  poll_.Add(pipe_mcl_[0].get_read_fd(), &client_poll_conn_);

  client_ = std::make_unique<Client>(
      kHosts_.get(), kNumExecutors_, pipe_mcl_[1],
      pipe_scl_[1],
      kCommBuffCapacity_,
      runtime_.get(),
      kMyId_);

  client_thread_ = std::make_unique<std::thread>(
      &Client::operator(),
      client_.get());
}

bool
Worker::HandleDriverMsg() {
  auto &pipe = driver_->pipe;
  auto &recv_buff = driver_->recv_buff;

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
    case message::Type::kDriverMsg:
      {
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
        LOG(INFO) << kMyId_ << " Worker handling Driver message ExecutorStop";
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

conn::RecvBuffer &
Worker::HandleReadEvent(PollConn *poll_conn_ptr) {
  CHECK(poll_conn_ptr->type == PollConn::ConnType::pipe);
  auto &pipe = reinterpret_cast<conn::PipeConn*>(
      poll_conn_ptr->conn)->pipe;
  auto &recv_buff = reinterpret_cast<conn::PipeConn*>(
      poll_conn_ptr->conn)->recv_buff;

  if (!recv_buff.ReceivedFullMsg()) {
    bool recv = pipe.Recv(&recv_buff);
    if (!recv) return recv_buff;
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
      }
      break;
    default:
      LOG(FATAL) << "unknown message!";
      break;
  }

  recv_buff.ClearOneMsg();
  return recv_buff;
}

void
Worker::Stop() {
  message::Helper::CreateMsg<
    message::ExecutorStop,
    message::DefaultPayloadCreator<message::Type::kExecutorStop>
   >(&send_buff_);
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

  pipe_driver_[0].Close();
  pipe_driver_[1].Close();
  listen_.Close();

  for (size_t i = 0; i < kNumExecutors_; i++) {
    if (kClients_[i].get_fd() != 0) {
      kClients_[i].Close();
    }
  }
  LOG(INFO) << kMyId_ << " master exiting";
}

}
}
