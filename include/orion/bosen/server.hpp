#pragma once

#include <orion/bosen/message.hpp>
#include <orion/constants.hpp>
#include <climits>
#include <stdint.h>
#include <orion/noncopyable.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <orion/helper.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/worker_runtime.hpp>
namespace orion {
namespace bosen {

class Server {
 private:
  struct PollConn {
    enum class ConnType {
      master = 0,
        client = 1,
        local_client = 2
    };

    void *conn {nullptr};
    ConnType type;
  };
  const size_t kCommBuffCapacity_;
  conn::Poll poll_;
  std::unique_ptr<std::unique_ptr<conn::SocketConn>[]> client_conn_;
  const size_t kNumClients_;
  size_t num_identified_clients_ {0};
  std::unique_ptr<uint8_t[]> send_mem_;
  conn::SendBuffer send_buff_;
  std::unique_ptr<uint8_t[]> recv_mem_;
  conn::PipeConn master_;
  conn::PipeConn local_client_;
  std::unique_ptr<PollConn[]> poll_conns_;
  bool stop_ {false};
  WorkerRuntime * const runtime_;
  const int32_t kMyId_;

 public:
  Server(conn::Socket* clients, size_t num_clients,
         conn::Pipe master,
         conn::Pipe local_client,
         size_t comm_buff_capacity,
         WorkerRuntime *runtime,
         int32_t my_id);
  void operator() ();
  DISALLOW_COPY(Server);
 private:
  bool HandleMasterMsg();
  bool HandleClientMsg(PollConn *poll_conn_ptr);
  void HandleClosedConnection(PollConn *poll_conn_ptr);
  void BroadcastPeers();
  conn::RecvBuffer &HandleReadEvent(PollConn *poll_conn_ptr);
};

Server::Server(
    conn::Socket* clients, size_t num_clients,
    conn::Pipe master,
    conn::Pipe local_client,
    size_t comm_buff_capacity,
    WorkerRuntime *runtime,
    int32_t my_id):
    kCommBuffCapacity_(comm_buff_capacity),
    client_conn_(std::make_unique<std::unique_ptr<conn::SocketConn>[]>(num_clients)),
    kNumClients_(num_clients),
    send_mem_(std::make_unique<uint8_t[]>(comm_buff_capacity)),
    send_buff_(send_mem_.get(), comm_buff_capacity),
    recv_mem_(std::make_unique<uint8_t[]>(comm_buff_capacity * (num_clients + 2))),
    master_(master, recv_mem_.get(), comm_buff_capacity),
  local_client_(local_client, recv_mem_.get() + comm_buff_capacity, comm_buff_capacity),
  poll_conns_(std::make_unique<PollConn[]>(num_clients + 2)),
  runtime_(runtime),
  kMyId_(my_id) {
  poll_conns_[0].conn = &master_;
  poll_conns_[0].type = PollConn::ConnType::master;
  poll_conns_[1].conn = &local_client_;
  poll_conns_[1].type = PollConn::ConnType::local_client;

  int ret = poll_.Init();
  CHECK(ret == 0) << "poll init failed";
  poll_.Add(master_.pipe.get_read_fd(), &poll_conns_[0]);
  poll_.Add(local_client_.pipe.get_read_fd(), &poll_conns_[1]);

  for (size_t i = 0; i < num_clients; ++i) {
    if (clients[i].get_fd() == 0) continue;
    auto client_conn_ptr = new conn::SocketConn(
        clients[i], recv_mem_.get() + comm_buff_capacity * (i + 2),
        comm_buff_capacity);
    poll_conns_[i + 2].conn = client_conn_ptr;
    poll_conns_[i + 2].type = PollConn::ConnType::client;
    poll_.Add(clients[i].get_fd(), &poll_conns_[i + 2]);
  }
}

bool
Server::HandleMasterMsg() {
  auto &pipe = master_.pipe;
  auto &recv_buff = master_.recv_buff;

  if (!recv_buff.ReceivedFullMsg()) {
    bool recv = pipe.Recv(&recv_buff);
    if (!recv) return false;
  }

  CHECK (!recv_buff.is_error()) << "server error during receiving " << errno;
  CHECK (!recv_buff.EOFAtIncompleteMsg()) << "server error : early EOF";
  // maybe EOF but not received anything
  if (!recv_buff.ReceivedFullMsg()) return false;

  auto msg_type = message::Helper::get_type(recv_buff);
  switch(msg_type) {
    case message::Type::kExecutorStop:
      {
        stop_ = true;
      }
      break;
    default:
      LOG(FATAL) << "unknown message";
      break;
  }
  return true;
}

bool
Server::HandleClientMsg(PollConn *poll_conn_ptr) {
  auto &sock_conn = *reinterpret_cast<conn::SocketConn*>(
      poll_conn_ptr->conn);
  auto &sock = sock_conn.sock;
  auto &recv_buff = sock_conn.recv_buff;

  if (!recv_buff.ReceivedFullMsg()) {
    bool recv = sock.Recv(&recv_buff);
    if (!recv) return false;
  }

  CHECK (!recv_buff.is_error()) << "server error during receiving " << errno
                                << " fd = " << sock.get_fd();
  CHECK (!recv_buff.EOFAtIncompleteMsg()) << "server error : early EOF";
  // maybe EOF but not received anything
  if (!recv_buff.ReceivedFullMsg()) return false;

  auto msg_type = message::Helper::get_type(recv_buff);
  switch(msg_type) {
    case message::Type::kExecutorIdentity:
      {
        auto *msg = message::Helper::get_msg<message::ExecutorIdentity>(recv_buff);
        client_conn_[msg->executor_id].reset(&sock_conn);
        num_identified_clients_++;
        if (num_identified_clients_ == kNumClients_ - 1) {
          message::Helper::CreateMsg<
            message::ExecutorConnectToPeersAck,
            message::DefaultPayloadCreator<message::Type::kExecutorConnectToPeersAck>
            >(
              &send_buff_);
          master_.pipe.Send(&send_buff_);
        }
      }
      break;
    default:
      LOG(FATAL) << "unknown message ";
      break;
  }

  return true;
}

void
Server::HandleClosedConnection(PollConn *poll_conn_ptr) {
  switch(poll_conn_ptr->type) {
    case PollConn::ConnType::master:
      {
        auto &pipe_conn = *reinterpret_cast<conn::PipeConn*>(
            poll_conn_ptr->conn);
        auto &pipe = pipe_conn.pipe;
        poll_.Remove(pipe.get_read_fd());
      }
      break;
    case PollConn::ConnType::client:
      {
        auto &sock_conn = *reinterpret_cast<conn::SocketConn*>(
            poll_conn_ptr->conn);
        auto &sock = sock_conn.sock;
        poll_.Remove(sock.get_fd());
      }
      break;
    default:
      LOG(FATAL);
  }
  return;
}

conn::RecvBuffer &
Server::HandleReadEvent(PollConn *poll_conn_ptr) {
  bool next_message = false;
  conn::RecvBuffer *recv_buff_ptr = nullptr;
  switch(poll_conn_ptr->type) {
    case PollConn::ConnType::master:
      {
        next_message = HandleMasterMsg();
        recv_buff_ptr = &master_.recv_buff;
      }
      break;
    case PollConn::ConnType::client:
      {
        next_message = HandleClientMsg(poll_conn_ptr);
        auto &sock_conn = *reinterpret_cast<conn::SocketConn*>(
            poll_conn_ptr->conn);
        recv_buff_ptr = &(sock_conn.recv_buff);
      }
      break;
    default:
      LOG(FATAL) << "what is this?";
      break;
  }
  if (next_message) {
    recv_buff_ptr->ClearOneMsg();
  }
  return *recv_buff_ptr;
}

void
Server::operator() () {
  static constexpr size_t kNumEvents = 100;
  epoll_event es[kNumEvents];
  while (1) {
    int num_events = poll_.Wait(es, kNumEvents);
    CHECK(num_events > 0);
    for (int i = 0; i < num_events; ++i) {
      //LOG(INFO) << "server polling";
      PollConn *poll_conn_ptr = conn::Poll::EventConn<PollConn>(es, i);
      //LOG(INFO) << "server got a message";
      if (es[i].events & EPOLLIN) {
        auto &recv_buff = HandleReadEvent(poll_conn_ptr);
        while (recv_buff.ReceivedFullMsg()) {
          HandleReadEvent(poll_conn_ptr);
        }
        if (recv_buff.is_eof()) {
          //LOG(INFO) << "someone has closed";
          HandleClosedConnection(poll_conn_ptr);
        }
      }
    }
    if (stop_) break;
  }
}

void
Server::BroadcastPeers() {
  for (int i = 0; i < kNumClients_; ++i) {
    if (kMyId_ == i) continue;
    size_t nsent = client_conn_[i]->sock.Send(&send_buff_);
    CHECK(conn::CheckSendSize(send_buff_, nsent));
  }
}
}
}
