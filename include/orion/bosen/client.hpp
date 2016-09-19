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

class Client {
 private:
  struct PollConn {
    enum class ConnType {
      master = 0,
        server = 1,
        local_server = 2
    };

    void *conn;
    ConnType type;
  };
  std::unique_ptr<conn::Socket[]> servers_;
  const HostInfo* hosts_;
  const size_t comm_buff_capacity_;
  conn::Poll poll_;
  std::unique_ptr<std::unique_ptr<conn::SocketConn>[]> server_conn_;
  const size_t num_servers_;
  std::unique_ptr<uint8_t[]> send_mem_;
  conn::SendBuffer send_buff_;
  std::unique_ptr<uint8_t[]> recv_mem_;
  conn::PipeConn master_;
  conn::PipeConn local_server_;
  std::unique_ptr<PollConn[]> poll_conns_;
  bool stop_ {false};
  WorkerRuntime * const runtime_;
  const int32_t my_id_;
  size_t num_peer_done_recved_ {0};
  bool partition_data_ { false };

 public:
  Client(const HostInfo* hosts,
         size_t num_servers,
         conn::Pipe master,
         conn::Pipe local_server,
         size_t comm_buff_capacity,
         WorkerRuntime *runtime,
         int32_t my_id);
  void operator() ();
  DISALLOW_COPY(Client);

 private:
  bool HandleMasterMsg();
  bool HandleLocalServerMsg();
  bool HandleServerMsg(PollConn *poll_conn_ptr);
  bool HandleServerDataMsg(PollConn *poll_conn_ptr);
  void HandleClosedConnection(PollConn *poll_conn_ptr);
  conn::RecvBuffer &HandleReadEvent(PollConn *poll_conn_ptr);
};

Client::Client(const HostInfo* hosts,
    size_t num_servers,
    conn::Pipe master,
    conn::Pipe local_server,
    size_t comm_buff_capacity,
    WorkerRuntime *runtime,
    int32_t my_id):
    servers_(std::make_unique<conn::Socket[]>(num_servers)),
    hosts_(hosts),
    comm_buff_capacity_(comm_buff_capacity),
    server_conn_(std::make_unique<std::unique_ptr<conn::SocketConn>[]>(num_servers)),
    num_servers_(num_servers),
    send_mem_(std::make_unique<uint8_t[]>(comm_buff_capacity)),
    send_buff_(send_mem_.get(), comm_buff_capacity),
    recv_mem_(std::make_unique<uint8_t[]>(comm_buff_capacity * (num_servers + 2))),
    master_(master, recv_mem_.get(), comm_buff_capacity),
  local_server_(local_server, recv_mem_.get() + comm_buff_capacity, comm_buff_capacity),
  poll_conns_(std::make_unique<PollConn[]>(num_servers + 2)),
  runtime_(runtime),
  my_id_(my_id) {
  poll_conns_[0].conn = &master_;
  poll_conns_[0].type = PollConn::ConnType::master;
  poll_conns_[1].conn = &local_server_;
  poll_conns_[1].type = PollConn::ConnType::local_server;

  int ret = poll_.Init();
  CHECK(ret == 0) << "poll init failed";
  poll_.Add(master_.pipe.get_read_fd(), &poll_conns_[0]);
  poll_.Add(local_server_.pipe.get_read_fd(), &poll_conns_[1]);
}

bool
Client::HandleMasterMsg() {
  auto &pipe = master_.pipe;
  auto &recv_buff = master_.recv_buff;

  if (!recv_buff.ReceivedFullMsg()) {
    bool recv = pipe.Recv(&recv_buff);
    if (!recv) return false;
  }

  CHECK (!recv_buff.is_error()) << "client error during receiving " << errno;
  CHECK (!recv_buff.EOFAtIncompleteMsg()) << "client error : early EOF";
  // maybe EOF but not received anything
  if (!recv_buff.ReceivedFullMsg()) return false;

  auto msg_type = message::Helper::get_type(recv_buff);
  switch(msg_type) {
    case message::Type::kExecutorConnectToPeers:
      {
        LOG(INFO) << my_id_ << " Client ConnectToPeers";
        message::Helper::CreateMsg<
          message::ExecutorIdentity>(&send_buff_, my_id_);
        for (size_t i = 0; i < num_servers_; ++i) {
          if (i == my_id_) continue;
          int ret = servers_[i].Connect(hosts_[i].ip, hosts_[i].port);
          CHECK(ret == 0) << "executor " << my_id_ << " connected to " << i
                          << " ret = " << ret
                          << " ip = " << hosts_[i].ip << " port = " << hosts_[i].port;
          servers_[i].Send(&send_buff_);
          server_conn_[i] = std::make_unique<conn::SocketConn>(
              servers_[i], recv_mem_.get() + comm_buff_capacity_ * (i + 2),
              comm_buff_capacity_);
          poll_conns_[i + 2].conn = server_conn_[i].get();
          poll_conns_[i + 2].type = PollConn::ConnType::server;
          if (servers_[i].get_fd() == 0) continue;
          poll_.Add(servers_[i].get_fd(), &poll_conns_[i + 2]);
        }
        message::Helper::CreateMsg<message::ExecutorConnectToPeersAck>(
            &send_buff_);
        master_.pipe.Send(&send_buff_);
      }
      break;
    case message::Type::kExecutorStop:
      {
        stop_ = true;
        for (size_t i = 0; i < num_servers_; i++) {
          if (servers_[i].get_fd() != 0) {
            servers_[i].Close();
            //LOG(INFO) << "close server " << servers_[i].get_fd();
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "unknown message!";
      break;
  }
  return true;
}

bool
Client::HandleLocalServerMsg() {
  auto &pipe = local_server_.pipe;
  auto &recv_buff = local_server_.recv_buff;

  if (!recv_buff.ReceivedFullMsg()) {
    bool recv = pipe.Recv(&recv_buff);
    if (!recv) return false;
  }

  CHECK (!recv_buff.is_error()) << "client error during receiving " << errno;
  CHECK (!recv_buff.EOFAtIncompleteMsg()) << "client error : early EOF";
  // maybe EOF but not received anything
  if (!recv_buff.ReceivedFullMsg()) return false;

  auto msg_type = message::Helper::get_type(recv_buff);
  switch(msg_type) {
    default:
      LOG(FATAL) << "unknown message!";
      break;
  }
  return true;
}

bool
Client::HandleServerMsg(PollConn *poll_conn_ptr) {
  auto &sock_conn = *reinterpret_cast<conn::SocketConn*>(
      poll_conn_ptr->conn);
  auto &sock = sock_conn.sock;
  auto &recv_buff = sock_conn.recv_buff;

  if (!recv_buff.ReceivedFullMsg()) {
    bool recv = sock.Recv(&recv_buff);
    if (!recv) return false;
  } else if (recv_buff.IsExepectingNextMsg()) {
    return HandleServerDataMsg(poll_conn_ptr);
  }

  CHECK (!recv_buff.is_error()) << "client error during receiving " << errno
                                << " sock fd = " << sock.get_fd();
  CHECK (!recv_buff.EOFAtIncompleteMsg()) << "server error : early EOF";
  // maybe EOF but not received anything
  if (!recv_buff.ReceivedFullMsg()) return false;

  auto msg_type = message::Helper::get_type(recv_buff);
  switch(msg_type) {
    default:
      LOG(FATAL) << "unknown message!";
  }
  return true;
}

bool
Client::HandleServerDataMsg(PollConn *poll_conn_ptr) {
  auto &sock_conn = *reinterpret_cast<conn::SocketConn*>(
      poll_conn_ptr->conn);
  auto &recv_buff = sock_conn.recv_buff;
  auto msg_type = message::Helper::get_type(recv_buff);
  switch(msg_type) {
    default:
      LOG(FATAL) << "unkonwn message type " << static_cast<int>(msg_type);
  }
  return false;
}

void
Client::HandleClosedConnection(PollConn *poll_conn_ptr) {
  switch(poll_conn_ptr->type) {
    case PollConn::ConnType::master:
      {
        auto &pipe_conn = *reinterpret_cast<conn::PipeConn*>(
            poll_conn_ptr->conn);
        auto &pipe = pipe_conn.pipe;
        poll_.Remove(pipe.get_read_fd());
      }
      break;
    case PollConn::ConnType::server:
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
Client::HandleReadEvent(PollConn *poll_conn_ptr) {
  bool next_message = false;
  conn::RecvBuffer *recv_buff_ptr = nullptr;
  switch(poll_conn_ptr->type) {
    case PollConn::ConnType::master:
      {
        next_message = HandleMasterMsg();
        recv_buff_ptr = &master_.recv_buff;
      }
      break;
    case PollConn::ConnType::server:
      {
        next_message = HandleServerMsg(poll_conn_ptr);
        auto &sock_conn = *reinterpret_cast<conn::SocketConn*>(
            poll_conn_ptr->conn);
        recv_buff_ptr = &(sock_conn.recv_buff);
      }
      break;
    case PollConn::ConnType::local_server:
      {
        next_message = HandleLocalServerMsg();
        recv_buff_ptr = &local_server_.recv_buff;
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
Client::operator() () {
  static constexpr size_t kNumEvents = 100;
  epoll_event es[kNumEvents];
  while (1) {
    //LOG(INFO) << "client polling";
    int num_events = poll_.Wait(es, kNumEvents);
    CHECK(num_events > 0);
    for (int i = 0; i < num_events; ++i) {
      PollConn *poll_conn_ptr = conn::Poll::EventConn<PollConn>(es, i);
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

}
}
