#pragma once

#include <memory>
#include <vector>

#include <orion/noncopyable.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/message.hpp>
#include <orion/bosen/execute_message.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/event_handler.hpp>
#include <orion/bosen/byte_buffer.hpp>
#include <orion/bosen/peer_recv_buffer.hpp>
#include <orion/bosen/recv_arbitrary_bytes.hpp>

namespace orion {
namespace bosen {

class PeerRecvThread {
  struct PollConn {
    enum class ConnType {
      peer = 1,
        executor = 2
    };
    void* conn;
    ConnType type;
    int32_t id;

    bool Receive() {
      if (type == ConnType::peer) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.Recv(&(sock_conn->recv_buff));
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.Recv(&(pipe_conn->recv_buff));
      }
    }

    bool Send() {
      if (type == ConnType::peer) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.Send(&(sock_conn->send_buff));
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.Send(&(pipe_conn->send_buff));
      }
    }

    conn::RecvBuffer& get_recv_buff() {
      if (type == ConnType::peer) {
        return reinterpret_cast<conn::SocketConn*>(conn)->recv_buff;
      } else {
        return reinterpret_cast<conn::PipeConn*>(conn)->recv_buff;
      }
    }

    conn::SendBuffer& get_send_buff() {
      if (type == ConnType::peer) {
        return reinterpret_cast<conn::SocketConn*>(conn)->send_buff;
      } else {
        return reinterpret_cast<conn::PipeConn*>(conn)->send_buff;
      }
    }

    bool is_connect_event() const {
      return false;
    }

    int get_read_fd() const {
      if (type == ConnType::peer) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.get_fd();
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.get_read_fd();
      }
    }

    int get_write_fd() const {
      if (type == ConnType::peer) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.get_fd();
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.get_write_fd();
      }
    }
  };

  enum class Action {
    kNone = 0,
      kExit = 1,
      kAckConnectToPeers = 2
            };

  const int32_t kId;
  const size_t kNumExecutors;
  const size_t kCommBuffCapacity;

  EventHandler<PollConn> event_handler_;
  Blob send_mem_;
  conn::SendBuffer send_buff_;

  Blob peer_send_mem_;
  Blob peer_recv_mem_;
  std::vector<std::unique_ptr<conn::SocketConn>> peer_;
  std::vector<PollConn> peer_conn_;
  std::vector<conn::Socket> peer_socks_;
  std::vector<int> peer_sock_fds_;
  std::vector<ByteBuffer> peer_recv_byte_buff_;

  conn::Pipe executor_pipe_[2];
  Blob executor_recv_mem_;
  Blob executor_send_mem_;
  std::unique_ptr<conn::PipeConn> executor_;
  PollConn executor_conn_;
  size_t num_identified_peers_ {0};
  Action action_ { Action::kNone };

  void *data_recv_buff_ { nullptr };

 public:
  PeerRecvThread(int32_t id,
                 const std::vector<conn::Socket> &peer_socks,
                 size_t buff_capacity);
  void operator() ();
  conn::Pipe GetExecutorPipe();
 private:
  int HandleMsg(PollConn* poll_conn_ptr);
  int HandlePeerMsg(PollConn* poll_conn_ptr);
  int HandleExecuteMsg(PollConn* poll_conn_ptr);
  int HandleExecutorMsg();
  int HandleClosedConnection(PollConn *poll_conn_ptr);

  void Send(PollConn* poll_conn_ptr, conn::PipeConn* pipe_conn);
};

PeerRecvThread::PeerRecvThread(
    int32_t id,
    const std::vector<conn::Socket> &peer_socks,
    size_t buff_capacity):
    kId(id),
    kNumExecutors(peer_socks.size()),
    kCommBuffCapacity(buff_capacity),
    send_mem_(kCommBuffCapacity),
    send_buff_(send_mem_.data(), kCommBuffCapacity),
    peer_send_mem_(buff_capacity * kNumExecutors),
    peer_recv_mem_(buff_capacity * kNumExecutors),
    peer_(peer_socks.size()),
    peer_conn_(peer_socks.size()),
    peer_socks_(peer_socks),
    peer_sock_fds_(peer_socks.size()),
    peer_recv_byte_buff_(peer_socks.size()),
    executor_recv_mem_(buff_capacity),
    executor_send_mem_(buff_capacity),
    num_identified_peers_(0) {
  int ret = conn::Pipe::CreateBiPipe(executor_pipe_);
  CHECK_EQ(ret, 0) << "create pipe failed";

  executor_ = std::make_unique<conn::PipeConn>(
      executor_pipe_[0],
      executor_recv_mem_.data(),
      executor_send_mem_.data(),
      kCommBuffCapacity);
  executor_conn_.type = PollConn::ConnType::executor;
  executor_conn_.conn = executor_.get();
}

void
PeerRecvThread::operator() () {

  event_handler_.SetClosedConnectionHandler(
      std::bind(&PeerRecvThread::HandleClosedConnection, this,
                std::placeholders::_1));

  event_handler_.SetReadEventHandler(
      std::bind(&PeerRecvThread::HandleMsg, this, std::placeholders::_1));

  event_handler_.SetDefaultWriteEventHandler();

  event_handler_.SetToReadOnly(&executor_conn_);

  for (size_t num_peers = 0; num_peers < kNumExecutors; num_peers++) {
    if (num_peers == kId) continue;
    auto &sock = peer_socks_[num_peers];
    uint8_t *recv_mem = peer_recv_mem_.data()
                        + kCommBuffCapacity * num_peers;

    uint8_t *send_mem = peer_send_mem_.data()
                        + kCommBuffCapacity * num_peers;

    auto *sock_conn = new conn::SocketConn(
        sock, recv_mem, send_mem, kCommBuffCapacity);
    auto &curr_poll_conn = peer_conn_[num_peers];
    curr_poll_conn.conn = sock_conn;
    curr_poll_conn.type = PollConn::ConnType::peer;
    curr_poll_conn.id = num_peers;
    int ret = event_handler_.SetToReadOnly(&curr_poll_conn);
    CHECK_EQ(ret, 0) << "errno = " << errno << " fd = " << sock.get_fd()
                     << " i = " << num_peers
                     << " id = " << kId;
  }

  while (true) {
    event_handler_.WaitAndHandleEvent();
    if (action_ == Action::kExit) break;
  }
}

conn::Pipe
PeerRecvThread::GetExecutorPipe() {
  return executor_pipe_[1];
}

int
PeerRecvThread::HandleMsg(PollConn* poll_conn_ptr) {
  int ret = 0;
  if (poll_conn_ptr->type == PollConn::ConnType::peer) {
    ret = HandlePeerMsg(poll_conn_ptr);
  } else {
    ret = HandleExecutorMsg();
  }

  while (action_ != Action::kNone
         && action_ != Action::kExit) {
    switch (action_) {
      case Action::kExit:
        break;
      case Action::kAckConnectToPeers:
        {
          message::Helper::CreateMsg<message::ExecutorConnectToPeersAck>(&send_buff_);
          send_buff_.set_next_to_send(
              peer_sock_fds_.data(), peer_sock_fds_.size() * sizeof(int));
          Send(&executor_conn_, executor_.get());
          send_buff_.clear_to_send();
          action_ = Action::kNone;
        }
        break;
      default:
        LOG(FATAL) << "unknown";
    }
  }
  return  ret;
}

int
PeerRecvThread::HandleExecutorMsg() {
  auto &recv_buff = executor_->recv_buff;

  auto msg_type = message::Helper::get_type(recv_buff);
  CHECK(msg_type == message::Type::kExecuteMsg);
  auto exec_msg_type = message::ExecuteMsgHelper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kNoAction;
  switch (exec_msg_type) {
    case message::ExecuteMsgType::kPeerRecvStop:
      {
        action_ = Action::kExit;
        ret = EventHandler<PollConn>::kClearOneMsg | EventHandler<PollConn>::kExit;
      }
      break;
    default:
      LOG(FATAL) << "unknown exec msg " << static_cast<int>(exec_msg_type);
  }
  return ret;
}

int
PeerRecvThread::HandlePeerMsg(PollConn* poll_conn_ptr) {
  auto &recv_buff = poll_conn_ptr->get_recv_buff();

  auto msg_type = message::Helper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kClearOneMsg;
  switch (msg_type) {
    case message::Type::kExecutorIdentity:
      {
        auto *msg = message::Helper::get_msg<message::ExecutorIdentity>(recv_buff);
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(poll_conn_ptr->conn);
        //LOG(INFO) << "id = " << kId << " executor_id = " << msg->executor_id
        //          << " sock_fd = " << sock_conn->sock.get_fd();
        peer_[msg->executor_id].reset(sock_conn);
        peer_sock_fds_[msg->executor_id] = sock_conn->sock.get_fd();
        poll_conn_ptr->id = msg->executor_id;
        num_identified_peers_++;
        if (num_identified_peers_ == kId) {
          action_ = Action::kAckConnectToPeers;
        }
        ret = EventHandler<PollConn>::kClearOneMsg;
      }
      break;
    case message::Type::kExecuteMsg:
      {
        ret = HandleExecuteMsg(poll_conn_ptr);
      }
      break;
    default:
      {
        LOG(FATAL) << "unknown message type " << static_cast<int>(msg_type)
                   << " from " << poll_conn_ptr->id;
      }
      break;
  }
  return ret;
}

int
PeerRecvThread::HandleExecuteMsg(PollConn* poll_conn_ptr) {
  auto &recv_buff = poll_conn_ptr->get_recv_buff();
  auto &sock = *reinterpret_cast<conn::Socket*>(poll_conn_ptr->conn);
  auto msg_type = message::ExecuteMsgHelper::get_type(recv_buff);
  LOG(INFO) << __func__ << " msg_type = " << static_cast<int>(msg_type)
            << " from " << poll_conn_ptr->id;
  int ret = EventHandler<PollConn>::kClearOneMsg;
  int32_t sender_id = poll_conn_ptr->id;
  switch (msg_type) {
    case message::ExecuteMsgType::kRepartitionDistArrayData:
      {
        LOG(INFO) << "received kRepartitionDistArrayData";
        auto *msg = message::ExecuteMsgHelper::get_msg<message::ExecuteMsgRepartitionDistArrayData>(
            recv_buff);
        //while(1);
        size_t expected_size = msg->data_size;
        bool received_next_msg = (expected_size == 0);
        if (data_recv_buff_ == nullptr) {
          auto *buff_ptr = new PeerRecvRepartitionDistArrayDataBuffer();
          LOG(INFO) << "received id = " << msg->dist_array_id;
          buff_ptr->dist_array_id = msg->dist_array_id;
          data_recv_buff_ = buff_ptr;
        }
        auto *repartition_recv_buff =
            reinterpret_cast<PeerRecvRepartitionDistArrayDataBuffer*>(
                data_recv_buff_);
        if (expected_size > 0) {
          auto &byte_buffs = repartition_recv_buff->byte_buffs;
          auto &byte_buff = byte_buffs[sender_id];
          if (byte_buff.GetCapacity() == 0) byte_buff.Reset(expected_size);
          received_next_msg =
              ReceiveArbitraryBytes(
                  sock, &recv_buff,
                  &byte_buff, expected_size);
        }

        if (received_next_msg) {
          ret = expected_size > 0
                ? EventHandler<PollConn>::kClearOneAndNextMsg
                : EventHandler<PollConn>::kClearOneMsg;
          repartition_recv_buff->num_executors_received += 1;
          if (repartition_recv_buff->num_executors_received
              == (kNumExecutors - 1)) {
            message::ExecuteMsgHelper::CreateMsg<
              message::ExecuteMsgRepartitionDistArrayRecved>(
                &send_buff_, data_recv_buff_);
            data_recv_buff_ = nullptr;
            Send(&executor_conn_, executor_.get());
            send_buff_.clear_to_send();
          }
        } else {
          ret = EventHandler<PollConn>::kNoAction;
        }
        action_ = Action::kNone;
      }
      break;
    default:
      LOG(FATAL) << "unexpected message type = " << static_cast<int>(msg_type);
  }
  return ret;
}

int
PeerRecvThread::HandleClosedConnection(PollConn *poll_conn_ptr) {
  int ret = event_handler_.Remove(poll_conn_ptr);
  CHECK_EQ(ret, 0);

  return EventHandler<PollConn>::kNoAction;
}

void
PeerRecvThread::Send(PollConn* poll_conn_ptr, conn::PipeConn* pipe_conn) {
  auto& send_buff = poll_conn_ptr->get_send_buff();
  if (send_buff.get_remaining_to_send_size() > 0
      || send_buff.get_remaining_next_to_send_size() > 0) {
    bool sent = pipe_conn->pipe.Send(&send_buff);
    while (!sent) {
      sent = pipe_conn->pipe.Send(&send_buff);
    }
    send_buff.clear_to_send();
  }
  bool sent = pipe_conn->pipe.Send(&send_buff_);
  if (!sent) {
    send_buff.Copy(send_buff_);
    event_handler_.SetToReadWrite(poll_conn_ptr);
  }
  send_buff_.reset_sent_sizes();
}

}
}
