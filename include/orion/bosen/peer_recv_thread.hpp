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
      my_executor = 1,
        executor = 2,
        server = 3
    };
    void* conn;
    ConnType type;
    int32_t id;

    bool Receive() {
      if (type == ConnType::executor
          || type == ConnType::server) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.Recv(&(sock_conn->recv_buff));
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.Recv(&(pipe_conn->recv_buff));
      }
    }

    bool Send() {
      if (type == ConnType::executor
          || type == ConnType::server) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.Send(&(sock_conn->send_buff));
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.Send(&(pipe_conn->send_buff));
      }
    }

    conn::RecvBuffer& get_recv_buff() {
      if (type == ConnType::executor
          || type == ConnType::server) {
        return reinterpret_cast<conn::SocketConn*>(conn)->recv_buff;
      } else {
        return reinterpret_cast<conn::PipeConn*>(conn)->recv_buff;
      }
    }

    conn::SendBuffer& get_send_buff() {
      if (type == ConnType::executor
          || type == ConnType::server) {
        return reinterpret_cast<conn::SocketConn*>(conn)->send_buff;
      } else {
        return reinterpret_cast<conn::PipeConn*>(conn)->send_buff;
      }
    }

    bool is_connect_event() const {
      return false;
    }

    int get_read_fd() const {
      if (type == ConnType::executor
          || type == ConnType::server) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.get_fd();
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.get_read_fd();
      }
    }

    int get_write_fd() const {
      if (type == ConnType::executor
          || type == ConnType::server) {
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
  const int32_t kExecutorId;
  const int32_t kServerId;
  const bool kIsServer;
  const size_t kNumExecutors;
  const size_t kNumServers;
  const size_t kCommBuffCapacity;

  EventHandler<PollConn> event_handler_;
  Blob send_mem_;
  conn::SendBuffer send_buff_;

  Blob executor_server_send_mem_;
  Blob executor_server_recv_mem_;
  std::vector<conn::Socket> executor_server_socks_;
  std::vector<PollConn> executor_server_conn_;
  std::vector<int> executor_server_sock_fds_;

  std::vector<std::unique_ptr<conn::SocketConn>> peer_;
  std::vector<ByteBuffer> peer_recv_byte_buff_;

  std::vector<std::unique_ptr<conn::SocketConn>> server_;
  std::vector<ByteBuffer> server_recv_byte_buff_;

  conn::Pipe executor_pipe_[2];
  Blob executor_recv_mem_;
  Blob executor_send_mem_;
  std::unique_ptr<conn::PipeConn> executor_;
  PollConn executor_conn_;
  size_t num_identified_peers_ {0};
  Action action_ { Action::kNone };

  void *data_recv_buff_ { nullptr };
  bool has_executor_requested_pipeline_time_partitions_ { false };
  std::vector<PeerRecvPipelinedTimePartitionsBuffer*> pipelined_time_partitions_buff_vec_;
  PeerRecvPipelinedTimePartitionsBuffer* pipelined_time_partitions_buff_ { nullptr };

  std::vector<PeerRecvGlobalIndexedDistArrayDataBuffer*> global_indexed_dist_arrays_buff_vec_;
  bool has_executor_requested_global_indexed_dist_array_data_ { false };

  bool pred_completed_ { false };
  bool has_executor_requested_pred_completion_ { false };
 public:
  PeerRecvThread(int32_t id,
                 int32_t executor_id,
                 int32_t server_id,
                 bool is_server,
                 size_t num_executors,
                 size_t num_servers,
                 const std::vector<conn::Socket> &executor_server_socks,
                 size_t buff_capacity);
  void operator() ();
  conn::Pipe GetExecutorPipe();
 private:
  int HandleMsg(PollConn* poll_conn_ptr);
  int HandlePeerMsg(PollConn* poll_conn_ptr);
  int HandleExecuteMsg(PollConn* poll_conn_ptr);
  int HandleExecutorMsg();
  int HandleClosedConnection(PollConn *poll_conn_ptr);
  void ServePipelinedTimePartitionsRequest();
  void ServeGlobalIndexedDistArrayDataRequest();
  void ServeExecForLoopPredCompletion();
  void SendToExecutor();
};

PeerRecvThread::PeerRecvThread(
    int32_t id,
    int32_t executor_id,
    int32_t server_id,
    bool is_server,
    size_t num_executors,
    size_t num_servers,
    const std::vector<conn::Socket> &executor_server_socks,
    size_t buff_capacity):
    kId(id),
    kExecutorId(executor_id),
    kServerId(server_id),
    kIsServer(is_server),
    kNumExecutors(num_executors),
    kNumServers(num_servers),
    kCommBuffCapacity(buff_capacity),
    send_mem_(kCommBuffCapacity),
    send_buff_(send_mem_.data(), kCommBuffCapacity),
    executor_server_send_mem_(buff_capacity * (kNumExecutors + kNumServers)),
    executor_server_recv_mem_(buff_capacity * (kNumExecutors + kNumServers)),
    executor_server_socks_(executor_server_socks),
    executor_server_conn_(kNumExecutors + kNumServers),
    executor_server_sock_fds_(kNumExecutors + kNumServers),
    peer_(kNumExecutors),
    peer_recv_byte_buff_(kNumExecutors),
    server_(kNumServers),
    server_recv_byte_buff_(kNumServers),
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
  executor_conn_.type = PollConn::ConnType::my_executor;
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

  for (size_t peer_index = 0; peer_index < kNumExecutors + kNumServers; peer_index++) {
    if (peer_index == kId) continue;
    auto &sock = executor_server_socks_[peer_index];
    uint8_t *recv_mem = executor_server_recv_mem_.data()
                        + kCommBuffCapacity * peer_index;

    uint8_t *send_mem = executor_server_send_mem_.data()
                        + kCommBuffCapacity * peer_index;

    auto *sock_conn = new conn::SocketConn(
        sock, recv_mem, send_mem, kCommBuffCapacity);
    auto &curr_poll_conn = executor_server_conn_[peer_index];
    curr_poll_conn.conn = sock_conn;
    curr_poll_conn.type = PollConn::ConnType::executor;
    curr_poll_conn.id = peer_index >= kNumExecutors ? peer_index - kNumExecutors : peer_index;
    int ret = event_handler_.SetToReadOnly(&curr_poll_conn);
    CHECK_EQ(ret, 0) << "errno = " << errno << " fd = " << sock.get_fd()
                     << " i = " << peer_index
                     << " id = " << kId
                     << " kIsServer = " << kIsServer;
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
  if (poll_conn_ptr->type == PollConn::ConnType::executor
      || poll_conn_ptr->type == PollConn::ConnType::server) {
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
              executor_server_sock_fds_.data(),
              executor_server_sock_fds_.size() * sizeof(int));
          SendToExecutor();
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
  CHECK(msg_type == message::Type::kExecuteMsg)
      << " type = " << static_cast<int>(msg_type);
  auto exec_msg_type = message::ExecuteMsgHelper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kNoAction;
  switch (exec_msg_type) {
    case message::ExecuteMsgType::kPeerRecvStop:
      {
        action_ = Action::kExit;
        ret = EventHandler<PollConn>::kClearOneMsg | EventHandler<PollConn>::kExit;
      }
      break;
    case message::ExecuteMsgType::kRequestExecForLoopGlobalIndexedDistArrays:
      {
        has_executor_requested_global_indexed_dist_array_data_ = true;
        ServeGlobalIndexedDistArrayDataRequest();
        ret = EventHandler<PollConn>::kClearOneMsg;
      }
      break;
    case message::ExecuteMsgType::kRequestExecForLoopPipelinedTimePartitions:
      {
        has_executor_requested_pipeline_time_partitions_ = true;
        ServePipelinedTimePartitionsRequest();
        ret = EventHandler<PollConn>::kClearOneMsg;
      }
      break;
    case message::ExecuteMsgType::kRequestExecForLoopPredecessorCompletion:
      {
        has_executor_requested_pred_completion_ = true;
        ServeExecForLoopPredCompletion();
        ret = EventHandler<PollConn>::kClearOneMsg;
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
        peer_[msg->executor_id].reset(sock_conn);
        executor_server_sock_fds_[msg->executor_id] = sock_conn->sock.get_fd();
        poll_conn_ptr->id = msg->executor_id;
        poll_conn_ptr->type = PollConn::ConnType::executor;
        num_identified_peers_++;
        if (kIsServer) {
          if (num_identified_peers_ == (kNumExecutors + kServerId)) {
            action_ = Action::kAckConnectToPeers;
          } else {
            action_ = Action::kNone;
          }
        } else {
          if (num_identified_peers_ == kExecutorId) {
            action_ = Action::kAckConnectToPeers;
          } else {
            action_ = Action::kNone;
          }
        }
        ret = EventHandler<PollConn>::kClearOneMsg;
      }
      break;
    case message::Type::kServerIdentity:
      {
        auto *msg = message::Helper::get_msg<message::ServerIdentity>(recv_buff);
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(poll_conn_ptr->conn);
        server_[msg->server_id].reset(sock_conn);
        executor_server_sock_fds_[msg->server_id + kNumExecutors] = sock_conn->sock.get_fd();
        poll_conn_ptr->id = msg->server_id;
        poll_conn_ptr->type = PollConn::ConnType::server;
        num_identified_peers_++;
        if (kIsServer) {
          if (num_identified_peers_ == (kNumExecutors + kServerId)) {
            action_ = Action::kAckConnectToPeers;
          } else {
            action_ = Action::kNone;
          }
        } else {
          if (num_identified_peers_ == kExecutorId) {
            action_ = Action::kAckConnectToPeers;
          } else {
            action_ = Action::kNone;
          }
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
  int ret = EventHandler<PollConn>::kClearOneMsg;
  int32_t sender_id = poll_conn_ptr->id;
  switch (msg_type) {
    case message::ExecuteMsgType::kRepartitionDistArrayData:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgRepartitionDistArrayData>(recv_buff);
        size_t expected_size = msg->data_size;
        bool received_next_msg = false;
        if (data_recv_buff_ == nullptr) {
          auto *buff_ptr = new PeerRecvRepartitionDistArrayDataBuffer();
          buff_ptr->dist_array_id = msg->dist_array_id;
          data_recv_buff_ = buff_ptr;
        }
        auto *repartition_recv_buff =
            reinterpret_cast<PeerRecvRepartitionDistArrayDataBuffer*>(
                data_recv_buff_);
        if (expected_size > 0) {
          auto &byte_buffs = repartition_recv_buff->byte_buffs;
          auto &byte_buff = byte_buffs[sender_id];
          //if (byte_buff.GetCapacity() == 0) byte_buff.Reset(expected_size);
          received_next_msg =
              ReceiveArbitraryBytes(
                  sock, &recv_buff,
                  &byte_buff, expected_size);
        }

        bool from_server = msg->from_server;

        if (expected_size == 0  || received_next_msg) {
          ret = expected_size > 0
                ? EventHandler<PollConn>::kClearOneAndNextMsg
                : EventHandler<PollConn>::kClearOneMsg;
          repartition_recv_buff->num_msgs_received += 1;
          size_t num_expected_recvs = from_server ? (kNumServers - (kIsServer ? 1 : 0))
                                      : (kNumExecutors - (!kIsServer ? 1 : 0));

          if (repartition_recv_buff->num_msgs_received
              == num_expected_recvs) {
            message::ExecuteMsgHelper::CreateMsg<
              message::ExecuteMsgRepartitionDistArrayRecved>(
                &send_buff_, data_recv_buff_);
            data_recv_buff_ = nullptr;
            SendToExecutor();
            send_buff_.clear_to_send();
          }
        } else {
          ret = EventHandler<PollConn>::kNoAction;
        }
        action_ = Action::kNone;
      }
      break;
    case message::ExecuteMsgType::kPipelinedTimePartitions:
      {
        auto* msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgPipelinedTimePartitions>(recv_buff);
        size_t expected_size = msg->data_size;

        if (pipelined_time_partitions_buff_ == nullptr) {
          pipelined_time_partitions_buff_=
              new PeerRecvPipelinedTimePartitionsBuffer();
        }
        pipelined_time_partitions_buff_->pred_notice = msg->pred_notice;
        bool received_next_msg = false;
        if (expected_size > 0) {
          auto &byte_buff = pipelined_time_partitions_buff_->byte_buff;
          received_next_msg = ReceiveArbitraryBytes(
              sock, &recv_buff,
              &byte_buff, expected_size);
        }

        if (received_next_msg || (expected_size == 0)) {
          pipelined_time_partitions_buff_vec_.push_back(pipelined_time_partitions_buff_);
          pipelined_time_partitions_buff_ = nullptr;
          ServePipelinedTimePartitionsRequest();
        }

        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
        } else if (expected_size == 0) {
          ret = EventHandler<PollConn>::kClearOneMsg;
        } else {
          ret = EventHandler<PollConn>::kNoAction;
        }
      }
      break;
    case message::ExecuteMsgType::kReplyExecForLoopPredecessorCompletion:
      {
        pred_completed_ = true;
        ServeExecForLoopPredCompletion();
      }
      break;
    case message::ExecuteMsgType::kRequestDistArrayValue:
      {
        send_buff_.Copy(recv_buff);
        SendToExecutor();
        send_buff_.clear_to_send();
        send_buff_.reset_sent_sizes();
        ret = EventHandler<PollConn>::kClearOneMsg;
      }
      break;
    case message::ExecuteMsgType::kRequestDistArrayValues:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgRequestDistArrayValues>(recv_buff);
        size_t expected_size = msg->request_size;
        auto &recv_byte_buff
            = msg->is_requester_executor ? peer_recv_byte_buff_[sender_id]
            : server_recv_byte_buff_[sender_id];
        bool received_next_msg = ReceiveArbitraryBytes(
            sock, &recv_buff,
            &recv_byte_buff, expected_size);
        if (received_next_msg) {
          send_buff_.Copy(recv_buff);
          send_buff_.set_next_to_send(recv_byte_buff.GetBytes(),
                                      recv_byte_buff.GetSize());
          SendToExecutor();
          send_buff_.clear_to_send();
          send_buff_.reset_sent_sizes();
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
        } else {
          ret = EventHandler<PollConn>::kNoAction;
        }
      }
      break;
    case message::ExecuteMsgType::kReplyDistArrayValues:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgReplyDistArrayValues>(recv_buff);
        size_t expected_size = msg->reply_size;
        auto &recv_byte_buff = server_recv_byte_buff_[sender_id];
        bool received_next_msg = ReceiveArbitraryBytes(
            sock, &recv_buff,
            &recv_byte_buff, expected_size);
        LOG(INFO) << "received ReplyDistArrayValues from " << sender_id
                  << " received_next_msg = " << received_next_msg;
        if (received_next_msg) {
          auto *global_indexed_dist_array_data_buff = new PeerRecvGlobalIndexedDistArrayDataBuffer();
          global_indexed_dist_array_data_buff->server_id = sender_id;
          global_indexed_dist_array_data_buff->byte_buff.resize(recv_byte_buff.GetSize());
          memcpy(global_indexed_dist_array_data_buff->byte_buff.data(),
                 recv_byte_buff.GetBytes(), recv_byte_buff.GetSize());
          global_indexed_dist_arrays_buff_vec_.push_back(global_indexed_dist_array_data_buff);
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          recv_byte_buff.Reset(0);
          ServeGlobalIndexedDistArrayDataRequest();
        } else {
          ret = EventHandler<PollConn>::kNoAction;
        }
      }
      break;
    case message::ExecuteMsgType::kExecForLoopDistArrayCacheData:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgExecForLoopDistArrayCacheData>(recv_buff);
        size_t expected_size = msg->num_bytes;
        auto &recv_byte_buff = peer_recv_byte_buff_[sender_id];
        bool received_next_msg = ReceiveArbitraryBytes(
            sock, &recv_buff,
            &recv_byte_buff, expected_size);
        if (received_next_msg) {
          uint8_t *bytes = new uint8_t[recv_byte_buff.GetSize()];
          memcpy(bytes, recv_byte_buff.GetBytes(), recv_byte_buff.GetSize());
          peer_recv_byte_buff_[sender_id].Reset(0);
          message::ExecuteMsgHelper::CreateMsg<
            message::ExecuteMsgExecForLoopDistArrayCacheDataPtr>(&send_buff_, bytes);
          SendToExecutor();
          send_buff_.clear_to_send();
          send_buff_.reset_sent_sizes();
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
        } else {
          ret = EventHandler<PollConn>::kNoAction;
        }
      }
      break;
    case message::ExecuteMsgType::kExecForLoopDistArrayBufferData:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgExecForLoopDistArrayBufferData>(recv_buff);
        size_t expected_size = msg->num_bytes;
        auto &recv_byte_buff = peer_recv_byte_buff_[sender_id];
        bool received_next_msg = ReceiveArbitraryBytes(
            sock, &recv_buff,
            &recv_byte_buff, expected_size);
        if (received_next_msg) {
          uint8_t *bytes = new uint8_t[recv_byte_buff.GetSize()];
          memcpy(bytes, recv_byte_buff.GetBytes(), recv_byte_buff.GetSize());
          peer_recv_byte_buff_[sender_id].Reset(0);
          message::ExecuteMsgHelper::CreateMsg<
            message::ExecuteMsgExecForLoopDistArrayBufferDataPtr>(&send_buff_, bytes);
          SendToExecutor();
          send_buff_.clear_to_send();
          send_buff_.reset_sent_sizes();
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
        } else {
          ret = EventHandler<PollConn>::kNoAction;
        }
      }
      break;
    case message::ExecuteMsgType::kExecForLoopDone:
      {
        send_buff_.Copy(recv_buff);
        SendToExecutor();
        send_buff_.clear_to_send();
        send_buff_.reset_sent_sizes();
        ret = EventHandler<PollConn>::kClearOneMsg;
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
PeerRecvThread::ServePipelinedTimePartitionsRequest() {
  if (!has_executor_requested_pipeline_time_partitions_) return;
  if (pipelined_time_partitions_buff_vec_.empty()) return;
  auto* buff_vec = new PeerRecvPipelinedTimePartitionsBuffer*[pipelined_time_partitions_buff_vec_.size()];
  memcpy(buff_vec, pipelined_time_partitions_buff_vec_.data(),
         pipelined_time_partitions_buff_vec_.size() * sizeof(PeerRecvPipelinedTimePartitionsBuffer*));
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgReplyExecForLoopPipelinedTimePartitions>(
      &send_buff_, buff_vec, pipelined_time_partitions_buff_vec_.size());
  pipelined_time_partitions_buff_vec_.clear();
  has_executor_requested_pipeline_time_partitions_ = false;
  SendToExecutor();
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
}

void
PeerRecvThread::ServeGlobalIndexedDistArrayDataRequest() {
  LOG(INFO) << __func__ << " "
            << has_executor_requested_global_indexed_dist_array_data_
            << " " << global_indexed_dist_arrays_buff_vec_.size();
  if (!has_executor_requested_global_indexed_dist_array_data_) return;
  if (global_indexed_dist_arrays_buff_vec_.empty()) return;
  auto *buff_vec = new PeerRecvGlobalIndexedDistArrayDataBuffer*[global_indexed_dist_arrays_buff_vec_.size()];
  memcpy(buff_vec, global_indexed_dist_arrays_buff_vec_.data(),
         global_indexed_dist_arrays_buff_vec_.size() * sizeof(PeerRecvGlobalIndexedDistArrayDataBuffer*));
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgReplyExecForLoopGlobalIndexedDistArrayData>(
      &send_buff_, buff_vec, global_indexed_dist_arrays_buff_vec_.size());
  global_indexed_dist_arrays_buff_vec_.clear();
  has_executor_requested_global_indexed_dist_array_data_ = false;
  SendToExecutor();
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
}

void
PeerRecvThread::ServeExecForLoopPredCompletion() {
  if (!has_executor_requested_pred_completion_) return;
  if (!pred_completed_) return;
  message::ExecuteMsgHelper::CreateMsg<
    message::ExecuteMsgReplyExecForLoopPredecessorCompletion>(&send_buff_);
  has_executor_requested_pred_completion_ = false;
  SendToExecutor();
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
  pred_completed_ = false;
}

void
PeerRecvThread::SendToExecutor() {
  auto& send_buff = executor_conn_.get_send_buff();
  if (send_buff.get_remaining_to_send_size() > 0
      || send_buff.get_remaining_next_to_send_size() > 0) {
    bool sent = executor_->pipe.Send(&send_buff);
    while (!sent) {
      sent = executor_->pipe.Send(&send_buff);
    }
    send_buff.clear_to_send();
  }
  bool sent = executor_->pipe.Send(&send_buff_);
  if (!sent) {
    send_buff.CopyAndMoveNextToSend(&send_buff_);
    event_handler_.SetToReadWrite(&executor_conn_);
  }
  send_buff_.reset_sent_sizes();
}

}
}
