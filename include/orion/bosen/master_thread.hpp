#pragma once

#include <memory>
#include <iostream>
#include <vector>
#include <functional>
#include <glog/logging.h>

#include <unordered_map>
#include <orion/bosen/task.pb.h>
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
#include <orion/bosen/task_type.hpp>
#include <orion/bosen/dist_array_meta.hpp>

namespace orion {
namespace bosen {

class MasterThread {
 private:
  struct PollConn {
    enum class ConnType {
      listen = 0,
        executor = 1,
        server = 2,
        driver = 3,
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
      kForwardDriverMsgToAllExecutors = 3,
      kForwardDriverMsgToAllExecutorsAndServers = 4,
      kCreateDistArray = 6,
      kRespondToDriver = 7,
      kWaitingExecutorResponse = 40
  };

  const size_t kNumExecutors;
  const size_t kNumServers;
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

  Blob executor_server_recv_mem_;
  Blob executor_server_send_mem_;
  std::vector<PollConn> executor_server_poll_conn_;
  std::unordered_map<int32_t, PollConn*> executor_id_to_poll_conn_ptr_;
  std::unordered_map<int32_t, PollConn*> server_id_to_poll_conn_ptr_;
  std::unordered_map<conn::SocketConn*, int32_t> sock_conn_to_id_;
  std::unordered_map<int32_t, ByteBuffer> executor_byte_buff_;
  std::unordered_map<int32_t, ByteBuffer> server_byte_buff_;

  std::vector<std::unique_ptr<conn::SocketConn>> executors_;
  std::vector<std::unique_ptr<conn::SocketConn>> servers_;

  TaskType task_type_ {TaskType::kNone};

  size_t num_expected_executor_acks_ {0};
  size_t num_recved_executor_acks_ {0};
  size_t accum_result_size_ {0};

  Blob send_mem_;
  conn::SendBuffer send_buff_;
  std::vector<HostInfo> host_info_;
  // ByteBuffers are used to buffer arbitrary sized messages.
  // When send gets blocked, the universal SendBuffer will be shallow copied to
  // the connection's private SendBuffer. That means the ByteBuffers are NOT
  // copied.
  // To make that work, we require
  // 1) each driver request causes exactly one response from the master
  // 2) the master thread should not send requests/tasks to executors until the
  // all relevant executors of the previous request/task have responded
  ByteBuffer driver_recv_byte_buff_;
  ByteBuffer driver_send_byte_buff_;
  int32_t executor_in_action_ {0};

  size_t num_accepted_executors_and_servers_ {0};
  size_t num_identified_executors_ {0};
  size_t num_identified_servers_ {0};
  size_t num_closed_conns_ {0};
  bool stopped_all_ {false};

  State state_ {State::kInitialization};
  Action action_ {Action::kNone};

  std::unordered_map<int32_t, DistArrayMeta> dist_array_metas_;

  std::string lib_path_;

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
  int HandleExecuteMsg(PollConn *poll_conn_ptr);
  void ConstructDriverResponse(int32_t executor_id, size_t num_responses, size_t result_size);
  void BroadcastToAllExecutorsAndServers();
  void BroadcastToAllExecutors();
  void BroadcastToAllServers();
  void SendToExecutor(int executor_index);
  void SendToDriver();
};

MasterThread::MasterThread(const Config &config):
    kNumExecutors(config.kNumExecutors),
    kNumServers(config.kNumServers),
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
    executor_server_recv_mem_(config.kCommBuffCapacity
                              * (config.kNumExecutors + config.kNumServers)),
    executor_server_send_mem_(config.kCommBuffCapacity
                              * (config.kNumExecutors + config.kNumServers)),
    executor_server_poll_conn_(config.kNumExecutors + config.kNumServers),
    executors_(config.kNumExecutors),
    servers_(config.kNumServers),
    send_mem_(config.kCommBuffCapacity),
    send_buff_(send_mem_.data(), config.kCommBuffCapacity),
    host_info_(config.kNumExecutors + config.kNumServers) { }

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
  LOG(INFO) << __func__;
  conn::Socket accepted;
  listen_.sock.Accept(&accepted);
  if (num_accepted_executors_and_servers_ < kNumExecutors + kNumServers) {
    uint8_t *recv_mem = executor_server_recv_mem_.data()
                        + num_accepted_executors_and_servers_*kCommBuffCapacity;
    uint8_t *send_mem = executor_server_send_mem_.data()
                        + num_accepted_executors_and_servers_*kCommBuffCapacity;
    auto *sock_conn = new conn::SocketConn(
        accepted, recv_mem, send_mem, kCommBuffCapacity);
    auto &curr_poll_conn
        = executor_server_poll_conn_[num_accepted_executors_and_servers_];
    curr_poll_conn = {sock_conn, PollConn::ConnType::executor};
    event_handler_.SetToReadOnly(&curr_poll_conn);
    num_accepted_executors_and_servers_++;
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
      BroadcastToAllExecutorsAndServers();
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
  if (poll_conn_ptr->type == PollConn::ConnType::executor
      || poll_conn_ptr->type == PollConn::ConnType::server) {
    ret = HandleExecutorMsg(poll_conn_ptr);
  } else {
    ret = HandleDriverMsg(poll_conn_ptr);
  }

  while (action_ != Action::kNone
         && action_ != Action::kWaitingExecutorResponse
         && action_ != Action::kExit) {
    switch (action_) {
      case Action::kExecutorConnectToPeers:
        {
          LOG(INFO) << "ConnectToPeers!";
          message::Helper::CreateMsg<message::ExecutorConnectToPeers>(
              &send_buff_, kNumExecutors, kNumServers);
          send_buff_.set_next_to_send(host_info_.data(),
                                      (kNumExecutors + kNumServers) * sizeof(HostInfo));
          BroadcastToAllExecutorsAndServers();
          send_buff_.clear_to_send();
          num_expected_executor_acks_ = kNumExecutors + kNumServers;
          num_recved_executor_acks_ = 0;
          action_ = Action::kWaitingExecutorResponse;
        }
        break;
      case Action::kForwardDriverMsgToAllExecutorsAndServers:
        {
          LOG(INFO) << "forwarding to workers";
          send_buff_.Copy(driver_poll_conn_.conn->recv_buff);
          send_buff_.set_next_to_send(driver_recv_byte_buff_.GetBytes(),
                                      driver_recv_byte_buff_.GetSize());
          BroadcastToAllExecutorsAndServers();
          send_buff_.clear_to_send();
          num_expected_executor_acks_ = kNumExecutors + kNumServers;
          num_recved_executor_acks_ = 0;
          action_ = Action::kWaitingExecutorResponse;
        }
        break;
      case Action::kForwardDriverMsgToAllExecutors:
        {
          LOG(INFO) << "forwarding to workers";
          send_buff_.Copy(driver_poll_conn_.conn->recv_buff);
          send_buff_.set_next_to_send(driver_recv_byte_buff_.GetBytes(),
                                      driver_recv_byte_buff_.GetSize());
          BroadcastToAllExecutors();
          send_buff_.clear_to_send();
          num_expected_executor_acks_ = kNumExecutors;
          num_recved_executor_acks_ = 0;
          action_ = Action::kWaitingExecutorResponse;
        }
        break;
      case Action::kCreateDistArray:
        {
          message::ExecuteMsgHelper::CreateMsg<
            message::ExecuteMsgCreateDistArray>(
                &send_buff_, driver_recv_byte_buff_.GetSize());
          send_buff_.set_next_to_send(driver_recv_byte_buff_.GetBytes(),
                                      driver_recv_byte_buff_.GetSize());
          LOG(INFO) << "send buff size = " << send_buff_.get_size();
          BroadcastToAllExecutorsAndServers();
          send_buff_.clear_to_send();

          task::CreateDistArray create_dist_array;
          std::string task_buff(
              reinterpret_cast<const char*>(driver_recv_byte_buff_.GetBytes()),
              driver_recv_byte_buff_.GetSize());
          LOG(INFO) << "driver_recv_byte_buff size = "
                    << driver_recv_byte_buff_.GetSize();
          create_dist_array.ParseFromString(task_buff);
          int32_t id = create_dist_array.id();
          size_t num_dims = create_dist_array.num_dims();
          auto parent_type = create_dist_array.parent_type();
          if (parent_type == task::TEXT_FILE) {
            num_expected_executor_acks_ = kNumExecutors;
          } else {
            num_expected_executor_acks_ = kNumExecutors + kNumServers;
          }
          num_recved_executor_acks_ = 0;

          auto init_type = create_dist_array.init_type();
          DistArrayMeta *parent_dist_array_meta_ptr = nullptr;
          auto symbol = create_dist_array.symbol();
          if (parent_type == task::DIST_ARRAY) {
            int32_t parent_id = create_dist_array.parent_id();
            auto &parent_meta = dist_array_metas_.at(parent_id);
            parent_dist_array_meta_ptr = &parent_meta;
          }
          auto iter_pair = dist_array_metas_.emplace(
              std::piecewise_construct,
              std::forward_as_tuple(id),
              std::forward_as_tuple(num_dims, parent_type, init_type,
                                    parent_dist_array_meta_ptr, false,
                                    symbol));
          auto meta_iter = iter_pair.first;
          if (init_type != task::EMPTY) {
            meta_iter->second.AssignDims(create_dist_array.dims().data());
          }

          action_ = Action::kWaitingExecutorResponse;
        }
        break;
      case Action::kRespondToDriver:
        {
          auto *msg = message::DriverMsgHelper::CreateMsg<message::DriverMsgMasterResponse>(
              &send_buff_, accum_result_size_);
          LOG(INFO) << "accum_result_size = " << accum_result_size_;
          if (accum_result_size_ > 0) {
            if (num_expected_executor_acks_ == kNumExecutors
                || num_expected_executor_acks_ == kNumServers + kNumExecutors) {
              ConstructDriverResponse(-1, num_expected_executor_acks_, accum_result_size_);
            } else {
              ConstructDriverResponse(executor_in_action_,
                                      num_expected_executor_acks_,
                                      accum_result_size_);
            }
            LOG(INFO) << "driver_send_byte_buff_.size = "
                      << driver_send_byte_buff_.GetSize();
            send_buff_.set_next_to_send(driver_send_byte_buff_.GetBytes(),
                                        driver_send_byte_buff_.GetSize());
            msg->result_bytes += sizeof(size_t) * num_expected_executor_acks_;
          }

          SendToDriver();
          send_buff_.reset_sent_sizes();
          send_buff_.clear_to_send();
          action_ = Action::kNone;
          accum_result_size_ = 0;
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
  CHECK(action_ == Action::kNone)
      << "currently have a task in hand and can't handle any driver message! "
      << " action = " << static_cast<int>(action_);
  auto &recv_buff = poll_conn_ptr->get_recv_buff();
  auto msg_type = message::Helper::get_type(recv_buff);
  CHECK(msg_type == message::Type::kDriverMsg);

  auto driver_msg_type = message::DriverMsgHelper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kClearOneMsg;
  switch (driver_msg_type) {
    case message::DriverMsgType::kStop:
      {
        LOG(INFO) << "Driver commands stop!";
        message::Helper::CreateMsg<message::ExecutorStop>(&send_buff_);
        BroadcastToAllExecutors();
        stopped_all_ = true;
      }
      break;
    case message::DriverMsgType::kEvalExpr:
      {
        auto *msg = message::DriverMsgHelper::get_msg<
            message::DriverMsgEvalExpr>(recv_buff);
        size_t expected_size = msg->ast_size;
        LOG(INFO) << "received EvalExpr from driver, expected_size = "
                   << expected_size;

        bool received_next_msg
            = ReceiveArbitraryBytes(driver_.sock, &recv_buff, &driver_recv_byte_buff_,
                                    expected_size);
        if (received_next_msg) {
          LOG(INFO) << "msg receive is completed";
          executor_in_action_ = -1;
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kForwardDriverMsgToAllExecutorsAndServers;
        } else ret = EventHandler<PollConn>::kNoAction;

      }
      break;
    case message::DriverMsgType::kCreateDistArray:
      {
        LOG(INFO) << "master received CreateDistArrayMsg";
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgCreateDistArray>(recv_buff);
        size_t expected_size = msg->task_size;
        bool received_next_msg
            = ReceiveArbitraryBytes(driver_.sock, &recv_buff, &driver_recv_byte_buff_,
                                    expected_size);
        if (received_next_msg) {
          executor_in_action_ = -1;
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kCreateDistArray;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::DriverMsgType::kDefineVar:
      {
        LOG(INFO) << "master received DefineVarMsg";
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgDefineVar>(recv_buff);
        size_t expected_size = msg->var_info_size;
        bool received_next_msg
            = ReceiveArbitraryBytes(driver_.sock, &recv_buff,
                                    &driver_recv_byte_buff_,
                                    expected_size);
        if (received_next_msg) {
          executor_in_action_ = -1;
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kForwardDriverMsgToAllExecutorsAndServers;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::DriverMsgType::kRepartitionDistArray:
      {
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgRepartitionDistArray>(recv_buff);
        size_t expected_size = msg->task_size;
        bool received_next_msg
            = ReceiveArbitraryBytes(driver_.sock, &recv_buff,
                                    &driver_recv_byte_buff_,
                                    expected_size);
        if (received_next_msg) {
          executor_in_action_ = -1;
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kForwardDriverMsgToAllExecutors;

          task::RepartitionDistArray repartition_task;
          std::string task_buff(
              reinterpret_cast<const char*>(driver_recv_byte_buff_.GetBytes()),
              driver_recv_byte_buff_.GetSize());
          repartition_task.ParseFromString(task_buff);
          int32_t id = repartition_task.id();
          auto &meta = dist_array_metas_.at(id);
          int32_t partition_scheme = repartition_task.partition_scheme();
          int32_t index_type = repartition_task.index_type();
          meta.SetPartitionScheme(static_cast<DistArrayPartitionScheme>(partition_scheme));
          meta.SetIndexType(static_cast<DistArrayIndexType>(index_type));
          meta.ResetMaxPartitionIds();
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::DriverMsgType::kExecForLoop:
      {
        LOG(INFO) << "master thread received ExecForLoop";
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgExecForLoop>(recv_buff);
        size_t expected_size = msg->task_size;
        bool received_next_msg
            = ReceiveArbitraryBytes(driver_.sock, &recv_buff,
                                    &driver_recv_byte_buff_,
                                    expected_size);
        if (received_next_msg) {
          executor_in_action_ = -1;
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kForwardDriverMsgToAllExecutors;
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
        int32_t executor_id = msg->executor_id;
        host_info_[executor_id] = msg->host_info;
        auto* sock_conn = poll_conn_ptr->conn;
        executors_[executor_id].reset(sock_conn);
        sock_conn_to_id_[sock_conn] = executor_id;
        executor_byte_buff_.emplace(std::make_pair(executor_id, ByteBuffer()));
        executor_id_to_poll_conn_ptr_[executor_id] = poll_conn_ptr;
        num_identified_executors_++;
        LOG(INFO) << "num_identified_executors = " << num_identified_executors_
                  << " NumExecutors = " << kNumExecutors
                  << " NumServer = " << kNumServers;
        if (state_ == State::kInitialization
            && (num_identified_executors_ == kNumExecutors
                && num_identified_servers_ == kNumServers)) {
          action_ = Action::kExecutorConnectToPeers;
        }
        ret = EventHandler<PollConn>::kClearOneMsg;
      }
      break;
    case message::Type::kServerIdentity:
      {
        LOG(INFO) << "received server identity";
        auto *msg = message::Helper::get_msg<message::ServerIdentity>(recv_buff);
        int32_t server_id = msg->server_id;
        host_info_[kNumExecutors + server_id] = msg->host_info;
        auto* sock_conn = poll_conn_ptr->conn;
        LOG(INFO) << "before reset " << " server id = " << server_id;
        LOG(INFO) << (void*) servers_[server_id].get();
        servers_[server_id].reset(sock_conn);
        sock_conn_to_id_[sock_conn] = server_id;
        server_byte_buff_.emplace(std::make_pair(server_id, ByteBuffer()));
        server_id_to_poll_conn_ptr_[server_id] = poll_conn_ptr;
        poll_conn_ptr->type = PollConn::ConnType::server;
        num_identified_servers_++;
        LOG(INFO) << "num_identified_servers = " << num_identified_servers_;
        if (state_ == State::kInitialization
            && (num_identified_executors_ == kNumExecutors
                && num_identified_servers_ == kNumServers)) {
          action_ = Action::kExecutorConnectToPeers;
        }
        ret = EventHandler<PollConn>::kClearOneMsg;
      }
      break;
    case message::Type::kExecutorConnectToPeersAck:
      {
        LOG(INFO) << "connect to peers ack! " << num_recved_executor_acks_;
        num_recved_executor_acks_++;
        if (num_recved_executor_acks_ == num_expected_executor_acks_) {
          std::cout << "Your Orion cluster is ready!" << std::endl;
          std::cout << "Connect your client application to "
                    << kMasterIp << ":" << kMasterPort << std::endl;
        }
        ret = EventHandler<PollConn>::kClearOneMsg;
        action_ = Action::kNone;
      }
      break;
    case message::Type::kExecuteMsg:
      {
        ret = HandleExecuteMsg(poll_conn_ptr);
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

int
MasterThread::HandleExecuteMsg(PollConn *poll_conn_ptr) {
  auto &recv_buff = poll_conn_ptr->get_recv_buff();
  auto msg_type = message::ExecuteMsgHelper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kClearOneMsg;
  switch (msg_type) {
    case message::ExecuteMsgType::kExecutorAck:
      {
        ret = EventHandler<PollConn>::kNoAction;
        auto *ack_msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgExecutorAck>(recv_buff);
        size_t expected_size = ack_msg->result_size;
        int32_t executor_id = sock_conn_to_id_[poll_conn_ptr->conn];
        LOG(INFO) << "received ExecutorAck from "
                  << executor_id << " recv_buff = " << &recv_buff
                  << " is_servre = " << static_cast<int>(poll_conn_ptr->type);

        executor_in_action_ = executor_id;

        ByteBuffer &result_byte_buff
            = poll_conn_ptr->type == PollConn::ConnType::executor
            ? executor_byte_buff_[executor_id] : server_byte_buff_[executor_id];

        if (expected_size > 0) {
          bool received_next_msg =
              ReceiveArbitraryBytes(poll_conn_ptr->conn->sock, &recv_buff,
                                    &result_byte_buff, expected_size);
          if (received_next_msg) {
            ret = EventHandler<PollConn>::kClearOneAndNextMsg;
            CHECK_EQ(num_recved_executor_acks_ * expected_size, accum_result_size_);
            num_recved_executor_acks_++;
            accum_result_size_ += expected_size;
            LOG(INFO) << "accum_result_size = " << accum_result_size_;
          }
        } else {
          if (num_recved_executor_acks_ > 0)
            CHECK_EQ(accum_result_size_, 0);
          LOG(INFO) << "set accum_result_size to 0";
          num_recved_executor_acks_++;
          ret = EventHandler<PollConn>::kClearOneMsg;
          accum_result_size_ = 0;
        }
        LOG(INFO) << "ExecutorAck, num_recved_acks = " << num_recved_executor_acks_
                  << " num_expected_acks = " << num_expected_executor_acks_;
        if (num_recved_executor_acks_ == num_expected_executor_acks_) {
          action_ = Action::kRespondToDriver;
        }
      }
      break;
    case message::ExecuteMsgType::kTextFileLoadAck:
      {
        auto *ack_msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgTextFileLoadAck>(recv_buff);
        size_t expected_size = ack_msg->num_dims * sizeof(int64_t);
        int32_t executor_id = sock_conn_to_id_[poll_conn_ptr->conn];
        int32_t dist_array_id = ack_msg->dist_array_id;
        auto &dist_array_meta = dist_array_metas_.at(dist_array_id);
        LOG(INFO) << "TextFileLoadAck received! from " << executor_id
                  << " " << (void*) poll_conn_ptr->conn
                  << " recv_buff = " << (void*) &recv_buff
                  << " expected size = " << expected_size;
        if (expected_size > 0) {
          bool received_next_msg =
              ReceiveArbitraryBytes(poll_conn_ptr->conn->sock, &recv_buff,
                                    &executor_byte_buff_[executor_id], expected_size);
          if (received_next_msg) {
            LOG(INFO) << "received next message!";
            ret = EventHandler<PollConn>::kClearOneAndNextMsg;
            num_recved_executor_acks_++;
            std::vector<int64_t> max_keys(ack_msg->num_dims);
            memcpy(max_keys.data(), executor_byte_buff_[executor_id].GetBytes(),
                   ack_msg->num_dims*sizeof(int64_t));
            dist_array_meta.UpdateDimsMax(max_keys);
          } else {
            ret = EventHandler<PollConn>::kNoAction;
          }
        } else {
          num_recved_executor_acks_++;
          ret = EventHandler<PollConn>::kClearOneMsg;
        }

        if (num_recved_executor_acks_ == num_expected_executor_acks_) {
          if (expected_size > 0) {
            const auto &dims = dist_array_meta.GetDims();
            message::ExecuteMsgHelper::CreateMsg<
              message::ExecuteMsgDistArrayDims>(&send_buff_, ack_msg->num_dims,
                                                dist_array_id);
            send_buff_.set_next_to_send(dims.data(), dims.size()*sizeof(int64_t));
            num_recved_executor_acks_ = 0;
            num_expected_executor_acks_ = kNumExecutors;
            BroadcastToAllExecutors();
            send_buff_.reset_sent_sizes();
            send_buff_.clear_to_send();
            action_ = Action::kWaitingExecutorResponse;
          } else {
            action_ = Action::kRespondToDriver;
          }
        }
      }
      break;
    case message::ExecuteMsgType::kCreateDistArrayAck:
      {
        auto *ack_msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgCreateDistArrayAck>(recv_buff);
        num_recved_executor_acks_++;
        if (num_recved_executor_acks_ == num_expected_executor_acks_) {
          int32_t dist_array_id = ack_msg->dist_array_id;
          const auto &dist_array_meta = dist_array_metas_.at(dist_array_id);
          message::DriverMsgHelper::CreateMsg<message::DriverMsgMasterResponse>(
              &send_buff_,
              dist_array_meta.GetDims().size()*sizeof(int64_t));
          LOG(INFO) << "CreateDistArrayAck send to driver "
                    << dist_array_meta.GetDims().size()*sizeof(int64_t);
          send_buff_.set_next_to_send(dist_array_meta.GetDims().data(),
                                      dist_array_meta.GetDims().size()*sizeof(int64_t));
          SendToDriver();
          send_buff_.reset_sent_sizes();
          send_buff_.clear_to_send();
          num_recved_executor_acks_ = 0;
          num_expected_executor_acks_ = 0;
        }
        ret = EventHandler<PollConn>::kClearOneMsg;
        action_ = Action::kNone;
      }
      break;
    case message::ExecuteMsgType::kRepartitionDistArrayAck:
      {
        auto *ack_msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgRepartitionDistArrayAck>(recv_buff);
        num_recved_executor_acks_++;
        int dist_array_id = ack_msg->dist_array_id;
        LOG(INFO) << "dist_array_id = " << dist_array_id;
        size_t num_dims = ack_msg->num_dims;
        LOG(INFO) << "Ack num dims = " << num_dims;
        int32_t *max_ids = ack_msg->max_ids;
        auto &dist_array_meta = dist_array_metas_.at(dist_array_id);
        dist_array_meta.AccumMaxPartitionIds(max_ids, num_dims);
        if (num_recved_executor_acks_ == kNumExecutors) {
          auto &max_ids = dist_array_meta.GetMaxPartitionIds();
          auto *msg = message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRepartitionDistArrayMaxPartitionIds>(
              &send_buff_,
              dist_array_id,
              max_ids.size());
          for (size_t i = 0; i < max_ids.size(); i++) {
            msg->max_ids[i] = max_ids[i];
          }
          BroadcastToAllExecutors();
          send_buff_.reset_sent_sizes();
          send_buff_.clear_to_send();
          accum_result_size_ = 0;
          action_ = Action::kRespondToDriver;
        }
      }
      break;
    default:
      LOG(FATAL) << "unknown message type";
  }
  return ret;
}

void
MasterThread::ConstructDriverResponse(int32_t executor_id,
                                      size_t num_responses,
                                      size_t result_size) {
  if (result_size == 0) return;
  LOG(INFO) << __func__;
  if (executor_id < 0) {
    if (num_responses == kNumExecutors) {
      driver_send_byte_buff_.Reset(result_size + sizeof(size_t)*kNumExecutors);
      for (auto &byte_buff_iter : executor_byte_buff_) {
        *reinterpret_cast<size_t*>(driver_send_byte_buff_.GetAvailMem())
            = byte_buff_iter.second.GetSize();

        memcpy(driver_send_byte_buff_.GetAvailMem() + sizeof(size_t),
               byte_buff_iter.second.GetBytes(),
               byte_buff_iter.second.GetSize());
        driver_send_byte_buff_.IncSize(byte_buff_iter.second.GetSize() + sizeof(size_t));
      }
    } else if (num_responses == kNumServers + kNumExecutors) {
      driver_send_byte_buff_.Reset(result_size
                                   + sizeof(size_t)*(kNumExecutors + kNumServers));
      for (auto &byte_buff_iter : executor_byte_buff_) {
        *reinterpret_cast<size_t*>(driver_send_byte_buff_.GetAvailMem())
            = byte_buff_iter.second.GetSize();
        LOG(INFO) << "executor response size = "
                  << byte_buff_iter.second.GetSize();
        memcpy(driver_send_byte_buff_.GetAvailMem() + sizeof(size_t),
               byte_buff_iter.second.GetBytes(),
               byte_buff_iter.second.GetSize());
        driver_send_byte_buff_.IncSize(byte_buff_iter.second.GetSize() + sizeof(size_t));
      }

      for (auto &byte_buff_iter : server_byte_buff_) {
        *reinterpret_cast<size_t*>(driver_send_byte_buff_.GetAvailMem())
            = byte_buff_iter.second.GetSize();
        LOG(INFO) << "server response size = "
                  << byte_buff_iter.second.GetSize();

        memcpy(driver_send_byte_buff_.GetAvailMem() + sizeof(size_t),
               byte_buff_iter.second.GetBytes(),
               byte_buff_iter.second.GetSize());
        driver_send_byte_buff_.IncSize(byte_buff_iter.second.GetSize() + sizeof(size_t));
      }
    } else {
      LOG(FATAL) << "unexpected number of responses";
    }
  } else {
    auto &byte_buff = executor_byte_buff_.at(executor_id);
    driver_send_byte_buff_.Reset(result_size + sizeof(size_t));
    *reinterpret_cast<size_t*>(driver_send_byte_buff_.GetAvailMem())
        = byte_buff.GetSize();
    memcpy(driver_send_byte_buff_.GetAvailMem() + sizeof(size_t),
           byte_buff.GetBytes(),
           byte_buff.GetSize());
    driver_send_byte_buff_.IncSize(byte_buff.GetSize() + sizeof(size_t));
  }
}

void
MasterThread::BroadcastToAllExecutorsAndServers() {
  BroadcastToAllExecutors();
  BroadcastToAllServers();
}

void
MasterThread::BroadcastToAllExecutors() {
  for (int i = 0; i < kNumExecutors; ++i) {
    LOG(INFO) << "sending to " << i;
    conn::SendBuffer& send_buff = executors_[i]->send_buff;
    if (send_buff.get_remaining_to_send_size() > 0
        || send_buff.get_remaining_next_to_send_size() > 0) {
      LOG(INFO) << "blocked sending to " << i;
      bool sent = executors_[i]->sock.Send(&send_buff);
      while (!sent) {
        sent = executors_[i]->sock.Send(&send_buff);
      }
      send_buff.clear_to_send();
    }
    bool sent = executors_[i]->sock.Send(&send_buff_);
    if (!sent) {
      send_buff.CopyAndMoveNextToSend(&send_buff_);
      event_handler_.SetToReadWrite(executor_id_to_poll_conn_ptr_[i]);
    }
    send_buff_.reset_sent_sizes();
  }
}

void
MasterThread::BroadcastToAllServers() {
  for (int i = 0; i < kNumServers; ++i) {
    LOG(INFO) << "sending to " << i;
    conn::SendBuffer& send_buff = servers_[i]->send_buff;
    if (send_buff.get_remaining_to_send_size() > 0
        || send_buff.get_remaining_next_to_send_size() > 0) {
      LOG(INFO) << "blocked sending to " << i;
      bool sent = servers_[i]->sock.Send(&send_buff);
      while (!sent) {
        sent = servers_[i]->sock.Send(&send_buff);
      }
      send_buff.clear_to_send();
    }
    bool sent = servers_[i]->sock.Send(&send_buff_);
    if (!sent) {
      send_buff.CopyAndMoveNextToSend(&send_buff_);
      event_handler_.SetToReadWrite(executor_id_to_poll_conn_ptr_[i]);
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
    send_buff.CopyAndMoveNextToSend(&send_buff_);
    event_handler_.SetToReadWrite(executor_id_to_poll_conn_ptr_[executor_index]);
  }
}

void
MasterThread::SendToDriver() {
  conn::SendBuffer& send_buff = driver_.send_buff;
  if (send_buff.get_remaining_to_send_size() > 0
      || send_buff.get_remaining_next_to_send_size() > 0) {
    bool sent = driver_.sock.Send(&send_buff);
    while (!sent) {
      sent = driver_.sock.Send(&send_buff);
    }
    send_buff.clear_to_send();
  }
  bool sent = driver_.sock.Send(&send_buff_);
  LOG(INFO) << __func__ << " sent = " << sent;
  if (!sent) {
    send_buff.CopyAndMoveNextToSend(&send_buff_);
    event_handler_.SetToReadWrite(&driver_poll_conn_);
  }
}

} // end namespace bosen
} // end namespace orion
