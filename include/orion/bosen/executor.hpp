#pragma once

#include <memory>
#include <iostream>
#include <vector>
#include <unordered_map>

#include <orion/bosen/peer_recv_thread.hpp>
#include <orion/bosen/config.hpp>
#include <orion/noncopyable.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/util.hpp>
#include <orion/bosen/message.hpp>
#include <orion/bosen/execute_message.hpp>
#include <orion/bosen/host_info.hpp>
#include <orion/bosen/server_thread.hpp>
#include <orion/bosen/client_thread.hpp>
#include <orion/bosen/event_handler.hpp>
#include <orion/bosen/thread_pool.hpp>
#include <orion/bosen/julia_eval_thread.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/byte_buffer.hpp>
#include <orion/bosen/task.pb.h>
#include <orion/bosen/recv_arbitrary_bytes.hpp>
#include <orion/bosen/task_type.hpp>
#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/driver_message.hpp>
#include <orion/bosen/key.hpp>
#include <orion/bosen/abstract_exec_for_loop.hpp>
#include <orion/bosen/exec_for_loop_1d.hpp>
#include <orion/bosen/exec_for_loop_space_time_unordered.hpp>

namespace orion {
namespace bosen {

class Executor {
 private:
  struct PollConn {
    enum class ConnType {
      listen = 0,
        master = 1,
        compute = 2,
        peer = 3,
        peer_recv_thr = 4,
        server = 5
    };
    void* conn;
    ConnType type;

    bool Receive() {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer
          || type == ConnType::server) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.Recv(&(sock_conn->recv_buff));
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.Recv(&(pipe_conn->recv_buff));
      }
    }

    bool Send() {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer
          || type == ConnType::server) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.Send(&(sock_conn->send_buff));
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.Send(&(pipe_conn->send_buff));
      }
    }

    conn::RecvBuffer& get_recv_buff() {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer
          || type == ConnType::server) {
        return reinterpret_cast<conn::SocketConn*>(conn)->recv_buff;
      } else {
        return reinterpret_cast<conn::PipeConn*>(conn)->recv_buff;
      }
    }

    conn::SendBuffer& get_send_buff() {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer
          || type == ConnType::server) {
        return reinterpret_cast<conn::SocketConn*>(conn)->send_buff;
      } else {
        return reinterpret_cast<conn::PipeConn*>(conn)->send_buff;
      }
    }

    bool is_connect_event() const {
      return type == ConnType::listen;
    }

    int get_read_fd() const {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer
          || type == ConnType::server) {
        auto* sock_conn = reinterpret_cast<conn::SocketConn*>(conn);
        return sock_conn->sock.get_fd();
      } else {
        auto* pipe_conn = reinterpret_cast<conn::PipeConn*>(conn);
        return pipe_conn->pipe.get_read_fd();
      }
    }

    int get_write_fd() const {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::peer
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
      kConnectToPeers = 2,
      kAckConnectToPeers = 3,
      kEvalExpr = 5,
      kCreateDistArray = 6,
      kTextFileLoadAck = 7,
      kCreateDistArrayAck = 8,
      kDefineVar = 9,
      kExecutorAck = 10,
      kRepartitionDistArray = 11,
      kRepartitionDistArraySend = 12,
      kCreateExecForLoop = 13,
      kDefineJuliaDistArray = 14
            };

  static const int32_t kPortSpan = 100;
  const size_t kCommBuffCapacity;
  const size_t kNumExecutors;
  const size_t kNumLocalExecutors;
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  const std::string kListenIp;
  const uint16_t kListenPort;
  const size_t kThreadPoolSize;
  const int32_t kId;
  const int32_t kExecutorId;
  const int32_t kServerId;
  const Config kConfig;
  const bool kIsServer;

  EventHandler<PollConn> event_handler_;
  Blob master_send_mem_;
  Blob master_recv_mem_;
  conn::SocketConn master_;
  PollConn master_poll_conn_;
  Blob listen_send_mem_;
  Blob listen_recv_mem_;
  PollConn listen_poll_conn_;
  conn::SocketConn listen_;

  Blob send_mem_;
  conn::SendBuffer send_buff_;
  Action action_ {Action::kNone};
  Blob peer_send_mem_;
  Blob peer_recv_mem_;
  std::vector<std::unique_ptr<conn::SocketConn>> peer_;
  std::vector<PollConn> peer_conn_;
  std::vector<conn::Socket> peer_socks_;

  Blob server_send_mem_;
  Blob server_recv_mem_;
  std::vector<std::unique_ptr<conn::SocketConn>> server_;
  std::vector<PollConn> server_conn_;
  std::vector<conn::Socket> server_socks_;

  Blob thread_pool_recv_mem_;
  Blob thread_pool_send_mem_;
  std::vector<std::unique_ptr<conn::PipeConn>> compute_;
  std::vector<PollConn> compute_conn_;
  ThreadPool thread_pool_;

  Blob julia_eval_recv_mem_;
  Blob julia_eval_send_mem_;
  std::unique_ptr<conn::PipeConn> julia_eval_;
  PollConn julia_eval_conn_;
  JuliaEvalThread julia_eval_thread_;

  ExecJuliaFuncTask exec_julia_func_task_;
  ExecCppFuncTask exec_cpp_func_task_;
  EvalJuliaExprTask eval_julia_expr_task_;

  ByteBuffer master_recv_byte_buff_;

  TaskType task_type_ {TaskType::kNone};
  std::unordered_map<int32_t, DistArray> dist_arrays_;

  int32_t dist_array_under_operation_ {0};

  size_t num_connected_peers_ {0};
  size_t num_identified_peers_ {0};
  std::vector<HostInfo> host_info_;
  size_t host_info_recved_size_ {0};

  std::unique_ptr<PeerRecvThread> peer_recv_thr_;
  std::thread prt_runner_;

  Blob prt_send_mem_;
  Blob prt_recv_mem_;
  std::unique_ptr<conn::PipeConn> prt_pipe_conn_;
  PollConn prt_poll_conn_;
  ByteBuffer prt_recv_byte_buff_;
  void *julia_eval_result_ { nullptr };
  bool result_needed_ { true };
  //std::unordered_map<int32_t, Blob> repartition_send_buff_;
  bool connected_to_peers_ { false };
  std::unique_ptr<AbstractExecForLoop> exec_for_loop_;

 public:
  Executor(const Config& config, int32_t id, bool is_server);
  ~Executor();
  DISALLOW_COPY(Executor);
  void operator() ();
 private:
  void InitListener();
  void InitPeerRecvThread();
  void HandleConnection(PollConn* poll_conn_ptr);
  int HandleClosedConnection(PollConn *poll_conn_ptr);
  void HandleWriteEvent(PollConn *poll_con_ptr);
  void ConnectToMaster();
  int HandleMsg(PollConn* poll_conn_ptr);
  int HandleMasterMsg();
  int HandlePeerRecvThrMsg(PollConn* poll_conn_ptr);
  int HandlePeerRecvThrExecuteMsg();
  int HandlePipeMsg(PollConn* poll_conn_ptr);
  int HandleExecuteMsg();
  int HandleDriverMsg();
  void ConnectToPeers();
  void SetUpThreadPool();
  void ClearBeforeExit();
  void Send(PollConn* poll_conn_ptr, conn::SocketConn* sock_conn);
  void Send(PollConn* poll_conn_ptr, conn::PipeConn* pipe_conn);
  void EvalExpr();
  bool CreateDistArray();
  void DefineVar();
  void RepartitionDistArray();
  void RepartitionDistArraySend(int32_t dist_array_id, DistArray *dist_array);
  void CreateExecForLoop();
  void DefineJuliaDistArray(DistArray *dist_array);
  bool CheckAndExecuteForLoop(bool* waiting_for_peers, bool *completed);
  bool CheckAndExecuteForLoopUntilNotRunnable();
  void ExecuteForLoopTile(AbstractDistArrayPartition* partition_to_exec,
                          const std::string &loop_batch_func_name);
  void ExecForLoopSendResults(int32_t time_partition_id_to_send);
  void RequestDistArrayData();
};

Executor::Executor(const Config& config,
                   int32_t index,
                   bool is_server):
    kCommBuffCapacity(config.kCommBuffCapacity),
    kNumExecutors(config.kNumExecutors),
    kNumLocalExecutors(config.kNumExecutorsPerWorker),
    kMasterIp(config.kMasterIp),
    kMasterPort(config.kMasterPort),
    kListenIp(config.kWorkerIp),
    kListenPort(config.kWorkerPort + index * kPortSpan),
    kThreadPoolSize(config.kExecutorThreadPoolSize),
    kId(config.kWorkerId * (config.kNumExecutorsPerWorker
                            + config.kNumServersPerWorker) + index),
    kExecutorId(config.kWorkerId * config.kNumExecutorsPerWorker + index),
    kServerId(config.kWorkerId * config.kNumServersPerWorker + (index - config.kNumExecutorsPerWorker)),
    kConfig(config),
    kIsServer(is_server),
    master_send_mem_(kCommBuffCapacity),
    master_recv_mem_(kCommBuffCapacity),
    master_(conn::Socket(),
            master_recv_mem_.data(),
            master_send_mem_.data(),
            kCommBuffCapacity),
    listen_send_mem_(kCommBuffCapacity),
    listen_recv_mem_(kCommBuffCapacity),
    listen_(conn::Socket(),
            listen_recv_mem_.data(),
            listen_send_mem_.data(),
            kCommBuffCapacity),
    send_mem_(kCommBuffCapacity),
    send_buff_(send_mem_.data(), kCommBuffCapacity),
    peer_send_mem_(config.kCommBuffCapacity*config.kNumExecutors),
    peer_recv_mem_(config.kCommBuffCapacity*config.kNumExecutors),
    peer_(config.kNumExecutors),
    peer_conn_(config.kNumExecutors),
    peer_socks_(config.kNumExecutors),
    server_send_mem_(is_server ? 0 : config.kCommBuffCapacity*config.kNumServers),
    server_recv_mem_(is_server ? 0 : config.kCommBuffCapacity*config.kNumServers),
    server_(is_server ? 0 : config.kNumServers),
    server_conn_(is_server ? 0 : config.kNumServers),
    server_socks_(is_server ? 0 : config.kNumServers),
    thread_pool_recv_mem_(kCommBuffCapacity*kThreadPoolSize),
    thread_pool_send_mem_(kCommBuffCapacity*kThreadPoolSize),
    compute_(kThreadPoolSize),
    compute_conn_(kThreadPoolSize),
    thread_pool_(config.kExecutorThreadPoolSize, config.kCommBuffCapacity),
    julia_eval_recv_mem_(kCommBuffCapacity),
    julia_eval_send_mem_(kCommBuffCapacity),
    julia_eval_thread_(kCommBuffCapacity, config.kOrionHome),
    host_info_(kNumExecutors + config.kNumServers),
    prt_send_mem_(kCommBuffCapacity),
    prt_recv_mem_(kCommBuffCapacity) { }

Executor::~Executor() { }

void
Executor::operator() () {
  InitListener();

  listen_poll_conn_ = {&listen_, PollConn::ConnType::listen};
  int ret = event_handler_.SetToReadOnly(&listen_poll_conn_);
  CHECK_EQ(ret, 0);
  ConnectToMaster();

  master_poll_conn_ = {&master_, PollConn::ConnType::master};
  ret = event_handler_.SetToReadOnly(&master_poll_conn_);
  CHECK_EQ(ret, 0);

  {
    HostInfo host_info;
    ret = GetIPFromStr(kListenIp.c_str(), &host_info.ip);
    CHECK_NE(ret, 0);
    host_info.port = kListenPort;

    if (kIsServer) {
      message::Helper::CreateMsg<
        message::ServerIdentity>(&send_buff_, kServerId, host_info);
    } else {
      message::Helper::CreateMsg<
        message::ExecutorIdentity>(&send_buff_, kExecutorId, host_info);
    }
    Send(&master_poll_conn_, &master_);
    send_buff_.clear_to_send();
  }

  event_handler_.SetConnectEventHandler(
      std::bind(&Executor::HandleConnection, this,
                std::placeholders::_1));

  event_handler_.SetClosedConnectionHandler(
      std::bind(&Executor::HandleClosedConnection, this,
                std::placeholders::_1));

  event_handler_.SetReadEventHandler(
      std::bind(&Executor::HandleMsg, this, std::placeholders::_1));

  event_handler_.SetWriteEventHandler(
      std::bind(&Executor::HandleWriteEvent, this, std::placeholders::_1));

  SetUpThreadPool();

  while (true) {
    event_handler_.WaitAndHandleEvent();
    if (action_ == Action::kExit) break;
  }
  ClearBeforeExit();
}

void
Executor::InitListener () {
  uint32_t ip;
  int ret = GetIPFromStr(kListenIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = listen_.sock.Bind(ip, kListenPort);
  CHECK_EQ(ret, 0);
  ret = listen_.sock.Listen(kNumExecutors);
  CHECK_EQ(ret, 0);
}

void
Executor::InitPeerRecvThread() {
  peer_recv_thr_.reset(
      new PeerRecvThread(
          kId,
          kExecutorId,
          kServerId,
          kIsServer,
          peer_socks_,
          server_socks_,
          kCommBuffCapacity));

  auto prt_pipe = peer_recv_thr_->GetExecutorPipe();
  prt_pipe_conn_.reset(
      new conn::PipeConn(
          prt_pipe,
          prt_recv_mem_.data(),
          prt_send_mem_.data(),
          kCommBuffCapacity));
  prt_poll_conn_.conn = prt_pipe_conn_.get();
  prt_poll_conn_.type = PollConn::ConnType::peer_recv_thr;
  event_handler_.SetToReadOnly(&prt_poll_conn_);

  prt_runner_ = std::thread(
      &PeerRecvThread::operator(),
      peer_recv_thr_.get());
}

void
Executor::HandleConnection(PollConn* poll_conn_ptr) {
  conn::Socket accepted;
  int ret = listen_.sock.Accept(&accepted);
  CHECK(ret == 0);

  peer_socks_[num_connected_peers_] = accepted;
  num_connected_peers_++;

  if (kIsServer) {
    if (num_connected_peers_ == kNumExecutors) {
      InitPeerRecvThread();
    }
  } else {
    CHECK(kExecutorId != 0);
    if (num_connected_peers_ == kExecutorId
        && connected_to_peers_) {
      InitPeerRecvThread();
    }
  }
}

int
Executor::HandleClosedConnection(PollConn *poll_conn_ptr) {
  int ret = EventHandler<PollConn>::kNoAction;
  auto type = poll_conn_ptr->type;
  if (type == PollConn::ConnType::listen
      || type == PollConn::ConnType::compute
      || type == PollConn::ConnType::peer
      || type == PollConn::ConnType::server) {
    int ret = event_handler_.Remove(poll_conn_ptr);
    CHECK_EQ(ret, 0);
  } else {
    int ret = event_handler_.Remove(poll_conn_ptr);
    CHECK_EQ(ret, 0);
    auto* conn = reinterpret_cast<conn::SocketConn*>(poll_conn_ptr->conn);
    conn->sock.Close();
    action_ = Action::kExit;
    ret |= EventHandler<PollConn>::kExit;
  }
  return ret;
}

void
Executor::HandleWriteEvent(PollConn* poll_conn_ptr) {
  bool sent = poll_conn_ptr->Send();

  if (sent) {
    auto &send_buff = poll_conn_ptr->get_send_buff();
    send_buff.clear_to_send();
    if (poll_conn_ptr->type == PollConn::ConnType::peer
        || poll_conn_ptr->type == PollConn::ConnType::server)
      event_handler_.Remove(poll_conn_ptr);
    else
      event_handler_.SetToReadOnly(poll_conn_ptr);
  }
}

void
Executor::ConnectToMaster() {
  uint32_t ip;
  int ret = GetIPFromStr(kMasterIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = master_.sock.Connect(ip, kMasterPort);
  CHECK(ret == 0) << "executor failed connecting to master";
}

int
Executor::HandleMsg(PollConn* poll_conn_ptr) {
  int ret = 0;
  if (poll_conn_ptr->type == PollConn::ConnType::master) {
    ret = HandleMasterMsg();
  } else if (poll_conn_ptr->type == PollConn::ConnType::peer_recv_thr) {
    ret = HandlePeerRecvThrMsg(poll_conn_ptr);
  } else if (poll_conn_ptr->type == PollConn::ConnType::peer
             || poll_conn_ptr->type == PollConn::ConnType::server) {
    LOG(FATAL) << "Should not receive peer msgs";
  } else {
    ret = HandlePipeMsg(poll_conn_ptr);
  }

  while (action_ != Action::kNone
         && action_ != Action::kExit) {
    switch (action_) {
      case Action::kConnectToPeers:
        {
          if (!kIsServer) {
            ConnectToPeers();
            connected_to_peers_ = true;
            if (kNumExecutors == 1 || kExecutorId == 0) {
              // if I don't expect to receive connections,
              // go ahead initialize peer recv threads
              action_ = Action::kAckConnectToPeers;
              InitPeerRecvThread();
            } else if (num_connected_peers_ == kExecutorId) {
              // I might receive connections before told so by driver
              CHECK(peer_recv_thr_.get() == nullptr);
              action_ = Action::kNone;
              InitPeerRecvThread();
            } else action_ = Action::kNone;
          } else {
            if (num_connected_peers_ == kNumExecutors) {
              action_ = Action::kNone;
              InitPeerRecvThread();
            } else action_ = Action::kNone;
          }
        }
        break;
      case Action::kAckConnectToPeers:
        {
          int ret = event_handler_.Remove(&prt_poll_conn_);
          CHECK_EQ(ret, 0) << ret;
          message::Helper::CreateMsg<message::ExecutorConnectToPeersAck>(&send_buff_);
          Send(&master_poll_conn_, &master_);
          send_buff_.clear_to_send();
          action_ = Action::kNone;
        }
        break;
      case Action::kEvalExpr:
        {
          task_type_ = TaskType::kEvalJuliaExpr;
          EvalExpr();
          action_ = Action::kNone;
        }
        break;
      case Action::kCreateDistArray:
        {
          task_type_ = TaskType::kExecCppFunc;
          bool server_ack = CreateDistArray();
          if (server_ack) {
            action_ = Action::kCreateDistArrayAck;
          } else {
            action_ = Action::kNone;
          }
        }
        break;
      case Action::kExecutorAck:
        {
          size_t result_size = 0;
          const void *result_mem = nullptr;
          if (result_needed_) {
            if (task_type_ == TaskType::kExecJuliaFunc) {
              result_size = exec_julia_func_task_.result_buff.size();
              result_mem = exec_julia_func_task_.result_buff.data();
            } else if (task_type_ == TaskType::kEvalJuliaExpr) {
              result_size = eval_julia_expr_task_.result_buff.size();
              result_mem = eval_julia_expr_task_.result_buff.data();
            } else if (task_type_ == TaskType::kExecCppFunc) {
            } else {
              LOG(FATAL) << "error!";
            }
          }

          LOG(INFO) << "task_type = " << static_cast<int>(task_type_)
                    << " result_size = " << result_size;

          message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgExecutorAck>(
              &send_buff_, result_size);
          if (result_size > 0)
            send_buff_.set_next_to_send(result_mem, result_size);
          Send(&master_poll_conn_, &master_);
          send_buff_.clear_to_send();
          send_buff_.reset_sent_sizes();
          action_ = Action::kNone;
          result_needed_ = true;
        }
        break;
      case Action::kTextFileLoadAck:
        {
          message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgTextFileLoadAck>(
              &send_buff_,
              exec_cpp_func_task_.result_buff.size() / sizeof(int64_t),
              dist_array_under_operation_);
          if (exec_cpp_func_task_.result_buff.size() > 0) {
            send_buff_.set_next_to_send(exec_cpp_func_task_.result_buff.data(),
                                        exec_cpp_func_task_.result_buff.size());
          }
          Send(&master_poll_conn_, &master_);
          send_buff_.clear_to_send();
          send_buff_.reset_sent_sizes();
          action_ = Action::kNone;
        }
        break;
      case Action::kCreateDistArrayAck:
        {
          message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgCreateDistArrayAck>(
              &send_buff_, dist_array_under_operation_);
          Send(&master_poll_conn_, &master_);
          send_buff_.clear_to_send();
          send_buff_.reset_sent_sizes();
          action_ = Action::kNone;
        }
        break;
      case Action::kDefineJuliaDistArray:
        {
          auto &dist_array = dist_arrays_.at(dist_array_under_operation_);
          DefineJuliaDistArray(&dist_array);
          action_ = Action::kNone;
        }
        break;
      case Action::kDefineVar:
        {
          task_type_ = TaskType::kExecCppFunc;
          DefineVar();
          action_ = Action::kNone;
        }
        break;
      case Action::kRepartitionDistArray:
        {
          task_type_ = TaskType::kExecCppFunc;
          RepartitionDistArray();
          action_ = Action::kNone;
        }
        break;
      case Action::kRepartitionDistArraySend:
        {
          if (kNumExecutors > 1) {
            auto &dist_array = dist_arrays_.at(dist_array_under_operation_);
            RepartitionDistArraySend(dist_array_under_operation_,
                                     &dist_array);
            action_ = Action::kNone;
            event_handler_.SetToReadOnly(&prt_poll_conn_);
          } else {
            action_ = Action::kExecutorAck;
          }
        }
        break;
      case Action::kCreateExecForLoop:
        {
          CreateExecForLoop();
          bool completed = CheckAndExecuteForLoopUntilNotRunnable();
          if (completed)
            action_ = Action::kExecutorAck;
          else
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
Executor::HandleMasterMsg() {
  auto &sock = master_.sock;
  auto &recv_buff = master_.recv_buff;

  auto msg_type = message::Helper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kNoAction;
  switch (msg_type) {
    case message::Type::kExecutorConnectToPeers:
      {
        auto *msg = message::Helper::get_msg<message::ExecutorConnectToPeers>(
              recv_buff);
        size_t expected_size = (msg->num_executors + msg->num_servers) * sizeof(HostInfo);
        //size_t expected_size = (msg->num_executors) * sizeof(HostInfo);
        bool received_next_msg =
            ReceiveArbitraryBytes(sock, &recv_buff,
                                  reinterpret_cast<uint8_t*>(host_info_.data()),
                                  &host_info_recved_size_, expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kConnectToPeers;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::Type::kExecutorStop:
      {
        LOG(INFO) << "master commands stop!";
        action_ = Action::kExit;
        ret = EventHandler<PollConn>::kClearOneMsg | EventHandler<PollConn>::kExit;
      }
      break;
    case message::Type::kExecuteMsg:
      {
        ret = HandleExecuteMsg();
      }
      break;
    case message::Type::kDriverMsg:
      {
        ret = HandleDriverMsg();
      }
      break;
    default:
      {
        LOG(FATAL) << "unknown message type " << static_cast<int>(msg_type);
      }
      break;
  }
  return ret;
}

int
Executor::HandlePeerRecvThrMsg(PollConn* poll_conn_ptr) {
  auto &pipe = reinterpret_cast<conn::PipeConn*>(poll_conn_ptr->conn)->pipe;
  auto &recv_buff = poll_conn_ptr->get_recv_buff();

  auto msg_type = message::Helper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kClearOneMsg;
  switch (msg_type) {
    case message::Type::kExecutorConnectToPeersAck:
      {
        size_t expected_size = kNumExecutors*sizeof(int);
        bool received_next_msg =
            ReceiveArbitraryBytes(pipe, &recv_buff,
                                  &prt_recv_byte_buff_,
                                  expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kAckConnectToPeers;
          int *sock_fds = reinterpret_cast<int*>(prt_recv_byte_buff_.GetBytes());

          for (int i = 0; i < kId; i++) {
            conn::Socket sock(sock_fds[i]);
            uint8_t *recv_mem = peer_recv_mem_.data()
                      + kCommBuffCapacity * i;

            uint8_t *send_mem = peer_send_mem_.data()
                                + kCommBuffCapacity * i;

            auto &curr_poll_conn = peer_conn_[i];

            auto *sock_conn = new conn::SocketConn(
                sock, recv_mem, send_mem, kCommBuffCapacity);

            curr_poll_conn.conn = sock_conn;
            curr_poll_conn.type = PollConn::ConnType::peer;
            peer_[i].reset(sock_conn);
          }
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::Type::kExecuteMsg:
      {
        ret = HandlePeerRecvThrExecuteMsg();
      }
      break;
    default:
      {
        LOG(FATAL) << "unknown message type " << static_cast<int>(msg_type);
      }
      break;
  }
  return ret;
}

int
Executor::HandlePeerRecvThrExecuteMsg() {
  auto &recv_buff = prt_poll_conn_.get_recv_buff();
  int ret = EventHandler<PollConn>::kClearOneMsg;
  auto msg_type = message::ExecuteMsgHelper::get_type(recv_buff);
  switch (msg_type) {
    case message::ExecuteMsgType::kRepartitionDistArrayRecved:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgRepartitionDistArrayRecved>(recv_buff);
        auto *partition_recv_buff = reinterpret_cast<PeerRecvRepartitionDistArrayDataBuffer*>(
            msg->data_buff);
        int32_t dist_array_id = partition_recv_buff->dist_array_id;
        auto &dist_array = dist_arrays_.at(dist_array_id);
        auto byte_buffs = partition_recv_buff->byte_buffs;
        for (auto &buff_pair : byte_buffs) {
          auto &blob = buff_pair.second;
          dist_array.RepartitionDeserialize(blob.GetBytes(), blob.GetSize());
        }
        delete partition_recv_buff;
        dist_array.CheckAndBuildIndex();

        auto &max_ids = dist_array.GetMeta().GetMaxPartitionIds();
        auto *ack_msg = message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRepartitionDistArrayAck>(
            &send_buff_,
            dist_array_id,
            max_ids.size());
         for (size_t i = 0; i < max_ids.size(); i++) {
          ack_msg->max_ids[i] = max_ids[i];
        }
        Send(&master_poll_conn_, &master_);
        send_buff_.clear_to_send();
        send_buff_.reset_sent_sizes();
        action_ = Action::kNone;
        event_handler_.Remove(&prt_poll_conn_);
      }
      break;
    case message::ExecuteMsgType::kReplyExecForLoopDistArrayData:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgReplyExecForLoopDistArrayData>(recv_buff);
        PeerRecvDistArrayDataBuffer* buffer_vec
            = reinterpret_cast<PeerRecvDistArrayDataBuffer*>(msg->buff_vec_ptr);
        size_t num_buffs = msg->num_buffs;

        for (size_t i = 0; i < num_buffs; i++) {
          auto &buff = buffer_vec[i];
          auto dist_array_id = buff.dist_array_id;
          auto partition_id = buff.partition_id;
          auto& dist_array = dist_arrays_.at(dist_array_id);
          auto* dist_array_partition = dist_array.CreatePartition();
          dist_array_partition->Deserialize(buff.data, buff.expected_size);
          dist_array.AddPartition(partition_id, dist_array_partition);
          delete[] buff.data;
        }
        delete[] msg->buff_vec_ptr;
        int event_handler_ret = event_handler_.Remove(&prt_poll_conn_);
        CHECK_EQ(event_handler_ret, 0) << event_handler_ret;

        bool completed = CheckAndExecuteForLoopUntilNotRunnable();
        if (completed)
          action_ = Action::kExecutorAck;
        else
          action_ = Action::kNone;
        ret = EventHandler<PollConn>::kClearOneMsg;
      }
      break;
    default:
      LOG(FATAL) << "unexpected msg type = " << static_cast<int>(msg_type);
  }
  return ret;
}

int
Executor::HandlePipeMsg(PollConn* poll_conn_ptr) {
  auto &recv_buff = poll_conn_ptr->get_recv_buff();
  auto msg_type = message::ExecuteMsgHelper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kClearOneMsg;
  switch (msg_type) {
    case message::ExecuteMsgType::kJuliaEvalAck:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgJuliaEvalAck>(
              recv_buff);
        if (auto *task = dynamic_cast<ExecCppFuncTask*>(msg->task)) {
          LOG(INFO) << "task->label = " << static_cast<int>(task->label);
          switch (task->label) {
            case TaskLabel::kNone:
              {
                action_ = Action::kExecutorAck;
              }
              break;
            case TaskLabel::kLoadDistArrayFromTextFile:
              {
                action_ = Action::kTextFileLoadAck;
              }
              break;
            case TaskLabel::kDefineVar:
              {
                action_ = Action::kExecutorAck;
              }
              break;
            case TaskLabel::kComputeRepartition:
              {
                action_ = Action::kRepartitionDistArraySend;
              }
              break;
            case TaskLabel::kRandomInitDistArray:
              {
                action_ = Action::kDefineJuliaDistArray;
              }
              break;
            case TaskLabel::kDefineJuliaDistArray:
              {
                action_ = Action::kCreateDistArrayAck;
              }
              break;
            case TaskLabel::kExecForLoopTile:
              {
                bool completed = CheckAndExecuteForLoopUntilNotRunnable();
                if (completed)
                  action_ = Action::kExecutorAck;
                else
                  action_ = Action::kNone;
              }
              break;
            default:
              LOG(FATAL) << "unknown task label";
          }
        } else {
          action_ = Action::kExecutorAck;
        }
      }
      break;
    default:
      {
        LOG(FATAL) << "unknown message type " << static_cast<int>(msg_type);
      }
      break;
  }
  return ret;
}

int
Executor::HandleExecuteMsg() {
  auto &recv_buff = master_.recv_buff;
  auto msg_type = message::ExecuteMsgHelper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kClearOneMsg;
  switch (msg_type) {
    case message::ExecuteMsgType::kCreateDistArray:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<message::ExecuteMsgCreateDistArray>(
              recv_buff);
        size_t expected_size = msg->task_size;
        bool received_next_msg =
            ReceiveArbitraryBytes(master_.sock, &recv_buff,
                                  &master_recv_byte_buff_,
                                  expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kCreateDistArray;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::ExecuteMsgType::kDistArrayDims:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgDistArrayDims>(recv_buff);
        size_t expected_size = msg->num_dims*sizeof(int64_t);
        bool received_next_msg =
            ReceiveArbitraryBytes(master_.sock, &recv_buff,
                                  &master_recv_byte_buff_,
                                  expected_size);
        if (received_next_msg) {
          std::vector<int64_t> dims(msg->num_dims, 0);
          memcpy(dims.data(), master_recv_byte_buff_.GetBytes(),
                 msg->num_dims*sizeof(int64_t));
          int32_t dist_array_id = msg->dist_array_id;
          auto &dist_array = dist_arrays_.at(dist_array_id);
          dist_array.SetDims(dims);
          dist_array_under_operation_ = dist_array_id;
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kDefineJuliaDistArray;

        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::ExecuteMsgType::kRepartitionDistArrayMaxPartitionIds:
    {
      auto *msg = message::ExecuteMsgHelper::get_msg<
        message::ExecuteMsgRepartitionDistArrayMaxPartitionIds>(recv_buff);
      size_t num_dims = msg->num_dims;
      int32_t *max_ids = msg->max_ids;
      int32_t dist_array_id = msg->dist_array_id;
      auto &dist_array_meta = dist_arrays_.at(dist_array_id).GetMeta();
      dist_array_meta.ResetMaxPartitionIds();
      dist_array_meta.AccumMaxPartitionIds(max_ids, num_dims);
      ret = EventHandler<PollConn>::kClearOneMsg;
      action_ = Action::kNone;
    }
    break;
    default:
      {
        LOG(FATAL) << "unknown message type " << static_cast<int>(msg_type);
      }
      break;
  }

  return ret;
}

int
Executor::HandleDriverMsg() {
  auto &recv_buff = master_.recv_buff;
  auto msg_type = message::DriverMsgHelper::get_type(recv_buff);
  int ret = EventHandler<PollConn>::kClearOneMsg;
  switch (msg_type) {
    case message::DriverMsgType::kEvalExpr:
      {
        auto* msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgEvalExpr>(recv_buff);
        size_t expected_size = msg->ast_size;
        bool received_next_msg
            = ReceiveArbitraryBytes(master_.sock, &recv_buff,
                                    &master_recv_byte_buff_,
                                    expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kEvalExpr;
          LOG(INFO) << __func__ << " EvalExpr";
        } else {
          ret = EventHandler<PollConn>::kNoAction;
          action_ = Action::kNone;
        }
      }
      break;
    case message::DriverMsgType::kDefineVar:
      {
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgDefineVar>(recv_buff);
        size_t expected_size = msg->var_info_size;
        bool received_next_msg =
            ReceiveArbitraryBytes(master_.sock,
                                  &recv_buff,
                                  &master_recv_byte_buff_,
                                  expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kDefineVar;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::DriverMsgType::kRepartitionDistArray:
      {
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgRepartitionDistArray>(recv_buff);
        size_t expected_size = msg->task_size;
        bool received_next_msg =
            ReceiveArbitraryBytes(master_.sock,
                                  &recv_buff,
                                  &master_recv_byte_buff_,
                                  expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kRepartitionDistArray;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::DriverMsgType::kExecForLoop:
      {
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgExecForLoop>(recv_buff);
        size_t expected_size = msg->task_size;
        bool received_next_msg =
            ReceiveArbitraryBytes(master_.sock,
                                  &recv_buff,
                                  &master_recv_byte_buff_,
                                  expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kCreateExecForLoop;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    default:
      {
        LOG(FATAL) << "unknown message type " << static_cast<int>(msg_type);
      }
      break;
  }

  return ret;
}

void
Executor::ConnectToPeers() {
  HostInfo host_info;
  int ret = GetIPFromStr(kListenIp.c_str(), &host_info.ip);
  CHECK_NE(ret, 0);
  host_info.port = kListenPort;
  message::Helper::CreateMsg<
    message::ExecutorIdentity>(&send_buff_, kId, host_info);

  for (int32_t i = kNumExecutors; i < kNumExecutors + kConfig.kNumServers; i++) {
    LOG(INFO) << __func__ << " " << i;
    uint32_t ip = host_info_[i].ip;
    uint16_t port = host_info_[i].port;
    int32_t server_id = i - kNumExecutors;
    conn::Socket &server_sock = server_socks_[server_id];
    ret = server_sock.Connect(ip, port);
    CHECK(ret == 0) << "executor failed connecting to server " << server_id
                    << " ip = " << ip << " port = " << port;
    server_[server_id].reset(new conn::SocketConn(server_sock,
                                                  server_recv_mem_.data() + kCommBuffCapacity * server_id,
                                                  server_send_mem_.data() + kCommBuffCapacity * server_id,
                                                  kCommBuffCapacity));
    server_conn_[server_id].conn = server_[server_id].get();
    server_conn_[server_id].type = PollConn::ConnType::server;
    Send(&server_conn_[server_id], server_[server_id].get());
  }

  for (int32_t i = kExecutorId + 1; i < kNumExecutors; i++) {
    uint32_t ip = host_info_[i].ip;
    uint16_t port = host_info_[i].port;
    conn::Socket &peer_sock = peer_socks_[i];
    ret = peer_sock.Connect(ip, port);
    CHECK(ret == 0) << "executor failed connecting to peer " << i
                    << " ip = " << ip << " port = " << port;
    peer_[i].reset(new conn::SocketConn(peer_sock,
                                        peer_recv_mem_.data() + kCommBuffCapacity*i,
                                        peer_send_mem_.data() + kCommBuffCapacity*i,
                                        kCommBuffCapacity));
    peer_conn_[i].conn = peer_[i].get();
    peer_conn_[i].type = PollConn::ConnType::peer;
    Send(&peer_conn_[i], peer_[i].get());
  }
  send_buff_.clear_to_send();
}

void
Executor::SetUpThreadPool() {
  auto read_pipe = julia_eval_thread_.get_read_pipe();
  julia_eval_ = std::make_unique<conn::PipeConn>(
      read_pipe, julia_eval_recv_mem_.data(),
      julia_eval_send_mem_.data(), kCommBuffCapacity);
  julia_eval_conn_.type = PollConn::ConnType::compute;
  julia_eval_conn_.conn = julia_eval_.get();
  int ret = event_handler_.SetToReadOnly(&julia_eval_conn_);
  CHECK_EQ(ret, 0);
  julia_eval_thread_.Start();
  for (int i = 0; i < kThreadPoolSize; ++i) {
    auto read_pipe = thread_pool_.get_read_pipe(i);
    compute_[i] = std::make_unique<conn::PipeConn>(
        read_pipe, thread_pool_recv_mem_.data() + kCommBuffCapacity*i,
        thread_pool_send_mem_.data() + kCommBuffCapacity*i,
        kCommBuffCapacity);
    compute_conn_[i].type = PollConn::ConnType::compute;
    compute_conn_[i].conn = compute_[i].get();
    int ret = event_handler_.SetToReadOnly(&compute_conn_[i]);
    CHECK_EQ(ret, 0);
  }
  thread_pool_.Start();
}

void
Executor::ClearBeforeExit() {
  LOG(INFO) << __func__;
  julia_eval_thread_.Stop();
  thread_pool_.StopAll();

  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgPeerRecvStop>(
      &send_buff_);
  LOG(INFO) << "stopping peer recv thread";
  Send(&prt_poll_conn_, prt_pipe_conn_.get());
  send_buff_.clear_to_send();
  prt_runner_.join();
}

void
Executor::Send(PollConn* poll_conn_ptr, conn::SocketConn* sock_conn) {
  auto& send_buff = poll_conn_ptr->get_send_buff();
  if (send_buff.get_remaining_to_send_size() > 0
      || send_buff.get_remaining_next_to_send_size() > 0) {
    bool sent = sock_conn->sock.Send(&send_buff);
    while (!sent) {
      sent = sock_conn->sock.Send(&send_buff);
    }
    send_buff.clear_to_send();
  }
  bool sent = sock_conn->sock.Send(&send_buff_);

  if (!sent) {
    send_buff.CopyAndMoveNextToSend(&send_buff_);
    if (poll_conn_ptr->type == PollConn::ConnType::peer
        || poll_conn_ptr->type == PollConn::ConnType::server)
      event_handler_.SetToWriteOnly(poll_conn_ptr);
    else
      event_handler_.SetToReadWrite(poll_conn_ptr);
  }
  send_buff_.reset_sent_sizes();

}

void
Executor::Send(PollConn* poll_conn_ptr, conn::PipeConn* pipe_conn) {
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
    send_buff.CopyAndMoveNextToSend(&send_buff_);
    event_handler_.SetToReadWrite(poll_conn_ptr);
  }
  send_buff_.reset_sent_sizes();
}

void
Executor::EvalExpr() {
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());
  task::EvalExpr eval_expr_task;
  eval_expr_task.ParseFromString(task_str);
  eval_julia_expr_task_.serialized_expr = eval_expr_task.serialized_expr();
  eval_julia_expr_task_.module = static_cast<JuliaModule>(
      eval_expr_task.module());
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&eval_julia_expr_task_));
}

bool
Executor::CreateDistArray() {
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());

  task::CreateDistArray create_dist_array;
  create_dist_array.ParseFromString(task_str);

  int32_t id = create_dist_array.id();
  bool is_dense = create_dist_array.is_dense();
  dist_array_under_operation_ = id;
  type::PrimitiveType value_type
      = static_cast<type::PrimitiveType>(create_dist_array.value_type());
  size_t num_dims = create_dist_array.num_dims();
  auto parent_type = create_dist_array.parent_type();
  auto init_type = create_dist_array.init_type();
  std::string symbol = create_dist_array.symbol();
  DistArrayMeta *parent_dist_array_meta_ptr = nullptr;
  if (parent_type == task::DIST_ARRAY) {
    int32_t parent_id = create_dist_array.parent_id();
    auto &parent_meta = dist_arrays_.at(parent_id).GetMeta();
    parent_dist_array_meta_ptr = &parent_meta;
  }
  auto iter_pair = dist_arrays_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(id),
      std::forward_as_tuple(kConfig, value_type, kId,
                            num_dims, parent_type, init_type,
                            parent_dist_array_meta_ptr, is_dense,
                            symbol));
  if (kIsServer) {
    if (parent_type != task::TEXT_FILE)
      return true;
    else
      return false;
  }

  auto &dist_array = iter_pair.first->second;

  task::DistArrayMapType map_type = create_dist_array.map_type();
  bool map = (map_type != task::NO_MAP);
  bool flatten_results = create_dist_array.flatten_results();
  switch (parent_type) {
    case task::TEXT_FILE:
      {
        std::string file_path = create_dist_array.file_path();
        JuliaModule mapper_func_module
            = map ? static_cast<JuliaModule>(create_dist_array.mapper_func_module())
            : JuliaModule::kNone;
        std::string mapper_func_name
            = map ? create_dist_array.mapper_func_name()
            : std::string();
        exec_cpp_func_task_.result_buff.resize(sizeof(int64_t) * num_dims, 0);
        auto cpp_func = std::bind(&DistArray::LoadPartitionsFromTextFile,
                                  &dist_array, std::placeholders::_1,
                                  file_path, map_type, flatten_results,
                                  num_dims, mapper_func_module, mapper_func_name,
                                  &exec_cpp_func_task_.result_buff);
        exec_cpp_func_task_.func = cpp_func;
        exec_cpp_func_task_.label = TaskLabel::kLoadDistArrayFromTextFile;
        julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
      }
      break;
    case task::DIST_ARRAY:
      {
      }
      break;
    case task::INIT:
      {
        auto init_type = create_dist_array.init_type();
        JuliaModule mapper_func_module
            = map ? static_cast<JuliaModule>(create_dist_array.mapper_func_module())
            : JuliaModule::kNone;
        std::string mapper_func_name
            = map ? create_dist_array.mapper_func_name()
            : std::string();
        type::PrimitiveType random_init_type
            = static_cast<type::PrimitiveType>(create_dist_array.random_init_type());
        dist_array.SetDims(create_dist_array.dims().data(), num_dims);
        auto cpp_func = std::bind(&DistArray::RandomInit,
                                  &dist_array,
                                  std::placeholders::_1,
                                  init_type,
                                  map_type,
                                  mapper_func_module,
                                  mapper_func_name,
                                  random_init_type);
        exec_cpp_func_task_.func = cpp_func;
        exec_cpp_func_task_.label = TaskLabel::kRandomInitDistArray;
        julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
        dist_array_under_operation_ = id;
      }
      break;
    default:
      LOG(FATAL) << "unknown!" << static_cast<int>(parent_type);
  }
  return false;
}

void
Executor::DefineVar() {
  LOG(INFO) << "Executor " << __func__;
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());

  task::DefineVar define_var_task;
  define_var_task.ParseFromString(task_str);

  std::string var_name = define_var_task.var_name();
  std::string var_value = define_var_task.var_value();
  auto cpp_func = std::bind(
      JuliaEvaluator::StaticDefineVar,
      std::placeholders::_1,
      var_name,
      var_value);
  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kDefineVar;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
}

void
Executor::RepartitionDistArray() {
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());

  task::RepartitionDistArray repartition_dist_array_task;
  repartition_dist_array_task.ParseFromString(task_str);
  int32_t id = repartition_dist_array_task.id();
  LOG(INFO) << __func__ << " dist_array_id = " << id;
  dist_array_under_operation_ = id;

  int32_t partition_scheme = repartition_dist_array_task.partition_scheme();
  int32_t index_type = repartition_dist_array_task.index_type();

  std::string partition_func_name
      = repartition_dist_array_task.partition_func_name();
  auto &dist_array_to_repartition = dist_arrays_.at(id);
  auto &meta = dist_array_to_repartition.GetMeta();
  meta.SetPartitionScheme(static_cast<DistArrayPartitionScheme>(partition_scheme));
  meta.SetIndexType(static_cast<DistArrayIndexType>(index_type));

  auto cpp_func = std::bind(
      JuliaEvaluator::StaticComputeRepartition,
      std::placeholders::_1,
      partition_func_name,
      &dist_array_to_repartition);
  //julia_eval_result_ = repartition_ids;
  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kComputeRepartition;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
}

void
Executor::RepartitionDistArraySend(int32_t dist_array_id,
                                   DistArray *dist_array) {
  LOG(INFO) << __func__ << " dist_array_id = " << dist_array_id;
  auto send_buff = std::unordered_map<int32_t, std::pair<uint8_t*, size_t>>();
  send_buff.clear();
  dist_array->RepartitionSerializeAndClear(&send_buff);
  for (size_t recv_id = 0; recv_id < kNumExecutors; recv_id++) {
    if (recv_id == kId) continue;
    auto iter = send_buff.find(recv_id);
    if (iter == send_buff.end()) {
      message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRepartitionDistArrayData>(
          &send_buff_, dist_array_id, 0);
      Send(&peer_conn_[recv_id], peer_[recv_id].get());
      send_buff_.clear_to_send();
      send_buff_.reset_sent_sizes();
    } else {
      auto &buff = iter->second;
      size_t send_size = buff.second;
      uint8_t* send_data = buff.first;
      message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRepartitionDistArrayData>(
          &send_buff_, dist_array_id, send_size);
      send_buff_.set_next_to_send(send_data, send_size, true);
      Send(&peer_conn_[recv_id], peer_[recv_id].get());
      send_buff_.clear_to_send();
      send_buff_.reset_sent_sizes();
    }
  }
}

void
Executor::DefineJuliaDistArray(DistArray *dist_array) {
  auto &meta = dist_array->GetMeta();
  auto &dims = meta.GetDims();
  auto &symbol = meta.GetSymbol();
  auto value_type = dist_array->GetValueType();
  void *access_ptr = dist_array->GetAccessPtr();
  auto is_dense = meta.IsDense();

  auto cpp_func = std::bind(
      JuliaEvaluator::StaticDefineDistArray,
      std::placeholders::_1,
      &symbol,
      value_type,
      &dims,
      is_dense,
      access_ptr);

  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kDefineJuliaDistArray;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
}

}
}
