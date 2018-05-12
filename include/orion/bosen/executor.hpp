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
#include <orion/bosen/julia_thread_requester.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/byte_buffer.hpp>
#include <orion/bosen/task.pb.h>
#include <orion/bosen/recv_arbitrary_bytes.hpp>
#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/driver_message.hpp>
#include <orion/bosen/key.hpp>
#include <orion/bosen/send_data_buffer.hpp>
#include <orion/bosen/abstract_exec_for_loop.hpp>
#include <orion/bosen/exec_for_loop_1d.hpp>
#include <orion/bosen/exec_for_loop_space_time_unordered.hpp>
#include <orion/bosen/worker.h>
#include <orion/bosen/dist_array_value_request_meta.hpp>
#include <orion/bosen/dist_array_buffer_info.hpp>
#include <orion/bosen/server_exec_for_loop.hpp>

namespace orion {
namespace bosen {

class Executor {
 private:
  struct PollConn {
    enum class ConnType {
      listen = 0,
        master = 1,
        compute = 2,
        executor = 3,
        peer_recv_thr = 4,
        server = 5
    };
    void* conn;
    ConnType type;

    bool Receive() {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::executor
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
          || type == ConnType::executor
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
          || type == ConnType::executor
          || type == ConnType::server) {
        return reinterpret_cast<conn::SocketConn*>(conn)->recv_buff;
      } else {
        return reinterpret_cast<conn::PipeConn*>(conn)->recv_buff;
      }
    }

    conn::SendBuffer& get_send_buff() {
      if (type == ConnType::listen
          || type == ConnType::master
          || type == ConnType::executor
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
          || type == ConnType::executor
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
          || type == ConnType::executor
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
      kExecutorAck = 10,
      kRepartitionDistArray = 11,
      kRepartitionDistArraySend = 12,
      kCreateExecForLoop = 13,
      kDefineJuliaDistArray = 14,
      kGetAccumulatorValue = 15,
      kRepartitionDistArrayAck = 16,
      kCreateDistArrayBuffer = 17,
      kDefineJuliaDistArrayBuffer = 18,
      kCreateDistArrayBufferAck = 19,
      kTextFileLoadDone = 20,
      kRepartitionDistArraySerialize = 21,
      kUpdateDistArrayIndex = 22,
      kDeleteAllDistArrays = 23
            };

  static const int32_t kPortSpan = 100;
  const size_t kCommBuffCapacity;
  const size_t kNumExecutors;
  const size_t kNumServers;
  const size_t kNumLocalExecutors;
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  const std::string kListenIp;
  const uint16_t kListenPort;
  const size_t kThreadPoolSize;
  const int32_t kExecutorId;
  const int32_t kServerId;
  const int32_t kId;
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
  Blob executor_server_send_mem_;
  Blob executor_server_recv_mem_;
  std::vector<std::unique_ptr<conn::SocketConn>> executor_;
  std::vector<std::unique_ptr<conn::SocketConn>> server_;
  std::vector<PollConn> executor_conn_;
  std::vector<PollConn> server_conn_;
  std::vector<conn::Socket> executor_server_socks_;

  Blob thread_pool_recv_mem_;
  Blob thread_pool_send_mem_;
  std::vector<std::unique_ptr<conn::PipeConn>> compute_;
  std::vector<PollConn> compute_conn_;
  ThreadPool thread_pool_;

  Blob julia_eval_recv_mem_;
  Blob julia_eval_send_mem_;
  std::unique_ptr<conn::PipeConn> julia_eval_pipe_conn_;
  PollConn julia_eval_conn_;
  JuliaEvalThread julia_eval_thread_;
  JuliaThreadRequester* julia_requester_ {nullptr} ;

  ExecCppFuncTask exec_cpp_func_task_;
  EvalJuliaExprTask eval_julia_expr_task_;
  JuliaTask *julia_task_ptr_ {nullptr};

  ByteBuffer master_recv_byte_buff_;
  std::unordered_map<int32_t, DistArray> dist_arrays_;
  std::unordered_map<int32_t, DistArray> dist_array_buffers_;

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
  bool connected_to_peers_ { false };

  ExecutorSendBufferMap repartition_send_buffer_;
  std::unique_ptr<AbstractExecForLoop> exec_for_loop_;
  std::unique_ptr<ServerExecForLoop> server_exec_for_loop_;

  bool repartition_recv_ { false };
  DistArrayValueRequestMeta dist_array_value_request_meta_;

  std::unordered_map<int32_t, DistArrayBufferInfo> dist_array_buffer_info_map_;

 public:
  Executor(const Config& config, int32_t local_index, bool is_server);
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
  void TextFileLoadDone();
  void ParseDistArrayTextBuffer(DistArray& dist_array,
                                const std::vector<size_t> &line_number_start);
  bool RepartitionDistArray();
  bool RepartitionDistArraySerialize(int32_t dist_array_id, DistArray *dist_array);
  void RepartitionDistArraySend();
  void UpdateDistArrayIndex();
  void DefineJuliaDistArray();
  void DefineJuliaDistArrayBuffer();
  void CreateExecForLoop();
  void CreateServerExecForLoop();
  void CheckAndExecuteForLoop(bool next_partition);
  bool CheckAndSerializeGlobalIndexedDistArrays();
  bool CheckAndSerializeDistArrayTimePartitions();
  void SendGlobalIndexedDistArrays();
  void SendPipelinedTimePartitions();
  void SendPredCompletion();
  void ExecForLoopClear();
  void ExecForLoopAck();
  void ServerExecForLoopAck();
  void RequestExecForLoopGlobalIndexedDistArrays();
  void RequestExecForLoopPipelinedTimePartitions();
  bool CheckAndRequestExecForLoopPredecesorCompletion();
  void SerializeAndSendExecForLoopPrefetchRequests();
  void ReplyDistArrayValues();
  void CacheGlobalIndexedDistArrayValues(PeerRecvGlobalIndexedDistArrayDataBuffer **buff_vec,
                                         size_t num_buffs);

  void GetAccumulatorValue();
  void ReplyGetAccumulatorValue();
  void SetDistArrayBufferInfo();
  void DeleteDistArrayBufferInfo(int32_t dist_array_buffer_id);
};

Executor::Executor(const Config& config,
                   int32_t local_index,
                   bool is_server):
    kCommBuffCapacity(config.kCommBuffCapacity),
    kNumExecutors(config.kNumExecutors),
    kNumServers(config.kNumServers),
    kNumLocalExecutors(config.kNumExecutorsPerWorker),
    kMasterIp(config.kMasterIp),
    kMasterPort(config.kMasterPort),
    kListenIp(config.kWorkerIp),
    kListenPort(config.kWorkerPort + local_index * kPortSpan),
    kThreadPoolSize(config.kExecutorThreadPoolSize),
    kExecutorId(config.kWorkerId * config.kNumExecutorsPerWorker + local_index),
    kServerId(config.kWorkerId * config.kNumServersPerWorker + (local_index - config.kNumExecutorsPerWorker)),
    kId(is_server ? kNumExecutors + kServerId : kExecutorId),
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
    executor_server_send_mem_(config.kCommBuffCapacity * (config.kNumExecutors + config.kNumServers)),
    executor_server_recv_mem_(config.kCommBuffCapacity * (config.kNumExecutors + config.kNumServers)),
    executor_(config.kNumExecutors),
    server_(config.kNumServers),
    executor_conn_(config.kNumExecutors),
    server_conn_(config.kNumServers),
    executor_server_socks_(config.kNumExecutors + config.kNumServers),
    thread_pool_recv_mem_(kCommBuffCapacity*kThreadPoolSize),
    thread_pool_send_mem_(kCommBuffCapacity*kThreadPoolSize),
    compute_(kThreadPoolSize),
    compute_conn_(kThreadPoolSize),
    thread_pool_(config.kExecutorThreadPoolSize, config.kCommBuffCapacity),
    julia_eval_recv_mem_(kCommBuffCapacity),
    julia_eval_send_mem_(kCommBuffCapacity),
    julia_eval_thread_(kCommBuffCapacity,
                       config.kOrionHome,
                       kNumServers,
                       kNumExecutors,
                       kIsServer ? kServerId : kExecutorId,
                       !kIsServer),
    julia_requester_(julia_eval_thread_.GetJuliaThreadRequester()),
    host_info_(kNumExecutors + config.kNumServers),
    prt_send_mem_(kCommBuffCapacity),
    prt_recv_mem_(kCommBuffCapacity) {
  set_julia_thread_requester(julia_requester_);
}

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
    send_buff_.reset_sent_sizes();
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
  ret = listen_.sock.Listen(kNumExecutors + kNumServers);
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
          kNumExecutors,
          kConfig.kNumServers,
          executor_server_socks_,
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

  executor_server_socks_[num_connected_peers_] = accepted;
  num_connected_peers_++;

  if (kIsServer) {
    if (num_connected_peers_ == kNumExecutors + kServerId
        && connected_to_peers_) {
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
      || type == PollConn::ConnType::executor
      || type == PollConn::ConnType::server) {
    int ret = event_handler_.Remove(poll_conn_ptr);
    CHECK_EQ(ret, 0);
  } else {
    int ret = event_handler_.Remove(poll_conn_ptr);
    CHECK_EQ(ret, 0);
    auto* conn = reinterpret_cast<conn::SocketConn*>(poll_conn_ptr->conn);
    conn->sock.Close();
    action_ = Action::kDeleteAllDistArrays;
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
    if (poll_conn_ptr->type == PollConn::ConnType::executor
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
  } else if (poll_conn_ptr->type == PollConn::ConnType::executor
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
          ConnectToPeers();
          connected_to_peers_ = true;
          // if I don't expect to receive connections,
          // go ahead initialize peer recv threads
          if (!kIsServer && kExecutorId == 0) {
            action_ = Action::kAckConnectToPeers;
            InitPeerRecvThread();
          } else if ((kIsServer && num_connected_peers_ == (kServerId + kNumExecutors))
                     || (!kIsServer && num_connected_peers_ == kExecutorId)) {
            // I might receive connections before told so by driver
            CHECK(peer_recv_thr_.get() == nullptr);
            action_ = Action::kNone;
            InitPeerRecvThread();
          } else action_ = Action::kNone;
        }
        break;
      case Action::kAckConnectToPeers:
        {
          int event_handler_ret = event_handler_.Remove(&prt_poll_conn_);
          CHECK_EQ(event_handler_ret, 0) << event_handler_ret;
          ret |= EventHandler<PollConn>::kExit;
          message::Helper::CreateMsg<message::ExecutorConnectToPeersAck>(&send_buff_);
          Send(&master_poll_conn_, &master_);
          send_buff_.clear_to_send();
          send_buff_.reset_sent_sizes();
          action_ = Action::kNone;
        }
        break;
      case Action::kEvalExpr:
        {
          EvalExpr();
          action_ = Action::kNone;
        }
        break;
      case Action::kCreateDistArray:
        {
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
            if (julia_task_ptr_ == &eval_julia_expr_task_) {
              result_size = eval_julia_expr_task_.result_buff.size();
              result_mem = eval_julia_expr_task_.result_buff.data();
            } else if (julia_task_ptr_ == &exec_cpp_func_task_) {
            } else {
              LOG(FATAL) << "error!";
            }
          }
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
      case Action::kTextFileLoadDone:
        {
          TextFileLoadDone();
          action_ = Action::kNone;
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
      case Action::kCreateDistArrayBufferAck:
        {
          message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgCreateDistArrayBufferAck>(
              &send_buff_, dist_array_under_operation_);
          Send(&master_poll_conn_, &master_);
          send_buff_.clear_to_send();
          send_buff_.reset_sent_sizes();
          action_ = Action::kNone;
        }
        break;
      case Action::kDefineJuliaDistArray:
        {
          DefineJuliaDistArray();
          action_ = Action::kNone;
        }
        break;
      case Action::kDefineJuliaDistArrayBuffer:
        {
          DefineJuliaDistArrayBuffer();
          action_ = Action::kNone;
        }
        break;
      case Action::kRepartitionDistArray:
        {
          bool repartition_mine = RepartitionDistArray();
          if (!repartition_mine && !repartition_recv_)
            action_ = Action::kRepartitionDistArrayAck;
          else {
            if (!repartition_mine && repartition_recv_) {
              event_handler_.SetToReadOnly(&prt_poll_conn_);
            }
            action_ = Action::kNone;
          }
        }
        break;
      case Action::kUpdateDistArrayIndex:
        {
          UpdateDistArrayIndex();
        }
        break;
      case Action::kRepartitionDistArraySerialize:
        {
          auto &dist_array = dist_arrays_.at(dist_array_under_operation_);
          bool serialized = RepartitionDistArraySerialize(dist_array_under_operation_,
                                                          &dist_array);
          if (!serialized && !repartition_recv_) {
            action_ = Action::kRepartitionDistArrayAck;
          } else {
            if (!serialized && repartition_recv_) {
              event_handler_.SetToReadOnly(&prt_poll_conn_);
            }
            action_ = Action::kNone;
          }
        }
        break;
      case Action::kRepartitionDistArraySend:
        {
          RepartitionDistArraySend();
          if (repartition_recv_)
            event_handler_.SetToReadOnly(&prt_poll_conn_);
          if (!repartition_recv_) {
            action_ = Action::kRepartitionDistArrayAck;
          } else {
            action_ = Action::kNone;
          }
        }
        break;
      case Action::kCreateExecForLoop:
        {
          if (kIsServer) {
            CreateServerExecForLoop();
            event_handler_.SetToReadOnly(&prt_poll_conn_);
          } else {
            CreateExecForLoop();
          }
          action_ = Action::kNone;
        }
        break;
      case Action::kGetAccumulatorValue:
        {
          GetAccumulatorValue();
          action_ = Action::kNone;
        }
        break;
      case Action::kRepartitionDistArrayAck:
        {
          int32_t dist_array_id = dist_array_under_operation_;
          auto &dist_array = dist_arrays_.at(dist_array_id);
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
        }
        break;
      case Action::kDeleteAllDistArrays:
        {
          auto cpp_func = std::bind(
              JuliaEvaluator::DeleteAllDistArrays,
              &dist_arrays_,
              &dist_array_buffers_);
          exec_cpp_func_task_.func = cpp_func;
          exec_cpp_func_task_.label = TaskLabel::kDeleteAllDistArrays;
          julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
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
        action_ = Action::kDeleteAllDistArrays;
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
        size_t expected_size = (kNumExecutors + kNumServers) * sizeof(int);
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
            uint8_t *recv_mem = executor_server_recv_mem_.data()
                                + kCommBuffCapacity * i;

            uint8_t *send_mem = executor_server_send_mem_.data()
                                + kCommBuffCapacity * i;

            PollConn *curr_poll_conn = nullptr;
            if (i < kNumExecutors) {
              curr_poll_conn = &(executor_conn_[i]);
            } else {
              curr_poll_conn = &(server_conn_[i - kNumExecutors]);
            }

            auto *sock_conn = new conn::SocketConn(
                sock, recv_mem, send_mem, kCommBuffCapacity);

            curr_poll_conn->conn = sock_conn;
            curr_poll_conn->type = (i < kNumExecutors) ? PollConn::ConnType::executor : PollConn::ConnType::server;
            if (i < kNumExecutors) {
              executor_[i].reset(sock_conn);
            } else {
              server_[i - kNumExecutors].reset(sock_conn);
            }
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
        auto *repartition_recv_buff = reinterpret_cast<PeerRecvRepartitionDistArrayDataBuffer*>(
            msg->data_buff);
        int32_t dist_array_id = repartition_recv_buff->dist_array_id;
        auto &dist_array = dist_arrays_.at(dist_array_id);

        auto cpp_func = std::bind(
            &DistArray::RepartitionDeserialize,
            &dist_array,
            repartition_recv_buff);

        exec_cpp_func_task_.func = cpp_func;
        exec_cpp_func_task_.label = TaskLabel::kRepartitionDeserialize;
        julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));

        dist_array_under_operation_ = dist_array_id;

        int event_handler_ret = event_handler_.Remove(&prt_poll_conn_);
        CHECK_EQ(event_handler_ret, 0) << event_handler_ret;
        ret = EventHandler<PollConn>::kClearOneMsg;
        ret |= EventHandler<PollConn>::kExit;
        action_ = Action::kNone;
      }
      break;
    case message::ExecuteMsgType::kReplyExecForLoopPipelinedTimePartitions:
      {
        LOG(INFO) <<"Got ReplyExecForLoopPipelinedTimePartitions";
        action_ = Action::kNone;
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgReplyExecForLoopPipelinedTimePartitions>(recv_buff);

        auto **buff_vec = reinterpret_cast<
          PeerRecvPipelinedTimePartitionsBuffer**>(msg->data_buff_vec);
        size_t num_buffs = msg->num_data_buffs;

        auto cpp_func = std::bind(
            &AbstractExecForLoop::DeserializePipelinedTimePartitionsBuffVec,
            exec_for_loop_.get(),
            buff_vec,
            num_buffs);
        exec_cpp_func_task_.func = cpp_func;
        exec_cpp_func_task_.label = TaskLabel::kDeserializeDistArrayTimePartitions;
        julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));

        int event_handler_ret = event_handler_.Remove(&prt_poll_conn_);
        CHECK_EQ(event_handler_ret, 0) << event_handler_ret;
        ret = EventHandler<PollConn>::kClearOneMsg;
        ret |= EventHandler<PollConn>::kExit;
      }
      break;
    case message::ExecuteMsgType::kReplyExecForLoopPredecessorCompletion:
      {
        LOG(INFO) <<"Got ReplyExecForLoopPredcessorCompletion";
        CHECK(exec_for_loop_.get());
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgReplyExecForLoopPredecessorCompletion>(recv_buff);
        auto **buff_vec = reinterpret_cast<
          PeerRecvPipelinedTimePartitionsBuffer**>(msg->data_buff_vec);
        size_t num_buffs = msg->num_data_buffs;
        if (buff_vec != nullptr) {
          auto cpp_func = std::bind(
              &AbstractExecForLoop::DeserializePipelinedTimePartitionsBuffVec,
              exec_for_loop_.get(),
              buff_vec,
              num_buffs);
          exec_cpp_func_task_.func = cpp_func;
          exec_cpp_func_task_.label = TaskLabel::kDeserializeDistArrayTimePartitionsPredCompletion;
          julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
        } else {
          ExecForLoopClear();
        }
        int event_handler_ret = event_handler_.Remove(&prt_poll_conn_);
        CHECK_EQ(event_handler_ret, 0) << event_handler_ret;
        ret = EventHandler<PollConn>::kClearOneMsg;
        ret |= EventHandler<PollConn>::kExit;
      }
      break;
    case message::ExecuteMsgType::kRequestDistArrayValue:
      {
        int event_handler_ret = event_handler_.Remove(&prt_poll_conn_);
        CHECK_EQ(event_handler_ret, 0) << event_handler_ret;
        ret = EventHandler<PollConn>::kClearOneMsg;
        ret |= EventHandler<PollConn>::kExit;
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgRequestDistArrayValue>(recv_buff);
        int32_t dist_array_id = msg->dist_array_id;
        int64_t key = msg->key;
        dist_array_value_request_meta_.requester_id = msg->requester_id;
        dist_array_value_request_meta_.is_requester_executor = msg->is_requester_executor;

        auto dist_array_iter = dist_arrays_.find(dist_array_id);
        CHECK(dist_array_iter != dist_arrays_.end());
        auto *dist_array_ptr = &dist_array_iter->second;

        auto cpp_func = std::bind(
            JuliaEvaluator::GetAndSerializeValue,
            dist_array_ptr,
            key,
            &exec_cpp_func_task_.result_buff);
        exec_cpp_func_task_.func = cpp_func;
        exec_cpp_func_task_.label = TaskLabel::kGetAndSerializeDistArrayValues;
        julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
      }
      break;
    case message::ExecuteMsgType::kRequestDistArrayValues:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgRequestDistArrayValues>(recv_buff);
        dist_array_value_request_meta_.requester_id = msg->requester_id;
        dist_array_value_request_meta_.is_requester_executor = msg->is_requester_executor;

        size_t expected_size = msg->request_size;
        auto &pipe = reinterpret_cast<conn::PipeConn*>(prt_poll_conn_.conn)->pipe;
        bool received_next_msg =
            ReceiveArbitraryBytes(pipe, &recv_buff,
                                  &prt_recv_byte_buff_,
                                  expected_size);
        if (received_next_msg) {
          int event_handler_ret = event_handler_.Remove(&prt_poll_conn_);
          CHECK_EQ(event_handler_ret, 0) << event_handler_ret;
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          ret |= EventHandler<PollConn>::kExit;
          const auto* request = prt_recv_byte_buff_.GetBytes();
          auto cpp_func = std::bind(
              JuliaEvaluator::GetAndSerializeValues,
              &dist_arrays_,
              request,
              &exec_cpp_func_task_.result_buff);
          exec_cpp_func_task_.func = cpp_func;
          exec_cpp_func_task_.label = TaskLabel::kGetAndSerializeDistArrayValues;
          julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
        } else {
          ret = EventHandler<PollConn>::kNoAction;
        }
      }
      break;
    case message::ExecuteMsgType::kExecForLoopDistArrayBufferDataPtr:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgExecForLoopDistArrayBufferDataPtr>(recv_buff);
        auto *bytes = msg->bytes;
        int event_handler_ret = event_handler_.Remove(&prt_poll_conn_);
        CHECK_EQ(event_handler_ret, 0) << event_handler_ret;
        ret = EventHandler<PollConn>::kClearOneMsg;
        ret |= EventHandler<PollConn>::kExit;
        auto cpp_func = std::bind(
            &ServerExecForLoop::DeserializeAndApplyDistArrayBuffers,
            server_exec_for_loop_.get(),
            bytes);
        exec_cpp_func_task_.func = cpp_func;
        exec_cpp_func_task_.label = TaskLabel::kExecForLoopApplyDistArrayBufferData;
        julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
      }
      break;
    case message::ExecuteMsgType::kExecForLoopDistArrayCacheDataPtr:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgExecForLoopDistArrayCacheDataPtr>(recv_buff);
        auto *bytes = msg->bytes;
        int event_handler_ret = event_handler_.Remove(&prt_poll_conn_);
        CHECK_EQ(event_handler_ret, 0) << event_handler_ret;
        ret = EventHandler<PollConn>::kClearOneMsg;
        ret |= EventHandler<PollConn>::kExit;
        auto cpp_func = std::bind(
            &ServerExecForLoop::DeserializeAndApplyDistArrayCaches,
            server_exec_for_loop_.get(),
            bytes);
        exec_cpp_func_task_.func = cpp_func;
        exec_cpp_func_task_.label = TaskLabel::kExecForLoopApplyDistArrayCacheData;
        julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
      }
      break;
    case message::ExecuteMsgType::kReplyExecForLoopGlobalIndexedDistArrayData:
      {
        action_ = Action::kNone;
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgReplyExecForLoopGlobalIndexedDistArrayData>(recv_buff);

        auto **buff_vec = reinterpret_cast<
          PeerRecvGlobalIndexedDistArrayDataBuffer**>(msg->data_buff_vec);
        size_t num_buffs = msg->num_data_buffs;

        CacheGlobalIndexedDistArrayValues(buff_vec, num_buffs);
        int event_handler_ret = event_handler_.Remove(&prt_poll_conn_);
        CHECK_EQ(event_handler_ret, 0) << event_handler_ret;
        ret = EventHandler<PollConn>::kClearOneMsg;
        ret |= EventHandler<PollConn>::kExit;
      }
      break;
    case message::ExecuteMsgType::kExecForLoopDone:
      {
        CHECK(kIsServer);
        bool clear = server_exec_for_loop_->NotifyExecForLoopDone();
        if (clear) {
          server_exec_for_loop_.reset();
          int event_handler_ret = event_handler_.Remove(&prt_poll_conn_);
          CHECK_EQ(event_handler_ret, 0) << event_handler_ret;
          ServerExecForLoopAck();
        }
        ret = EventHandler<PollConn>::kClearOneMsg;
        ret |= EventHandler<PollConn>::kExit;
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
        julia_task_ptr_ = msg->task;
        if (auto *task = dynamic_cast<ExecCppFuncTask*>(msg->task)) {
          switch (task->label) {
            case TaskLabel::kNone:
              {
                action_ = Action::kExecutorAck;
              }
              break;
            case TaskLabel::kLoadDistArrayFromTextFile:
              {
                action_ = Action::kTextFileLoadDone;
              }
              break;
            case TaskLabel::kParseDistArrayTextBuffer:
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
                action_ = Action::kRepartitionDistArraySerialize;
              }
              break;
            case TaskLabel::kRepartitionSerialize:
              {
                action_ = Action::kRepartitionDistArraySend;
              }
              break;
            case TaskLabel::kRepartitionDeserialize:
              {
                action_ = Action::kRepartitionDistArrayAck;
              }
              break;
            case TaskLabel::kInitDistArray:
              {
                action_ = Action::kCreateDistArrayAck;
              }
              break;
              case TaskLabel::kMapDistArray:
              {
                action_ = Action::kCreateDistArrayAck;
              }
              break;
            case TaskLabel::kDefineJuliaDistArray:
              {
                action_ = Action::kCreateDistArray;
              }
              break;
            case TaskLabel::kDefineJuliaDistArrayBuffer:
              {
                action_ = Action::kCreateDistArrayBufferAck;
              }
              break;
            case TaskLabel::kExecForLoopPartition:
              {
                action_ = Action::kNone;
                bool serialize_dist_array_time_partitions = false;
                bool serialize_global_indexed_dist_arrays
                    = CheckAndSerializeGlobalIndexedDistArrays();
                if (!serialize_global_indexed_dist_arrays) {
                  serialize_dist_array_time_partitions
                      = CheckAndSerializeDistArrayTimePartitions();
                }
                if (!serialize_global_indexed_dist_arrays
                    && !serialize_dist_array_time_partitions) {
                  CheckAndExecuteForLoop(true);
                }
              }
              break;
            case TaskLabel::kSerializeGlobalIndexedDistArrays:
              {
                SendGlobalIndexedDistArrays();
                bool serialize_dist_array_time_partitions
                      = CheckAndSerializeDistArrayTimePartitions();
                action_ = Action::kNone;
                if (!serialize_dist_array_time_partitions) {
                  CheckAndExecuteForLoop(true);
                }
              }
              break;
            case TaskLabel::kSerializeDistArrayTimePartitions:
              {
                action_ = Action::kNone;
                SendPipelinedTimePartitions();
                CheckAndExecuteForLoop(true);
              }
              break;
            case TaskLabel::kDeserializeDistArrayTimePartitions:
              {
                action_ = Action::kNone;
                CheckAndExecuteForLoop(false);
              }
              break;
            case TaskLabel::kDeserializeDistArrayTimePartitionsPredCompletion:
              {
                action_ = Action::kNone;
                ExecForLoopClear();
              }
              break;
            case TaskLabel::kGetAccumulatorValue:
              {
                ReplyGetAccumulatorValue();
              }
              break;
            case TaskLabel::kSetDistArrayDims:
              {
                action_ = Action::kCreateDistArrayAck;
              }
              break;
            case TaskLabel::kComputePrefetchIndices:
              {
                SerializeAndSendExecForLoopPrefetchRequests();
                CheckAndExecuteForLoop(false);
              }
              break;
            case TaskLabel::kGetAndSerializeDistArrayValues:
              {
                ReplyDistArrayValues();
                event_handler_.SetToReadOnly(&prt_poll_conn_);
              }
              break;
            case TaskLabel::kCachePrefetchDistArrayValues:
              {
                CheckAndExecuteForLoop(false);
              }
              break;
            case TaskLabel::kUpdateDistArrayIndex:
              {
                action_ = Action::kExecutorAck;
              }
              break;
            case TaskLabel::kExecForLoopApplyDistArrayBufferData:
              {
                action_ = Action::kNone;
                event_handler_.SetToReadOnly(&prt_poll_conn_);
              }
              break;
            case TaskLabel::kExecForLoopApplyDistArrayCacheData:
              {
                action_ = Action::kNone;
                event_handler_.SetToReadOnly(&prt_poll_conn_);
              }
              break;
            case TaskLabel::kExecForLoopInit:
              {
                CheckAndExecuteForLoop(false);
                action_ = Action::kNone;
              }
              break;
            case TaskLabel::kExecForLoopClear:
              {
                ExecForLoopAck();
                action_ = Action::kNone;
              }
              break;
            case TaskLabel::kDeleteAllDistArrays:
              {
                LOG(INFO) << "DeleteAllDistArrays done";
                action_ = Action::kExit;
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
    case message::ExecuteMsgType::kRequestDistArrayValue:
      {
        auto *msg = message::Helper::get_msg<
          message::ExecuteMsgRequestDistArrayValue>(recv_buff);
        int64_t key = msg->key;
        int32_t server_id = key % kNumServers;
        send_buff_.Copy(recv_buff);
        Send(&server_conn_[server_id], server_[server_id].get());
        send_buff_.clear_to_send();
        send_buff_.reset_sent_sizes();
        event_handler_.SetToReadOnly(&prt_poll_conn_);
        RequestExecForLoopGlobalIndexedDistArrays();
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
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgCreateDistArray>(recv_buff);
        size_t expected_size = msg->task_size;
        bool received_next_msg =
            ReceiveArbitraryBytes(master_.sock, &recv_buff,
                                  &master_recv_byte_buff_,
                                  expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kDefineJuliaDistArray;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::ExecuteMsgType::kPartitionNumLines:
      {
        auto *msg = message::ExecuteMsgHelper::get_msg<
          message::ExecuteMsgPartitionNumLines>(recv_buff);
        size_t num_partitions = msg->num_partitions;
        if (num_partitions > 0) {
          size_t expected_size = num_partitions * sizeof(size_t);
          bool received_next_msg =
              ReceiveArbitraryBytes(master_.sock, &recv_buff,
                                    &master_recv_byte_buff_,
                                    expected_size);
          if (received_next_msg) {
            std::vector<size_t> line_number_start(num_partitions);
            memcpy(line_number_start.data(),
                   master_recv_byte_buff_.GetBytes(),
                   expected_size);
            int64_t dist_array_id = msg->dist_array_id;
            auto &dist_array = dist_arrays_.at(dist_array_id);
            ParseDistArrayTextBuffer(dist_array, line_number_start);
            ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          } else ret = EventHandler<PollConn>::kNoAction;
          action_ = Action::kNone;
        } else {
          exec_cpp_func_task_.result_buff.clear();
          action_ = Action::kTextFileLoadAck;
          ret = EventHandler<PollConn>::kClearOneMsg;
        }
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

          auto &dist_array_meta = dist_array.GetMeta();
          const auto &dist_array_symbol = dist_array_meta.GetSymbol();
          const auto &dist_array_dims = dist_array.GetDims();

          auto cpp_func = std::bind(
              JuliaEvaluator::SetDistArrayDims,
              dist_array_symbol,
              dist_array_dims);

          exec_cpp_func_task_.func = cpp_func;
          exec_cpp_func_task_.label = TaskLabel::kSetDistArrayDims;
          julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));

          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kNone;
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
        dist_array_meta.SetMaxPartitionIds(max_ids, num_dims);
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
        } else {
          ret = EventHandler<PollConn>::kNoAction;
          action_ = Action::kNone;
        }
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
    case message::DriverMsgType::kUpdateDistArrayIndex:
      {
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgUpdateDistArrayIndex>(recv_buff);
        size_t expected_size = msg->task_size;
        bool received_next_msg =
            ReceiveArbitraryBytes(master_.sock,
                                  &recv_buff,
                                  &master_recv_byte_buff_,
                                  expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kUpdateDistArrayIndex;
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
    case message::DriverMsgType::kGetAccumulatorValue:
      {
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgGetAccumulatorValue>(recv_buff);
        size_t expected_size = msg->task_size;
        bool received_next_msg =
            ReceiveArbitraryBytes(master_.sock,
                                  &recv_buff,
                                  &master_recv_byte_buff_,
                                  expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kGetAccumulatorValue;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::DriverMsgType::kCreateDistArrayBuffer:
      {
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgCreateDistArrayBuffer>(recv_buff);
        size_t expected_size = msg->task_size;
        bool received_next_msg =
            ReceiveArbitraryBytes(master_.sock, &recv_buff,
                                  &master_recv_byte_buff_,
                                  expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kDefineJuliaDistArrayBuffer;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::DriverMsgType::kSetDistArrayBufferInfo:
      {
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgSetDistArrayBufferInfo>(recv_buff);
        size_t expected_size = msg->info_size;
        bool received_next_msg
            = ReceiveArbitraryBytes(master_.sock, &recv_buff,
                                    &master_recv_byte_buff_,
                                    expected_size);
        if (received_next_msg) {
          SetDistArrayBufferInfo();
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          action_ = Action::kExecutorAck;
        } else ret = EventHandler<PollConn>::kNoAction;
      }
      break;
    case message::DriverMsgType::kDeleteDistArrayBufferInfo:
      {
        auto *msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgDeleteDistArrayBufferInfo>(recv_buff);
        int32_t dist_array_buffer_id = msg->dist_array_buffer_id;
        DeleteDistArrayBufferInfo(dist_array_buffer_id);
        action_ = Action::kExecutorAck;
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
  if (kIsServer) {
    message::Helper::CreateMsg<
      message::ServerIdentity>(&send_buff_, kServerId, host_info);
  } else {
    message::Helper::CreateMsg<
      message::ExecutorIdentity>(&send_buff_, kExecutorId, host_info);
  }

  if (kIsServer) {
    for (int32_t i = kNumExecutors + kServerId + 1; i < kNumExecutors + kConfig.kNumServers; i++) {
      LOG(INFO) << __func__ << " " << i;
      uint32_t ip = host_info_[i].ip;
      uint16_t port = host_info_[i].port;
      int32_t server_id = i - kNumExecutors;
      conn::Socket &server_sock = executor_server_socks_[i];
      ret = server_sock.Connect(ip, port);
      CHECK(ret == 0) << "executor failed connecting to server " << server_id
                      << " ip = " << ip << " port = " << port;
      server_[server_id].reset(new conn::SocketConn(server_sock,
                                                    executor_server_recv_mem_.data() + kCommBuffCapacity * i,
                                                    executor_server_send_mem_.data() + kCommBuffCapacity * i,
                                                    kCommBuffCapacity));
      server_conn_[server_id].conn = server_[server_id].get();
      server_conn_[server_id].type = PollConn::ConnType::server;
      Send(&server_conn_[server_id], server_[server_id].get());
    }
  } else {
    for (int32_t i = kNumExecutors; i < kNumExecutors + kConfig.kNumServers; i++) {
      LOG(INFO) << __func__ << " " << i;
      uint32_t ip = host_info_[i].ip;
      uint16_t port = host_info_[i].port;
      int32_t server_id = i - kNumExecutors;
      conn::Socket &server_sock = executor_server_socks_[i];
      ret = server_sock.Connect(ip, port);
      CHECK(ret == 0) << "executor failed connecting to server " << server_id
                      << " ip = " << ip << " port = " << port;
      server_[server_id].reset(new conn::SocketConn(server_sock,
                                                    executor_server_recv_mem_.data() + kCommBuffCapacity * i,
                                                    executor_server_send_mem_.data() + kCommBuffCapacity * i,
                                                    kCommBuffCapacity));
      server_conn_[server_id].conn = server_[server_id].get();
      server_conn_[server_id].type = PollConn::ConnType::server;
      Send(&server_conn_[server_id], server_[server_id].get());
    }

    for (int32_t i = kExecutorId + 1; i < kNumExecutors; i++) {
      uint32_t ip = host_info_[i].ip;
      uint16_t port = host_info_[i].port;
      conn::Socket &executor_sock = executor_server_socks_[i];
      ret = executor_sock.Connect(ip, port);
      CHECK(ret == 0) << "executor failed connecting to peer " << i
                      << " ip = " << ip << " port = " << port;
      executor_[i].reset(new conn::SocketConn(executor_sock,
                                              executor_server_recv_mem_.data() + kCommBuffCapacity * i,
                                              executor_server_send_mem_.data() + kCommBuffCapacity * i,
                                              kCommBuffCapacity));
      executor_conn_[i].conn = executor_[i].get();
      executor_conn_[i].type = PollConn::ConnType::executor;
      Send(&executor_conn_[i], executor_[i].get());
    }
  }
  send_buff_.clear_to_send();
}

void
Executor::SetUpThreadPool() {
  auto read_pipe = julia_eval_thread_.get_read_pipe();
  julia_eval_pipe_conn_ = std::make_unique<conn::PipeConn>(
      read_pipe, julia_eval_recv_mem_.data(),
      julia_eval_send_mem_.data(), kCommBuffCapacity);
  julia_eval_conn_.type = PollConn::ConnType::compute;
  julia_eval_conn_.conn = julia_eval_pipe_conn_.get();
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
  auto& send_buff = sock_conn->send_buff;
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
    if (poll_conn_ptr->type == PollConn::ConnType::executor
        || poll_conn_ptr->type == PollConn::ConnType::server)
      event_handler_.SetToWriteOnly(poll_conn_ptr);
    else
      event_handler_.SetToReadWrite(poll_conn_ptr);
  }
  send_buff_.reset_sent_sizes();
}

void
Executor::Send(PollConn* poll_conn_ptr, conn::PipeConn* pipe_conn) {
  auto& send_buff = pipe_conn->send_buff;
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
  auto &dist_array = dist_arrays_.at(id);
  auto parent_type = static_cast<DistArrayParentType>(create_dist_array.parent_type());

  dist_array_under_operation_ = id;
  if (kIsServer) {
    if (parent_type != DistArrayParentType::kTextFile)
      return true;
    else
      return false;
  }
  switch (parent_type) {
    case DistArrayParentType::kTextFile:
      {
        std::string file_path = create_dist_array.file_path();
        auto cpp_func = std::bind(
            &DistArray::LoadPartitionsFromTextFile,
            &dist_array,
            file_path);
        exec_cpp_func_task_.func = cpp_func;
        exec_cpp_func_task_.label = TaskLabel::kLoadDistArrayFromTextFile;
        julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
      }
      break;
    case DistArrayParentType::kDistArray:
      {
        int32_t parent_id = create_dist_array.parent_id();
        auto &parent_dist_array = dist_arrays_.at(parent_id);
        auto &parent_dist_array_meta = parent_dist_array.GetMeta();
        auto &child_dist_array_meta = dist_array.GetMeta();
        auto map_type = child_dist_array_meta.GetMapType();
        if (parent_dist_array_meta.IsContiguousPartitions() &&
            (map_type == DistArrayMapType::kMapFixedKeys || map_type == DistArrayMapType::kMapValues)) {
          child_dist_array_meta.SetContiguousPartitions(true);
        }
        auto cpp_func = std::bind(&DistArray::Map, &parent_dist_array, &dist_array);
        exec_cpp_func_task_.func = cpp_func;
        exec_cpp_func_task_.label = TaskLabel::kMapDistArray;
        julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
      }
      break;
    case DistArrayParentType::kInit:
      {
        auto &dist_array_meta = dist_array.GetMeta();
        auto init_type = dist_array_meta.GetInitType();
        if (init_type == DistArrayInitType::kUniformRandom ||
            init_type == DistArrayInitType::kNormalRandom ||
            init_type == DistArrayInitType::kFill) {
          dist_array_meta.SetContiguousPartitions(true);
        }
        auto cpp_func = std::bind(&DistArray::Init, &dist_array);
        exec_cpp_func_task_.func = cpp_func;
        exec_cpp_func_task_.label = TaskLabel::kInitDistArray;
        julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
      }
      break;
    default:
      LOG(FATAL) << "unknown!" << static_cast<int>(parent_type);
  }
  return false;
}

void
Executor::TextFileLoadDone() {
  LOG(INFO) << __func__;
  auto &dist_array = dist_arrays_.at(dist_array_under_operation_);
  auto &dist_array_meta = dist_array.GetMeta();
  auto map_type = dist_array_meta.GetMapType();
  if (map_type == DistArrayMapType::kMap ||
      map_type == DistArrayMapType::kMapFixedKeys) {
    std::vector<int64_t> partition_ids;
    std::vector<size_t> num_lines;
    dist_array.GetPartitionTextBufferNumLines(&partition_ids,
                                              &num_lines);
    if (partition_ids.size() > 0) {
      size_t buff_size = partition_ids.size() * sizeof(int64_t)
                         + num_lines.size() * sizeof(size_t);
      uint8_t* send_bytes_buff = new uint8_t[buff_size];

      memcpy(send_bytes_buff, partition_ids.data(),
             partition_ids.size() * sizeof(int64_t));
      memcpy(send_bytes_buff + partition_ids.size() * sizeof(int64_t),
             num_lines.data(), num_lines.size() * sizeof(size_t));
      message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgPartitionNumLines>(
          &send_buff_, dist_array_under_operation_, partition_ids.size());
      send_buff_.set_next_to_send(send_bytes_buff, buff_size, true);
      Send(&master_poll_conn_, &master_);
      send_buff_.clear_to_send();
      send_buff_.reset_sent_sizes();
    } else {
      message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgPartitionNumLines>(
          &send_buff_, dist_array_under_operation_, 0);
      Send(&master_poll_conn_, &master_);
      send_buff_.clear_to_send();
      send_buff_.reset_sent_sizes();
    }
  } else {
    std::vector<size_t> line_number_start;
    ParseDistArrayTextBuffer(dist_array, line_number_start);
  }
}

void
Executor::ParseDistArrayTextBuffer(DistArray& dist_array,
                                   const std::vector<size_t> &line_number_start) {
  LOG(INFO) << __func__ << " " << line_number_start.size();
  size_t num_dims = dist_array.GetMeta().GetDims().size();
  exec_cpp_func_task_.result_buff.resize(sizeof(int64_t) * num_dims, 0);
  std::vector<size_t> my_line_number_start = line_number_start;
  LOG(INFO) << __func__ << " size = " << my_line_number_start.size();
  exec_cpp_func_task_.func = std::bind(&DistArray::ParseBufferedText,
                            &dist_array,
                            &exec_cpp_func_task_.result_buff,
                            my_line_number_start);

  exec_cpp_func_task_.label = TaskLabel::kParseDistArrayTextBuffer;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
}

bool
Executor::RepartitionDistArray() {
  LOG(INFO) << __func__;
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());

  task::RepartitionDistArray repartition_dist_array_task;
  repartition_dist_array_task.ParseFromString(task_str);
  int32_t id = repartition_dist_array_task.id();
  dist_array_under_operation_ = id;

  auto partition_scheme = static_cast<DistArrayPartitionScheme>(repartition_dist_array_task.partition_scheme());
  auto index_type = static_cast<DistArrayIndexType>(repartition_dist_array_task.index_type());

  auto &dist_array_to_repartition = dist_arrays_.at(id);
  auto &meta = dist_array_to_repartition.GetMeta();
  auto orig_partition_scheme = meta.GetPartitionScheme();
  auto contiguous_partitions = repartition_dist_array_task.contiguous_partitions();

  meta.SetPartitionScheme(partition_scheme);
  meta.SetIndexType(index_type);
  meta.SetContiguousPartitions(contiguous_partitions);

  bool from_server = orig_partition_scheme == DistArrayPartitionScheme::kHashServer;
  bool to_server = partition_scheme == DistArrayPartitionScheme::kHashServer;

  repartition_recv_ = true;
  if ((to_server && (!kIsServer || (from_server && kConfig.kNumServers == 1))) ||
      (!to_server && (kIsServer || (!from_server && kConfig.kNumExecutors == 1)))
      ) {
    repartition_recv_ = false;
  }

  if ((from_server && !kIsServer) || (!from_server && kIsServer)) return false;

  std::function<void()> cpp_func;
  if (partition_scheme == DistArrayPartitionScheme::kSpaceTime ||
      partition_scheme == DistArrayPartitionScheme::k1D) {
    std::string partition_func_name
        = repartition_dist_array_task.partition_func_name();
    cpp_func = std::bind(
        &DistArray::ComputeRepartition,
        &dist_array_to_repartition,
        partition_func_name);
  } else if (partition_scheme == DistArrayPartitionScheme::kRange) {
    LOG(FATAL);
  } else if (partition_scheme == DistArrayPartitionScheme::kHashServer){
    cpp_func = std::bind(
        &DistArray::ComputeHashRepartition,
        &dist_array_to_repartition,
        kConfig.kNumServers);
  } else {
    CHECK(partition_scheme == DistArrayPartitionScheme::kHashExecutor);
    cpp_func = std::bind(
        &DistArray::ComputeHashRepartition,
        &dist_array_to_repartition,
        kConfig.kNumExecutors);
  }

  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kComputeRepartition;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
  return true;
}

bool
Executor::RepartitionDistArraySerialize(int32_t dist_array_id,
                                        DistArray *dist_array) {
  auto &meta = dist_array->GetMeta();
  auto partition_scheme = meta.GetPartitionScheme();
  if ((partition_scheme == DistArrayPartitionScheme::kSpaceTime ||
       partition_scheme == DistArrayPartitionScheme::k1D ||
       partition_scheme == DistArrayPartitionScheme::kRange ||
       partition_scheme == DistArrayPartitionScheme::kHashExecutor) &&
      (!kIsServer && kNumExecutors == 1)) {
    return false;
  }
  if ((partition_scheme == DistArrayPartitionScheme::kHashServer) &&
      (kIsServer && kNumServers == 1)) {
    return false;
  }

  auto cpp_func = std::bind(
      &DistArray::RepartitionSerializeAndClear,
      dist_array,
      &repartition_send_buffer_);

  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kRepartitionSerialize;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
  return true;
}

void
Executor::RepartitionDistArraySend() {
  auto dist_array_id = dist_array_under_operation_;
  auto &dist_array_to_repartition = dist_arrays_.at(dist_array_under_operation_);
  auto &meta = dist_array_to_repartition.GetMeta();
  auto partition_scheme = meta.GetPartitionScheme();

  bool to_server = partition_scheme == DistArrayPartitionScheme::kHashServer;
  size_t num_receivers = to_server ? kNumServers : kNumExecutors;
  size_t skip_id = num_receivers;

  if (kIsServer && to_server) {
    skip_id = kServerId;
  } else if (!kIsServer && !to_server) {
    skip_id = kExecutorId;
  }

  auto &recv_conn = to_server ? server_conn_ : executor_conn_;
  auto &receiver = to_server ? server_ : executor_;
  for (size_t recv_id = 0; recv_id < num_receivers; recv_id++) {
    if (recv_id == skip_id) continue;
    auto iter = repartition_send_buffer_.find(recv_id);
    if (iter == repartition_send_buffer_.end()) {
      message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRepartitionDistArrayData>(
          &send_buff_, dist_array_id, 0, kIsServer);
      Send(&recv_conn[recv_id], receiver[recv_id].get());
      send_buff_.clear_to_send();
      send_buff_.reset_sent_sizes();
    } else {
      auto &buff = iter->second;
      size_t send_size = buff.second;
      uint8_t* send_data = buff.first;
      message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgRepartitionDistArrayData>(
          &send_buff_, dist_array_id, send_size, kIsServer);
      send_buff_.set_next_to_send(send_data, send_size, true);
      Send(&recv_conn[recv_id], receiver[recv_id].get());
      send_buff_.clear_to_send();
      send_buff_.reset_sent_sizes();
    }
  }
  repartition_send_buffer_.clear();
}

void
Executor::UpdateDistArrayIndex() {
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());
  task::UpdateDistArrayIndex update_dist_array_index_task;
  update_dist_array_index_task.ParseFromString(task_str);
  int32_t id = update_dist_array_index_task.id();
  auto index_type = static_cast<DistArrayIndexType>(update_dist_array_index_task.index_type());

  auto &dist_array_to_update = dist_arrays_.at(id);
  auto &meta = dist_array_to_update.GetMeta();
  meta.SetIndexType(index_type);

  auto cpp_func = std::bind(
      &DistArray::CheckAndBuildIndex,
      &dist_array_to_update);

  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kUpdateDistArrayIndex;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
}

void
Executor::DefineJuliaDistArray() {
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());

  task::CreateDistArray create_dist_array;
  create_dist_array.ParseFromString(task_str);

  int32_t id = create_dist_array.id();
  dist_array_under_operation_ = id;
  type::PrimitiveType value_type
      = static_cast<type::PrimitiveType>(create_dist_array.value_type());
  std::string serialized_value_type = create_dist_array.serialized_value_type();
  size_t num_dims = create_dist_array.num_dims();
  auto parent_type = static_cast<DistArrayParentType>(create_dist_array.parent_type());

  auto init_type = create_dist_array.has_init_type() ?
                                static_cast<DistArrayInitType>(create_dist_array.init_type()) :
                                DistArrayInitType::kEmpty;

  auto map_type = static_cast<DistArrayMapType>(create_dist_array.map_type());
  auto map_func_module = create_dist_array.has_map_func_module() ?
                         static_cast<JuliaModule>(create_dist_array.map_func_module()) :
                         JuliaModule::kNone;

  auto map_func_name = create_dist_array.has_map_func_name() ?
                       create_dist_array.map_func_name() :
                       "";

  auto random_init_type = create_dist_array.has_random_init_type() ?
                          static_cast<type::PrimitiveType>(create_dist_array.random_init_type())
                          : type::PrimitiveType::kVoid;

  bool flatten_results = create_dist_array.flatten_results();
  bool is_dense = create_dist_array.is_dense();
  auto symbol = create_dist_array.symbol();
  auto partition_scheme = static_cast<DistArrayPartitionScheme>(
      create_dist_array.partition_scheme());

  auto iter_pair = dist_arrays_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(id),
      std::forward_as_tuple(id,
                            kConfig,
                            kIsServer,
                            value_type,
                            kExecutorId,
                            kServerId,
                            num_dims,
                            parent_type,
                            init_type,
                            map_type,
                            partition_scheme,
                            map_func_module,
                            map_func_name,
                            random_init_type,
                            flatten_results,
                            is_dense,
                            symbol,
                            julia_requester_));

  DistArray &dist_array = iter_pair.first->second;
  if (create_dist_array.dims_size() > 0) {
    const int64_t *dims = create_dist_array.dims().data();
    CHECK_EQ(num_dims, create_dist_array.dims_size());
    dist_array.SetDims(dims, num_dims);
  }

  if (create_dist_array.has_serialized_init_value()) {
    std::string serialized_init_value = create_dist_array.serialized_init_value();
    dist_array.GetMeta().SetInitValue(
        reinterpret_cast<const uint8_t*>(serialized_init_value.data()),
        serialized_init_value.size());
  }

  auto &meta = dist_array.GetMeta();
  const auto &dims = meta.GetDims();
  auto cpp_func = std::bind(
      JuliaEvaluator::DefineDistArray,
      id,
      symbol,
      serialized_value_type,
      dims,
      is_dense,
      false,
      std::vector<uint8_t>(),
      &dist_array);

  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kDefineJuliaDistArray;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
}

void
Executor::DefineJuliaDistArrayBuffer() {
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());
  task::CreateDistArrayBuffer create_dist_array_buffer;
  create_dist_array_buffer.ParseFromString(task_str);

  int32_t id = create_dist_array_buffer.id();
  size_t num_dims = create_dist_array_buffer.num_dims();
  type::PrimitiveType value_type
      = static_cast<type::PrimitiveType>(create_dist_array_buffer.value_type());
  std::string serialized_value_type = create_dist_array_buffer.serialized_value_type();

  auto parent_type = DistArrayParentType::kInit;
  auto init_type = DistArrayInitType::kEmpty;
  auto map_type = DistArrayMapType::kNoMap;
  auto partition_scheme = DistArrayPartitionScheme::kNaive;
  auto map_func_module = JuliaModule::kNone;
  auto map_func_name = "";
  auto random_init_type = type::PrimitiveType::kVoid;
  auto flatten_results = false;
  auto is_dense = create_dist_array_buffer.is_dense();
  auto symbol = create_dist_array_buffer.symbol();

  auto iter_pair = dist_array_buffers_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(id),
      std::forward_as_tuple(id,
                            kConfig,
                            kIsServer,
                            value_type,
                            kExecutorId,
                            kServerId,
                            num_dims,
                            parent_type,
                            init_type,
                            map_type,
                            partition_scheme,
                            map_func_module,
                            map_func_name,
                            random_init_type,
                            flatten_results,
                            is_dense,
                            symbol,
                            julia_requester_));

  DistArray &dist_array = iter_pair.first->second;
  const int64_t *dims = create_dist_array_buffer.dims().data();
  CHECK_EQ(num_dims, create_dist_array_buffer.dims_size());
  dist_array.SetDims(dims, num_dims);

  auto &meta = dist_array.GetMeta();
  std::string serialized_init_value = create_dist_array_buffer.serialized_init_value();
  meta.SetInitValue(reinterpret_cast<const uint8_t*>(serialized_init_value.data()),
                    serialized_init_value.size());
  meta.SetContiguousPartitions(true);

  auto cpp_func = std::bind(
      &DistArray::CreateDistArrayBuffer,
      &dist_array,
      serialized_value_type);

  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kDefineJuliaDistArrayBuffer;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
}

void
Executor::GetAccumulatorValue() {
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());

  task::GetAccumulatorValue get_accumulator_value_task;
  get_accumulator_value_task.ParseFromString(task_str);
  std::string symbol = get_accumulator_value_task.symbol();

  auto cpp_func = std::bind(
      JuliaEvaluator::GetVarValue,
      symbol,
      &(exec_cpp_func_task_.result_buff));

  exec_cpp_func_task_.func = cpp_func;
  exec_cpp_func_task_.label = TaskLabel::kGetAccumulatorValue;
  julia_eval_thread_.SchedTask(static_cast<JuliaTask*>(&exec_cpp_func_task_));
}

void
Executor::ReplyGetAccumulatorValue() {
  size_t result_size = exec_cpp_func_task_.result_buff.size();
  message::ExecuteMsgHelper::CreateMsg<message::ExecuteMsgReplyGetAccumulatorValue>(
      &send_buff_, result_size);
  send_buff_.set_next_to_send(exec_cpp_func_task_.result_buff.data(),
                              exec_cpp_func_task_.result_buff.size());
  Send(&master_poll_conn_, &master_);
  send_buff_.clear_to_send();
  send_buff_.reset_sent_sizes();
}

void
Executor::SetDistArrayBufferInfo() {
  std::string task_str(
      reinterpret_cast<const char*>(master_recv_byte_buff_.GetBytes()),
      master_recv_byte_buff_.GetSize());
  task::SetDistArrayBufferInfo set_dist_array_buffer_info;
  set_dist_array_buffer_info.ParseFromString(task_str);
  int32_t dist_array_buffer_id = set_dist_array_buffer_info.dist_array_buffer_id();
  int32_t dist_array_id = set_dist_array_buffer_info.dist_array_id();
  std::string apply_buffer_func_name = set_dist_array_buffer_info.apply_buffer_func_name();
  const int32_t *helper_dist_array_ids = set_dist_array_buffer_info.helper_dist_array_ids().data();
  size_t num_helper_dist_arrays = set_dist_array_buffer_info.helper_dist_array_ids_size();
  const int32_t *helper_dist_array_buffer_ids = set_dist_array_buffer_info.helper_dist_array_buffer_ids().data();
  size_t num_helper_dist_array_buffers = set_dist_array_buffer_info.helper_dist_array_buffer_ids_size();

  dist_array_buffer_info_map_.emplace(dist_array_buffer_id,
                                      DistArrayBufferInfo(dist_array_id,
                                                          apply_buffer_func_name,
                                                          helper_dist_array_ids,
                                                          num_helper_dist_arrays,
                                                          helper_dist_array_buffer_ids,
                                                          num_helper_dist_array_buffers));
}

void
Executor::DeleteDistArrayBufferInfo(int32_t dist_array_buffer_id) {
  dist_array_buffer_info_map_.erase(dist_array_buffer_id);
}
}

}
