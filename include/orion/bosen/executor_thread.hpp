#pragma once

#include <memory>
#include <iostream>
#include <vector>

#include <orion/bosen/config.hpp>
#include <orion/noncopyable.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/util.hpp>
#include <orion/bosen/message.hpp>
#include <orion/bosen/host_info.hpp>
#include <orion/bosen/server_thread.hpp>
#include <orion/bosen/client_thread.hpp>

namespace orion {
namespace bosen {

class ExecutorThread {
 private:
    struct PollConn {
    enum class ConnType {
      listen = 0,
        master = 1,
        pipe = 2
    };
    void *conn;
    ConnType type;
  };
  static const int32_t kPortSpan = 100;
  const size_t kCommBuffCapacity;
  const size_t kNumExecutors;
  const size_t kNumLocalExecutors;
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  const std::string kListenIp;
  const uint16_t kListenPort;
  const int32_t kId;

  conn::Poll poll_;
  std::vector<uint8_t> master_recv_mem_;
  conn::SocketConn master_;
  conn::Socket listen_;
  std::vector<uint8_t> send_mem_;
  conn::SendBuffer send_buff_;
  //State state_;

  std::vector<conn::Socket> clients_;
  size_t num_connected_clients_ {0};
  std::vector<HostInfo> host_info_;
  size_t num_executors_ {0};
  size_t num_executors_recved_ {0};
  // 0: executor <> server :1
  conn::Pipe pipe_ese_[2];
  // 0: executor <> client :1
  conn::Pipe pipe_ecl_[2];
  // 0: server <> client :1
  conn::Pipe pipe_scl_[2];

  std::vector<uint8_t> pipe_recv_mem_;
  std::unique_ptr<ServerThread> server_;
  std::thread server_thread_;
  std::unique_ptr<conn::PipeConn> server_pipe_conn_;
  PollConn server_poll_conn_;

  std::unique_ptr<ClientThread> client_;
  std::thread client_thread_;
  std::unique_ptr<conn::PipeConn> client_pipe_conn_;
  PollConn client_poll_conn_;

  size_t num_peer_connected_threads_ {0};

 public:
  ExecutorThread(const Config& config, int32_t id);
  ~ExecutorThread();
  DISALLOW_COPY(ExecutorThread);
  void operator() ();
 private:
  void InitListener();
  //  void HandleConnection(PollConn* poll_conn_ptr);
  void ConnectToMaster();
};

ExecutorThread::ExecutorThread(const Config& config, int32_t index):
    kCommBuffCapacity(config.kCommBuffCapacity),
    kNumExecutors(config.kNumExecutors),
    kNumLocalExecutors(config.kNumExecutorsPerWorker),
    kMasterIp(config.kMasterIp),
    kMasterPort(config.kMasterPort),
    kListenIp(config.kWorkerIp),
    kListenPort(config.kWorkerPort + index * kPortSpan),
    kId(config.kWorkerId*config.kNumExecutorsPerWorker + index),
    master_recv_mem_(kCommBuffCapacity),
    master_(conn::Socket(), master_recv_mem_.data(),
            kCommBuffCapacity),
    send_mem_(kCommBuffCapacity),
    send_buff_(send_mem_.data(), kCommBuffCapacity),
    clients_(kNumExecutors),
    host_info_(kNumExecutors),
    pipe_recv_mem_(kCommBuffCapacity*3) { }

ExecutorThread::~ExecutorThread() { }

void
ExecutorThread::operator() () {
  //static constexpr size_t kNumEvents = 100;
  InitListener();

  int ret = -1;
  ret = poll_.Init();
  CHECK(ret == 0) << "poll init failed";
  PollConn listen_poll_conn = {&listen_, PollConn::ConnType::listen};
  poll_.Add(listen_.get_fd(), &listen_poll_conn);

  ConnectToMaster();
}

void
ExecutorThread::InitListener () {
  uint32_t ip;
  int ret = GetIPFromStr(kListenIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = listen_.Bind(ip, kListenPort);
  CHECK_EQ(ret, 0);
  ret = listen_.Listen(kNumExecutors);
}


void
ExecutorThread::ConnectToMaster() {
  uint32_t ip;
  int ret = GetIPFromStr(kMasterIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = master_.sock.Connect(ip, kMasterPort);
  CHECK(ret == 0) << "executor has connected";

  HostInfo host_info;
  ret = GetIPFromStr(kListenIp.c_str(), &host_info.ip);
  CHECK_NE(ret, 0);
  host_info.port = kListenPort;
  message::Helper::CreateMsg<
    message::ExecutorIdentity>(&send_buff_, kId, host_info);
  size_t sent_size = master_.sock.Send(&send_buff_);
  CHECK(conn::CheckSendSize(send_buff_, sent_size));
}

}
}
