#pragma once

#include <vector>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/driver_message.hpp>
#include <orion/bosen/util.hpp>
#include <orion/bosen/types.hpp>


namespace orion {
namespace bosen {

class DriverConfig {
 public:
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  const size_t kCommBuffCapacity;
  DriverConfig(
      const std::string& master_ip,
      uint16_t master_port,
      size_t comm_buff_capacity):
      kMasterIp(master_ip),
      kMasterPort(master_port),
      kCommBuffCapacity(comm_buff_capacity) { }
};

class Driver {
 private:
  const size_t kCommBuffCapacity;
  const std::string kMasterIp;
  const uint16_t kMasterPort;

  std::vector<uint8_t> master_recv_mem_;
  std::vector<uint8_t> master_send_mem_;
  conn::SocketConn master_;

 private:
  void BlockSendToMaster();

 public:
  Driver(const DriverConfig& driver_config):
      kCommBuffCapacity(driver_config.kCommBuffCapacity),
      kMasterIp(driver_config.kMasterIp),
      kMasterPort(driver_config.kMasterPort),
      master_recv_mem_(kCommBuffCapacity),
      master_send_mem_(kCommBuffCapacity),
      master_(conn::Socket(),
              master_recv_mem_.data(),
              master_send_mem_.data(),
              kCommBuffCapacity) { }
  ~Driver() { }

  void ConnectToMaster();
  void ExecuteOnAny(const std::string &cmd, size_t result_size);
  void Stop();
};

void
Driver::BlockSendToMaster() {
  bool sent = master_.sock.Send(&master_.send_buff);
  while (!sent) {
    LOG(INFO) << "send!";
    sent = master_.sock.Send(&master_.send_buff);
  }
  master_.send_buff.clear_to_send();
}

void
Driver::ConnectToMaster() {
  uint32_t ip;
  int ret = GetIPFromStr(kMasterIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = master_.sock.Connect(ip, kMasterPort);
  CHECK(ret == 0) << "executor failed connecting to master";
}

void
Driver::ExecuteOnAny(const std::string &cmd,
                     type::PrimitiveType result_type,
                     void* result_mem) {
  message::DriverMsgHelper::CreateMsg<message::DriverMsgExecuteOnAny>(
      &master_.send_buff, cmd.length(), result_type);
  send_buff_.set_next_to_send(cmd.c_str(),
                              kNumExecutors*sizeof(HostInfo),
                              nullptr);
  BlockSendToMaster();
}

void
Driver::Stop() {
  message::DriverMsgHelper::CreateMsg<message::DriverMsgStop>(
      &master_.send_buff);
  BlockSendToMaster();
  master_.sock.Close();
}

}
}
