#pragma once

#include <vector>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/driver_message.hpp>
#include <orion/bosen/util.hpp>
#include <orion/bosen/type.hpp>
#include <orion/bosen/task.pb.h>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/byte_buffer.hpp>
#include <orion/bosen/event_handler.hpp>
#include <orion/bosen/recv_arbitrary_bytes.hpp>

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
  struct PollConn {
    conn::SocketConn* conn;
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
      return false;
    }
    int get_read_fd() const {
      return conn->sock.get_fd();
    }
    int get_write_fd() const {
      return conn->sock.get_fd();
    }
  };

  const size_t kCommBuffCapacity;
  const std::string kMasterIp;
  const uint16_t kMasterPort;

  Blob master_recv_mem_;
  Blob master_send_mem_;
  conn::SocketConn master_;
  std::string msg_buff_;
  PollConn poll_conn_;
  EventHandler<PollConn> event_handler_;
  message::DriverMsgType expected_msg_type_;
  ByteBuffer *result_buff_ { nullptr };
  bool received_from_master_ { false };
  bool sent_to_master_ { true };

 private:
  void CreateExecuteMsg(
      const std::string &code,
      int32_t base_table,
      const std::vector<task::TableDep> &read_dep,
      const std::vector<task::TableDep> &write_dep,
      task::ExecuteGranularity granularity,
      size_t repetition,
      type::PrimitiveType result_type);

  int HandleMasterMsg(PollConn *poll_conn_ptr);
  void HandleWriteEvent(PollConn *poll_conn_ptr);
  int HandleClosedConnection(PollConn *poll_conn_ptr);

  void BlockSendToMaster();
  void BlockRecvFromMaster();

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
              kCommBuffCapacity) {
    poll_conn_.conn = &master_;
    event_handler_.SetReadEventHandler(
      std::bind(&Driver::HandleMasterMsg, this,
               std::placeholders::_1));

    event_handler_.SetWriteEventHandler(
      std::bind(&Driver::HandleWriteEvent, this,
               std::placeholders::_1));

    event_handler_.SetClosedConnectionHandler(
      std::bind(&Driver::HandleClosedConnection, this,
               std::placeholders::_1));
  }
  ~Driver() { }

  void ConnectToMaster();

  void ExecuteOnOne(
      int32_t executor_id,
      const std::string &code,
      int32_t base_table,
      const std::vector<task::TableDep> &read_dep,
      const std::vector<task::TableDep> &write_dep,
      task::ExecuteGranularity granularity,
      size_t repetition,
      type::PrimitiveType result_type,
      ByteBuffer *result_buff);

  void ExecuteOnAll(
      const std::string &code,
      int32_t base_table,
      const std::vector<task::TableDep> &read_dep,
      const std::vector<task::TableDep> &write_dep,
      task::ExecuteGranularity granularity,
      size_t repetition,
      type::PrimitiveType result_type,
      ByteBuffer *result_buff);

  void Stop();
};

void
Driver::BlockSendToMaster() {
  bool sent = master_.sock.Send(&master_.send_buff);
  while (!sent && !sent_to_master_) {
    event_handler_.WaitAndHandleEvent();
  }
  sent_to_master_ = false;
}

void
Driver::BlockRecvFromMaster() {
  while (!received_from_master_) {
    event_handler_.WaitAndHandleEvent();
  }
}

int
Driver::HandleMasterMsg(PollConn *poll_conn_ptr) {
  auto &recv_buff = master_.recv_buff;
  auto driver_msg_type = message::DriverMsgHelper::get_type(master_.recv_buff);
  CHECK(driver_msg_type == expected_msg_type_);
  int ret = EventHandler<PollConn>::kNoAction;
  switch (driver_msg_type) {
    case message::DriverMsgType::kExecuteResponse:
      {
        auto *response_msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgExecuteResponse>(recv_buff);
        size_t expected_size = response_msg->result_bytes;
        bool received_next_msg =
            ReceiveArbitraryBytes(master_.sock, &recv_buff, result_buff_,
                                  expected_size);
        if (received_next_msg) {
          ret = EventHandler<PollConn>::kClearOneAndNextMsg;
          received_from_master_ = true;
        } else {
          ret = EventHandler<PollConn>::kNoAction;
        }
      }
      break;
    default:
      LOG(FATAL) << "";
  }
  return ret;
}

void
Driver::HandleWriteEvent(PollConn* poll_conn_ptr) {
  bool sent = poll_conn_ptr->Send();

  if (sent) {
    auto &send_buff = poll_conn_ptr->get_send_buff();
    send_buff.clear_to_send();
    event_handler_.SetToReadOnly(poll_conn_ptr);
    sent_to_master_ = true;
  }
}

int
Driver::HandleClosedConnection(PollConn *poll_conn_ptr) {
  LOG(FATAL) << "Lost connection to master";
  return EventHandler<PollConn>::kExit;
}

void
Driver::CreateExecuteMsg(
    const std::string &code,
    int32_t base_table,
    const std::vector<task::TableDep> &read_dep,
    const std::vector<task::TableDep> &write_dep,
    task::ExecuteGranularity granularity,
    size_t repetition,
    type::PrimitiveType result_type) {
  task::Execute execute;
  execute.set_code(code);
  if (base_table >= 0)
    execute.set_base_table(base_table);
  for (const auto &dep : read_dep) {
    auto* dep_ptr = execute.add_read_dep();
    *dep_ptr = dep;
  }
  for (const auto &dep : write_dep) {
    auto* dep_ptr = execute.add_write_dep();
    *dep_ptr = dep;
  }
  execute.set_granularity(granularity);
  execute.set_repetition(repetition);
  execute.set_result_type(static_cast<int32_t>(result_type));
  execute.SerializeToString(&msg_buff_);
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
Driver::ExecuteOnOne(
      int32_t executor_id,
      const std::string &code,
      int32_t base_table,
      const std::vector<task::TableDep> &read_dep,
      const std::vector<task::TableDep> &write_dep,
      task::ExecuteGranularity granularity,
      size_t repetition,
      type::PrimitiveType result_type,
      ByteBuffer *result_buff) {
  auto *execute_msg = message::DriverMsgHelper::CreateMsg<message::DriverMsgExecuteOnOne>(
      &master_.send_buff, executor_id);
  CreateExecuteMsg(code, base_table, read_dep, write_dep, granularity, repetition, result_type);
  execute_msg->task_size = msg_buff_.size();
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  expected_msg_type_ = message::DriverMsgType::kExecuteResponse;
  BlockRecvFromMaster();
}

void
Driver::ExecuteOnAll(
      const std::string &code,
      int32_t base_table,
      const std::vector<task::TableDep> &read_dep,
      const std::vector<task::TableDep> &write_dep,
      task::ExecuteGranularity granularity,
      size_t repetition,
      type::PrimitiveType result_type,
      ByteBuffer *result_buff) {
  auto* execute_msg = message::DriverMsgHelper::CreateMsg<message::DriverMsgExecuteOnAll>(
      &master_.send_buff);
  CreateExecuteMsg(code, base_table, read_dep, write_dep, granularity, repetition, result_type);
  execute_msg->task_size = msg_buff_.size();
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  expected_msg_type_ = message::DriverMsgType::kExecuteResponse;
  BlockRecvFromMaster();
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
