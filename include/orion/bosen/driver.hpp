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
  PollConn master_poll_conn_;
  EventHandler<PollConn> event_handler_;
  message::DriverMsgType expected_msg_type_;
  ByteBuffer result_buff_;
  bool received_from_master_ { false };
  bool sent_to_master_ { true };

 private:
  void CreateCallFuncTask(
      const std::string &function_name,
      task::BaseTableType base_table_type,
      const task::VirtualBaseTable &virtual_base_table,
      const task::ConcreteBaseTable &concrete_base_table,
      const std::vector<task::TableDep> &deps,
      task::Repetition repetition,
      int32_t num_iterations,
      type::PrimitiveType result_type);

  void CreateExecuteCodeTask(
      const std::string &code,
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

  void ExecuteCodeOnOne(
      int32_t executor_id,
      const std::string &code,
      type::PrimitiveType result_type,
      void *result_buff);

  void ExecuteCodeOnAll(
      const std::string &code,
      type::PrimitiveType result_type,
      void *result_buff);

  void CallFuncOnOne(
      int32_t executor_id,
      const std::string &function_name,
      task::BaseTableType base_table_type,
      const task::VirtualBaseTable &virtual_base_table,
      const task::ConcreteBaseTable &concrete_base_table,
      const std::vector<task::TableDep> &deps,
      task::Repetition repetition,
      int32_t num_iterations,
      type::PrimitiveType result_type,
      void *result_buff);

  void CallFuncOnAll(
      const std::string &function_name,
      task::BaseTableType base_table_type,
      const task::VirtualBaseTable &virtual_base_table,
      const task::ConcreteBaseTable &concrete_base_table,
      const std::vector<task::TableDep> &deps,
      task::Repetition repetition,
      int32_t num_iterations,
      type::PrimitiveType result_type,
      void *result_buff);

  void Stop();
};

void
Driver::BlockSendToMaster() {
  bool sent = master_.sock.Send(&master_.send_buff);
  if (sent) return;
  event_handler_.SetToReadWrite(&master_poll_conn_);
  sent_to_master_ = false;
  while (!sent && !sent_to_master_) {
    event_handler_.WaitAndHandleEvent();
  }
}

void
Driver::BlockRecvFromMaster() {
  while (!received_from_master_) {
    event_handler_.WaitAndHandleEvent();
  }
}

int
Driver::HandleMasterMsg(PollConn *poll_conn_ptr) {
  LOG(INFO) << __func__;
  auto &recv_buff = master_.recv_buff;
  auto driver_msg_type = message::DriverMsgHelper::get_type(master_.recv_buff);
  CHECK(driver_msg_type == expected_msg_type_)
      << "received message type = " << static_cast<int>(driver_msg_type);
  int ret = EventHandler<PollConn>::kNoAction;
  switch (driver_msg_type) {
    case message::DriverMsgType::kMasterResponse:
      {
        LOG(INFO) << "handle MasterResponse";
        auto *response_msg = message::DriverMsgHelper::get_msg<
          message::DriverMsgMasterResponse>(recv_buff);
        size_t expected_size = response_msg->result_bytes;
        bool received_next_msg =
            ReceiveArbitraryBytes(master_.sock, &recv_buff, &result_buff_,
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
Driver::CreateCallFuncTask(
      const std::string &function_name,
      task::BaseTableType base_table_type,
      const task::VirtualBaseTable &virtual_base_table,
      const task::ConcreteBaseTable &concrete_base_table,
      const std::vector<task::TableDep> &deps,
      task::Repetition repetition,
      int32_t num_iterations,
      type::PrimitiveType result_type) { }

void
Driver::CreateExecuteCodeTask(
    const std::string &code,
    type::PrimitiveType result_type) {
  task::ExecuteCode execute_code_task;
  execute_code_task.set_code(code);
  execute_code_task.set_result_type(static_cast<int32_t>(result_type));
  execute_code_task.SerializeToString(&msg_buff_);
}

void
Driver::ConnectToMaster() {
  uint32_t ip;
  int ret = GetIPFromStr(kMasterIp.c_str(), &ip);
  CHECK_NE(ret, 0);

  ret = master_.sock.Connect(ip, kMasterPort);
  CHECK(ret == 0) << "executor failed connecting to master";

  master_poll_conn_ = {&master_};
  ret = event_handler_.SetToReadOnly(&master_poll_conn_);
  CHECK_EQ(ret, 0);
}

void
Driver::ExecuteCodeOnOne(
    int32_t executor_id,
    const std::string &code,
    type::PrimitiveType result_type,
    void *result_buff) {
  result_buff_.Reset(type::SizeOf(result_type));
  auto *execute_code_msg = message::DriverMsgHelper::CreateMsg<
    message::DriverMsgExecuteCodeOnOne>(
        &master_.send_buff, executor_id);
  CreateExecuteCodeTask(code, result_type);
  execute_code_msg->task_size = msg_buff_.size();
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  BlockRecvFromMaster();
  auto* msg = message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_.recv_buff);
  LOG(INFO) << "received response, nbytes = " << msg->result_bytes;
  memcpy(result_buff, result_buff_.GetBytes(), result_buff_.GetSize());
}

void
Driver::ExecuteCodeOnAll(
    const std::string &code,
    type::PrimitiveType result_type,
    void *result_buff) { }

void
Driver::CallFuncOnOne(
    int32_t executor_id,
    const std::string &function_name,
    task::BaseTableType base_table_type,
    const task::VirtualBaseTable &virtual_base_table,
    const task::ConcreteBaseTable &concrete_base_table,
    const std::vector<task::TableDep> &deps,
    task::Repetition repetition,
    int32_t num_iterations,
    type::PrimitiveType result_type,
    void *result_buff) { }

void
Driver::CallFuncOnAll(
    const std::string &function_name,
    task::BaseTableType base_table_type,
    const task::VirtualBaseTable &virtual_base_table,
    const task::ConcreteBaseTable &concrete_base_table,
    const std::vector<task::TableDep> &deps,
    task::Repetition repetition,
    int32_t num_iterations,
    type::PrimitiveType result_type,
    void *result_buff) { }

void
Driver::Stop() {
  message::DriverMsgHelper::CreateMsg<message::DriverMsgStop>(
      &master_.send_buff);
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  master_.sock.Close();
}

}
}
