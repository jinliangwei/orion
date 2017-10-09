#pragma once

#include <vector>
#include <julia.h>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/driver_message.hpp>
#include <orion/bosen/util.hpp>
#include <orion/bosen/type.hpp>
#include <orion/bosen/task.pb.h>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/byte_buffer.hpp>
#include <orion/bosen/event_handler.hpp>
#include <orion/bosen/recv_arbitrary_bytes.hpp>
#include <orion/bosen/julia_module.hpp>

namespace orion {
namespace bosen {

class DriverConfig {
 public:
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  const size_t kCommBuffCapacity;
  const size_t kNumExecutors;
  DriverConfig(
      const char* master_ip,
      uint16_t master_port,
      size_t comm_buff_capacity,
      size_t num_executors):
      kMasterIp(master_ip),
      kMasterPort(master_port),
      kCommBuffCapacity(comm_buff_capacity),
      kNumExecutors(num_executors) { }
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
  const size_t kNumExecutors;

  Blob master_recv_mem_;
  Blob master_send_mem_;
  conn::SocketConn master_;
  Blob master_recv_temp_mem_;
  conn::RecvBuffer master_recv_temp_buff_;
  std::string msg_buff_;
  PollConn master_poll_conn_;
  EventHandler<PollConn> event_handler_;
  message::DriverMsgType expected_msg_type_;
  ByteBuffer result_buff_;
  bool received_from_master_ { false };
  bool sent_to_master_ { true };

 private:
  void CreateCallFuncTask(
      const char* function_name,
      task::BaseTableType base_table_type,
      const task::VirtualBaseTable &virtual_base_table,
      const task::ConcreteBaseTable &concrete_base_table,
      const std::vector<task::TableDep> &deps,
      task::Repetition repetition,
      int32_t num_iterations,
      type::PrimitiveType result_type);

  void CreateExecuteCodeTask(
      const char *code,
      type::PrimitiveType result_type);

  int HandleMasterMsg(PollConn *poll_conn_ptr);
  void HandleWriteEvent(PollConn *poll_conn_ptr);
  int HandleClosedConnection(PollConn *poll_conn_ptr);

  void BlockSendToMaster();
  void BlockRecvFromMaster();

  static jl_function_t *GetFunction(jl_module_t* module,
                                    const char* func_name);

 public:
  Driver(const DriverConfig& driver_config):
      kCommBuffCapacity(driver_config.kCommBuffCapacity),
      kMasterIp(driver_config.kMasterIp),
      kMasterPort(driver_config.kMasterPort),
      kNumExecutors(driver_config.kNumExecutors),
      master_recv_mem_(kCommBuffCapacity),
      master_send_mem_(kCommBuffCapacity),
      master_(conn::Socket(),
              master_recv_mem_.data(),
              master_send_mem_.data(),
              kCommBuffCapacity),
      master_recv_temp_mem_(kCommBuffCapacity),
      master_recv_temp_buff_(master_recv_temp_mem_.data(),
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
    jl_init(NULL);
  }
  ~Driver() { }

  void ConnectToMaster();

  jl_value_t* EvalExprOnAll(
      const uint8_t* expr,
      size_t expr_size,
      JuliaModule module);

  const uint8_t* CallFuncOnOne(
      int32_t executor_id,
      const char* function_name,
      task::BaseTableType base_table_type,
      const task::VirtualBaseTable &virtual_base_table,
      const task::ConcreteBaseTable &concrete_base_table,
      const std::vector<task::TableDep> &deps,
      task::Repetition repetition,
      int32_t num_iterations,
      size_t *result_size);

  const uint8_t* CallFuncOnAll(
      const char* function_name,
      task::BaseTableType base_table_type,
      const task::VirtualBaseTable &virtual_base_table,
      const task::ConcreteBaseTable &concrete_base_table,
      const std::vector<task::TableDep> &deps,
      task::Repetition repetition,
      int32_t num_iterations,
      size_t *result_size);

  void CreateDistArray(
      int32_t id,
      task::DistArrayParentType parent_type,
      int32_t map_type,
      bool flatten_results,
      size_t num_dims,
      type::PrimitiveType value_type,
      const char* file_path,
      int32_t parent_id,
      task::DistArrayInitType init_type,
      JuliaModule parser_func_module,
      const char* parser_func_name,
      int64_t *dims,
      int32_t random_init_type);

  void DefineVariable(const char *var_name,
                      const uint8_t *var_value,
                      size_t value_size);

  void RepartitionDistArray(
      int32_t id,
      const char *partition_func_name,
      int32_t partition_scheme,
      int32_t index_type);

  void Stop();
};

jl_function_t*
Driver::GetFunction(jl_module_t* module,
                    const char* func_name) {
  auto* func = jl_get_function(module, func_name);
  CHECK(func != nullptr) << "func_name = " << func_name;
  return func;
}

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
        if (expected_size == 0) {
            received_from_master_ = true;
            ret = EventHandler<PollConn>::kClearOneMsg;
            master_recv_temp_buff_.CopyOneMsg(recv_buff);
        } else {
          bool received_next_msg =
              ReceiveArbitraryBytes(master_.sock, &recv_buff, &result_buff_,
                                    expected_size);
          if (received_next_msg) {
            received_from_master_ = true;
            ret = EventHandler<PollConn>::kClearOneAndNextMsg;
            master_recv_temp_buff_.CopyOneMsg(recv_buff);
          } else {
            ret = EventHandler<PollConn>::kNoAction;
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown message type " << static_cast<int>(driver_msg_type);
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
    const char* func_name,
    task::BaseTableType base_table_type,
    const task::VirtualBaseTable &virtual_base_table,
    const task::ConcreteBaseTable &concrete_base_table,
    const std::vector<task::TableDep> &deps,
    task::Repetition repetition,
    int32_t num_iterations,
    type::PrimitiveType result_type) { }

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

jl_value_t*
Driver::EvalExprOnAll(
    const uint8_t* expr,
    size_t expr_size,
    JuliaModule module) {
  task::EvalExpr eval_expr_task;
  eval_expr_task.set_serialized_expr(
      std::string(reinterpret_cast<const char*>(expr), expr_size));
  LOG(INFO) << __func__ << " module = " << static_cast<int32_t>(module);
  eval_expr_task.set_module(static_cast<int32_t>(module));
  eval_expr_task.SerializeToString(&msg_buff_);
  message::DriverMsgHelper::CreateMsg<message::DriverMsgEvalExpr>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();
  auto* msg = message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);

  jl_value_t *result_array = nullptr;
  jl_value_t *result_array_type = jl_apply_array_type(jl_any_type, 1);

  if (msg->result_bytes > 0) {
    uint8_t *cursor = result_buff_.GetBytes();
    jl_value_t *bytes_array_type = jl_apply_array_type(jl_uint8_type, 1);

    jl_function_t *io_buffer_func
        = GetFunction(jl_base_module, "IOBuffer");
    jl_function_t *deserialize_func
        = GetFunction(jl_base_module, "deserialize");

    jl_value_t *serialized_result_array = nullptr,
                *serialized_result_buff = nullptr,
                   *deserialized_result = nullptr;
    JL_GC_PUSH4(&serialized_result_array, &serialized_result_buff,
                &deserialized_result, &result_array);
    result_array = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(result_array_type, 0));
    for (size_t i = 0; i < kNumExecutors; i++) {
      size_t curr_result_size = *reinterpret_cast<const size_t*>(cursor);
      serialized_result_array = reinterpret_cast<jl_value_t*>(
          jl_ptr_to_array_1d(
              bytes_array_type,
              cursor + sizeof(size_t),
              curr_result_size, 0));
      serialized_result_buff = jl_call1(io_buffer_func,
                                        reinterpret_cast<jl_value_t*>(serialized_result_array));
      deserialized_result = jl_call1(deserialize_func, serialized_result_buff);
      if (jl_exception_occurred()) {
        jl_exception_clear();
        continue;
      }
      jl_array_ptr_1d_push(reinterpret_cast<jl_array_t*>(result_array), deserialized_result);
      LOG(INFO) << "pushed 1 object " << (void*) deserialized_result;
      cursor += curr_result_size + sizeof(size_t);
    }
    JL_GC_POP();
    master_recv_temp_buff_.ClearOneAndNextMsg();
  } else {
    result_array = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(result_array_type, 0));
    master_recv_temp_buff_.ClearOneMsg();
  }
  return result_array;
}

const uint8_t*
Driver::CallFuncOnOne(
    int32_t executor_id,
    const char* function_name,
    task::BaseTableType base_table_type,
    const task::VirtualBaseTable &virtual_base_table,
    const task::ConcreteBaseTable &concrete_base_table,
    const std::vector<task::TableDep> &deps,
    task::Repetition repetition,
    int32_t num_iterations,
    size_t *result_size) {
  return nullptr;
}

const uint8_t*
Driver::CallFuncOnAll(
    const char* function_name,
    task::BaseTableType base_table_type,
    const task::VirtualBaseTable &virtual_base_table,
    const task::ConcreteBaseTable &concrete_base_table,
    const std::vector<task::TableDep> &deps,
    task::Repetition repetition,
    int32_t num_iterations,
    size_t *result_size) {
  return nullptr;
}

void
Driver::CreateDistArray(
      int32_t id,
      task::DistArrayParentType parent_type,
      int32_t map_type,
      bool flatten_results,
      size_t num_dims,
      type::PrimitiveType value_type,
      const char* file_path,
      int32_t parent_id,
      task::DistArrayInitType init_type,
      JuliaModule mapper_func_module,
      const char* mapper_func_name,
      int64_t *dims,
      int32_t random_init_type) {
  task::CreateDistArray create_dist_array;
  create_dist_array.set_id(id);
  create_dist_array.set_parent_type(parent_type);
  switch (parent_type) {
    case task::TEXT_FILE:
      {
        create_dist_array.set_file_path(file_path);
      }
      break;
    case task::DIST_ARRAY:
      {
        create_dist_array.set_parent_id(parent_id);
      }
      break;
    case task::INIT:
      {
        create_dist_array.set_init_type(init_type);
        if (init_type != task::EMPTY) {
          for (size_t i = 0; i < num_dims; i++) {
            LOG(INFO) << "add dim[" << i << "] = " << dims[i];
            create_dist_array.add_dims(dims[i]);
          }
        }
        if (map_type != task::NO_MAP && init_type != task::EMPTY) {
          create_dist_array.set_random_init_type(random_init_type);
        }
      }
      break;
    default:
      LOG(FATAL) << "unrecognized parent type = "
                 << static_cast<int>(parent_type);
  }
  create_dist_array.set_map_type(static_cast<task::DistArrayMapType>(map_type));
  create_dist_array.set_flatten_results(flatten_results);
  if (map_type != task::NO_MAP) {
   create_dist_array.set_mapper_func_module(
        static_cast<int>(mapper_func_module));
    create_dist_array.set_mapper_func_name(mapper_func_name);
  }
  create_dist_array.set_num_dims(num_dims);
  create_dist_array.set_value_type(static_cast<int>(value_type));
  create_dist_array.SerializeToString(&msg_buff_);
  LOG(INFO) << "task size = " << msg_buff_.size();

  message::DriverMsgHelper::CreateMsg<message::DriverMsgCreateDistArray>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();

  auto* msg = message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  CHECK_EQ(msg->result_bytes/sizeof(int64_t), num_dims);
  if (parent_type == task::TEXT_FILE && msg->result_bytes > 0) {
    memcpy(dims, result_buff_.GetBytes(), result_buff_.GetSize());
    master_recv_temp_buff_.ClearOneAndNextMsg();
  } else {
    master_recv_temp_buff_.ClearOneMsg();
  }
}

void
Driver::Stop() {
  message::DriverMsgHelper::CreateMsg<message::DriverMsgStop>(
      &master_.send_buff);
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  master_.sock.Close();
}

void
Driver::DefineVariable(const char *var_name,
                       const uint8_t *var_value,
                       size_t value_size) {
  task::DefineVar define_var_task;
  define_var_task.set_var_name(var_name);
  define_var_task.set_var_value(var_value, value_size);
  define_var_task.SerializeToString(&msg_buff_);
  message::DriverMsgHelper::CreateMsg<message::DriverMsgDefineVar>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  LOG(INFO) << "waiting from master";
  received_from_master_ = false;
  BlockRecvFromMaster();
  LOG(INFO) << "waiting done";
  message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  master_recv_temp_buff_.ClearOneMsg();
}

void
Driver::RepartitionDistArray(
    int32_t id,
    const char *partition_func_name,
    int32_t partition_scheme,
    int32_t index_type) {
  task::RepartitionDistArray repartition_dist_array_task;
  repartition_dist_array_task.set_id(id);
  repartition_dist_array_task.set_partition_func_name(partition_func_name);
  repartition_dist_array_task.set_partition_scheme(partition_scheme);
  repartition_dist_array_task.set_index_type(index_type);
  repartition_dist_array_task.SerializeToString(&msg_buff_);

  message::DriverMsgHelper::CreateMsg<message::DriverMsgRepartitionDistArray>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  LOG(INFO) << "waiting from master";
  received_from_master_ = false;
  BlockRecvFromMaster();
  LOG(INFO) << "waiting done";
  message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  master_recv_temp_buff_.ClearOneMsg();
}

}
}
