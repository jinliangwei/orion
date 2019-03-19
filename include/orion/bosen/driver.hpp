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
#include <orion/bosen/dist_array_meta.hpp>
#include <orion/bosen/julia_evaluator.hpp>

namespace orion {
namespace bosen {

class DriverConfig {
 public:
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  const size_t kCommBuffCapacity;
  DriverConfig(
      const char* master_ip,
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
    jl_init();
  }
  ~Driver() { }

  void ConnectToMaster();

  jl_value_t* EvalExprOnAll(
      const uint8_t* expr,
      size_t expr_size,
      JuliaModule module);

  void CreateDistArray(
      int32_t id,
      int32_t parent_type,
      int32_t map_type,
      int32_t partition_scheme,
      bool flatten_results,
      size_t num_dims,
      int32_t value_type,
      const char* file_path,
      int32_t parent_id,
      int32_t init_type,
      JuliaModule map_func_module,
      const char* map_func_name,
      int64_t *dims,
      int32_t random_init_type,
      bool is_dense,
      const char* symbol,
      const uint8_t* value_type_bytes,
      size_t value_type_size,
      const uint8_t* init_value_bytes,
      size_t init_value_size,
      const char* key_func_name);

  void RepartitionDistArray(
      int32_t id,
      const char *partition_func_name,
      int32_t partition_scheme,
      int32_t index_type,
      bool contiguous_partitions,
      size_t *partition_dims,
      size_t num_partition_dims);

  void UpdateDistArrayIndex(
      int32_t id,
      int32_t new_index_type);

  void CreateDistArrayBuffer(
      int32_t id,
      int64_t *dims,
      size_t num_dims,
      bool is_dense,
      int32_t value_type,
      jl_value_t *init_value,
      const char* symbol,
      const uint8_t* value_type_bytes,
      size_t value_type_size);

  void DeleteDistArray(
      int32_t id);

  void SetDistArrayBufferInfo(
      int32_t dist_array_buffer_id,
      int32_t dist_array_id,
      const char *apply_buffer_func_name,
      const int32_t *helper_buffer_ids,
      size_t num_helper_buffers,
      const int32_t *helper_dist_array_ids,
      size_t num_helper_dist_arrays,
      int32_t dist_array_buffer_delay_mode,
      size_t max_delay);

  void DeleteDistArrayBufferInfo(
      int32_t dist_array_buffer_id);

  void ExecForLoop(
      int32_t exec_for_loop_id,
      int32_t iteration_space_id,
      int32_t parallel_scheme,
      const int32_t *space_partitioned_dist_array_ids,
      size_t num_space_partitioned_dist_arrays,
      const int32_t *time_partitioned_dist_array_ids,
      size_t num_time_partitioned_dist_arrays,
      const int32_t *global_indexed_dist_array_ids,
      size_t num_gloabl_indexed_dist_arrays,
      const int32_t *dist_array_buffer_ids,
      size_t num_dist_array_buffers,
      const int32_t *written_dist_array_ids,
      size_t num_written_dist_array_ids,
      const int32_t *accessed_dist_array_ids,
      size_t num_accessed_dist_arrays,
      jl_value_t *global_read_only_var_vals,
      const char **accumulator_var_syms,
      size_t num_accumulator_var_syms,
      const char* loop_batch_func_name,
      const char *prefetch_batch_func_name,
      bool is_ordered,
      bool is_repeated);

  jl_value_t* GetAccumulatorValue(
      const char *symbol,
      const char *combiner);

  void RandomRemapPartialKeys(
      int32_t dist_array_id,
      size_t* dim_indices,
      size_t num_dim_indices);

  jl_value_t* ComputeHistogram(
      int32_t dist_array_id,
      size_t dim_index,
      size_t num_bins);

  void SaveAsTextFile(
      int32_t dist_array_id,
      const char* to_string_func_name,
      const char* file_path);

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
  jl_value_t *result_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_any_type), 1);

  if (msg->result_bytes > 0) {
    uint8_t *cursor = result_buff_.GetBytes();
    jl_value_t *bytes_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_uint8_type), 1);

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
    size_t num_responses = *reinterpret_cast<const size_t*>(cursor);
    cursor += sizeof(size_t);
    for (size_t i = 0; i < num_responses; i++) {
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

void
Driver::CreateDistArray(
      int32_t id,
      int32_t parent_type,
      int32_t map_type,
      int32_t partition_scheme,
      bool flatten_results,
      size_t num_dims,
      int32_t value_type,
      const char* file_path,
      int32_t parent_id,
      int32_t init_type,
      JuliaModule map_func_module,
      const char* map_func_name,
      int64_t *dims,
      int32_t random_init_type,
      bool is_dense,
      const char* symbol,
      const uint8_t* value_type_bytes,
      size_t value_type_size,
      const uint8_t* init_value_bytes,
      size_t init_value_size,
      const char* key_func_name) {
  task::CreateDistArray create_dist_array;
  create_dist_array.set_id(id);
  create_dist_array.set_parent_type(parent_type);
  switch (static_cast<DistArrayParentType>(parent_type)) {
    case DistArrayParentType::kTextFile:
      {
        create_dist_array.set_file_path(file_path);
      }
      break;
    case DistArrayParentType::kDistArray:
      {
        for (size_t i = 0; i < num_dims; i++) {
            create_dist_array.add_dims(dims[i]);
        }

        create_dist_array.set_parent_id(parent_id);
      }
      break;
    case DistArrayParentType::kInit:
      {
        create_dist_array.set_init_type(init_type);
        for (size_t i = 0; i < num_dims; i++) {
            create_dist_array.add_dims(dims[i]);
        }

        if (static_cast<DistArrayInitType>(init_type) == DistArrayInitType::kNormalRandom
            || static_cast<DistArrayInitType>(init_type) == DistArrayInitType::kUniformRandom) {
          create_dist_array.set_random_init_type(random_init_type);
        }
      }
      break;
    default:
      LOG(FATAL) << "unrecognized parent type = "
                 << static_cast<int>(parent_type)
                 << " id = " << id;
  }
  create_dist_array.set_map_type(map_type);
  create_dist_array.set_partition_scheme(partition_scheme);
  create_dist_array.set_flatten_results(flatten_results);
  if (static_cast<DistArrayMapType>(map_type) != DistArrayMapType::kNoMap) {
   create_dist_array.set_map_func_module(
        static_cast<int>(map_func_module));
    create_dist_array.set_map_func_name(map_func_name);
  }
  if (static_cast<DistArrayMapType>(map_type) == DistArrayMapType::kGroupBy) {
    create_dist_array.set_key_func_name(key_func_name);
  }
  create_dist_array.set_num_dims(num_dims);
  create_dist_array.set_value_type(value_type);
  create_dist_array.set_is_dense(is_dense);
  create_dist_array.set_symbol(symbol);
  LOG(INFO) << "create_dist_array, symbol = " << symbol;
  create_dist_array.set_serialized_value_type(
      std::string(reinterpret_cast<const char*>(value_type_bytes), value_type_size));
  if (init_value_size > 0) {
    create_dist_array.set_serialized_init_value(
        std::string(reinterpret_cast<const char*>(init_value_bytes), init_value_size));
  }
  create_dist_array.SerializeToString(&msg_buff_);

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
  CHECK_EQ(msg->result_bytes / sizeof(int64_t), num_dims);
  if (static_cast<DistArrayParentType>(parent_type) == DistArrayParentType::kTextFile
      && msg->result_bytes > 0) {
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
Driver::RepartitionDistArray(
    int32_t id,
    const char *partition_func_name,
    int32_t partition_scheme,
    int32_t index_type,
    bool contiguous_partitions,
    size_t *partition_dims,
    size_t num_partition_dims) {
  task::RepartitionDistArray repartition_dist_array_task;
  repartition_dist_array_task.set_id(id);
  auto my_partition_scheme = static_cast<DistArrayPartitionScheme>(partition_scheme);
  if (my_partition_scheme == DistArrayPartitionScheme::kSpaceTime ||
      my_partition_scheme == DistArrayPartitionScheme::k1D ||
      my_partition_scheme == DistArrayPartitionScheme::k1DOrdered ||
      my_partition_scheme == DistArrayPartitionScheme::kHashExecutor) {
    repartition_dist_array_task.set_partition_func_name(partition_func_name);
  }
  repartition_dist_array_task.set_partition_scheme(partition_scheme);
  repartition_dist_array_task.set_index_type(index_type);
  repartition_dist_array_task.set_contiguous_partitions(contiguous_partitions);
  for (size_t i = 0; i < num_partition_dims; i++) {
    repartition_dist_array_task.add_partition_dims(partition_dims[i]);
  }
  repartition_dist_array_task.SerializeToString(&msg_buff_);

  message::DriverMsgHelper::CreateMsg<message::DriverMsgRepartitionDistArray>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();
  message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  master_recv_temp_buff_.ClearOneMsg();
}

void
Driver::UpdateDistArrayIndex(
    int32_t id,
    int32_t new_index_type) {
  task::UpdateDistArrayIndex update_dist_array_index_task;
  update_dist_array_index_task.set_id(id);
  update_dist_array_index_task.set_index_type(new_index_type);
  update_dist_array_index_task.SerializeToString(&msg_buff_);

  message::DriverMsgHelper::CreateMsg<message::DriverMsgUpdateDistArrayIndex>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();

  message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  master_recv_temp_buff_.ClearOneMsg();
}

void
Driver::CreateDistArrayBuffer(
    int32_t id,
    int64_t *dims,
    size_t num_dims,
    bool is_dense,
    int32_t value_type,
    jl_value_t *init_value,
    const char* symbol,
    const uint8_t* value_type_bytes,
    size_t value_type_size) {
  task::CreateDistArrayBuffer create_dist_array_buffer;
  create_dist_array_buffer.set_id(id);
  create_dist_array_buffer.set_num_dims(num_dims);
  for (size_t i = 0; i < num_dims; i++) {
    create_dist_array_buffer.add_dims(dims[i]);
  }
  create_dist_array_buffer.set_is_dense(is_dense);
  create_dist_array_buffer.set_value_type(value_type);

  jl_value_t *buff = nullptr;
  jl_value_t *serialized_result_array = nullptr;
  JL_GC_PUSH2(&buff, &serialized_result_array);
  jl_function_t *io_buffer_func
      = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");

  buff = jl_call0(io_buffer_func);
  jl_function_t *serialize_func
      = JuliaEvaluator::GetFunction(jl_base_module, "serialize");
  CHECK(serialize_func != nullptr);
  jl_call2(serialize_func, buff, init_value);
  jl_function_t *takebuff_array_func
      = JuliaEvaluator::GetFunction(jl_base_module, "take!");
  serialized_result_array = jl_call1(takebuff_array_func, buff);
  size_t result_array_length = jl_array_len(serialized_result_array);
  uint8_t* array_bytes = reinterpret_cast<uint8_t*>(jl_array_data(serialized_result_array));
  create_dist_array_buffer.set_serialized_init_value(
      array_bytes, result_array_length);
  JL_GC_POP();

  create_dist_array_buffer.set_symbol(symbol);
  create_dist_array_buffer.set_serialized_value_type(
      std::string(reinterpret_cast<const char*>(value_type_bytes), value_type_size));

  create_dist_array_buffer.SerializeToString(&msg_buff_);

  message::DriverMsgHelper::CreateMsg<message::DriverMsgCreateDistArrayBuffer>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();

  message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  master_recv_temp_buff_.ClearOneMsg();
}

void
Driver::DeleteDistArray(
    int32_t id) {
  message::DriverMsgHelper::CreateMsg<message::DriverMsgDeleteDistArray>(
      &master_.send_buff, id);
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();

  message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  master_recv_temp_buff_.ClearOneMsg();
}

void
Driver::SetDistArrayBufferInfo(
    int32_t dist_array_buffer_id,
    int32_t dist_array_id,
    const char *apply_buffer_func_name,
    const int32_t *helper_buffer_ids,
    size_t num_helper_buffers,
    const int32_t *helper_dist_array_ids,
    size_t num_helper_dist_arrays,
    int32_t dist_array_buffer_delay_mode,
    size_t max_delay) {
  task::SetDistArrayBufferInfo set_dist_array_buffer_info;
  set_dist_array_buffer_info.set_dist_array_buffer_id(dist_array_buffer_id);
  set_dist_array_buffer_info.set_dist_array_id(dist_array_id);
  set_dist_array_buffer_info.set_apply_buffer_func_name(apply_buffer_func_name);
  for (auto i = 0; i < num_helper_buffers; i++) {
    set_dist_array_buffer_info.add_helper_dist_array_buffer_ids(helper_buffer_ids[i]);
  }
  for (auto i = 0; i < num_helper_dist_arrays; i++) {
    set_dist_array_buffer_info.add_helper_dist_array_ids(helper_dist_array_ids[i]);
  }
  set_dist_array_buffer_info.set_delay_mode(dist_array_buffer_delay_mode);
  set_dist_array_buffer_info.set_max_delay(max_delay);
  set_dist_array_buffer_info.SerializeToString(&msg_buff_);

  message::DriverMsgHelper::CreateMsg<message::DriverMsgSetDistArrayBufferInfo>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();

  message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  master_recv_temp_buff_.ClearOneMsg();
}

void
Driver::DeleteDistArrayBufferInfo(
    int32_t dist_array_buffer_id) {
  message::DriverMsgHelper::CreateMsg<message::DriverMsgDeleteDistArrayBufferInfo>(
      &master_.send_buff, dist_array_buffer_id);
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();
  message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  master_recv_temp_buff_.ClearOneMsg();
}

void
Driver::ExecForLoop(
    int32_t exec_for_loop_id,
    int32_t iteration_space_id,
    int32_t parallel_scheme,
    const int32_t *space_partitioned_dist_array_ids,
    size_t num_space_partitioned_dist_arrays,
    const int32_t *time_partitioned_dist_array_ids,
    size_t num_time_partitioned_dist_arrays,
    const int32_t *global_indexed_dist_array_ids,
    size_t num_global_indexed_dist_arrays,
    const int32_t *dist_array_buffer_ids,
    size_t num_dist_array_buffers,
    const int32_t *written_dist_array_ids,
    size_t num_written_dist_array_ids,
    const int32_t *accessed_dist_array_ids,
    size_t num_accessed_dist_arrays,
    jl_value_t *global_read_only_var_vals,
    const char **accumulator_var_syms,
    size_t num_accumulator_var_syms,
    const char *loop_batch_func_name,
    const char *prefetch_batch_func_name,
    bool is_ordered,
    bool is_repeated) {
  LOG(INFO) << __func__;
  task::ExecForLoop exec_for_loop_task;
  exec_for_loop_task.set_exec_for_loop_id(exec_for_loop_id);
  exec_for_loop_task.set_iteration_space_id(iteration_space_id);
  exec_for_loop_task.set_parallel_scheme(parallel_scheme);
  for (size_t i = 0; i < num_space_partitioned_dist_arrays; i++) {
    exec_for_loop_task.add_space_partitioned_dist_array_ids(
        space_partitioned_dist_array_ids[i]);
  }
  for (size_t i = 0; i < num_time_partitioned_dist_arrays; i++) {
    exec_for_loop_task.add_time_partitioned_dist_array_ids(
        time_partitioned_dist_array_ids[i]);
  }
  for (size_t i = 0; i < num_global_indexed_dist_arrays; i++) {
    exec_for_loop_task.add_global_indexed_dist_array_ids(
        global_indexed_dist_array_ids[i]);
  }

  for (size_t i = 0; i < num_dist_array_buffers; i++) {
    exec_for_loop_task.add_dist_array_buffer_ids(
        dist_array_buffer_ids[i]);
  }

  for (size_t i = 0; i < num_dist_array_buffers; i++) {
    exec_for_loop_task.add_written_dist_array_ids(
        written_dist_array_ids[i]);
  }

  for (size_t i = 0; i < num_accessed_dist_arrays; i++) {
    exec_for_loop_task.add_accessed_dist_array_ids(accessed_dist_array_ids[i]);
  }

  size_t num_global_read_only_var_vals = jl_array_len(global_read_only_var_vals);
  for (size_t i = 0; i < num_global_read_only_var_vals; i++) {
    jl_value_t *var_val = jl_arrayref(reinterpret_cast<jl_array_t*>(global_read_only_var_vals), i);
    const uint8_t* val_bytes = reinterpret_cast<uint8_t*>(jl_array_data(var_val));
    size_t num_bytes = jl_array_len(var_val);
    exec_for_loop_task.add_global_read_only_var_vals(val_bytes, num_bytes);
  }

  for (size_t i = 0; i < num_accumulator_var_syms; i++) {
    exec_for_loop_task.add_accumulator_var_syms(accumulator_var_syms[i]);
  }

  exec_for_loop_task.set_loop_batch_func_name(loop_batch_func_name);
  if (prefetch_batch_func_name != nullptr)
    exec_for_loop_task.set_prefetch_batch_func_name(prefetch_batch_func_name);
  exec_for_loop_task.set_is_ordered(is_ordered);
  exec_for_loop_task.set_is_repeated(is_repeated);

  exec_for_loop_task.SerializeToString(&msg_buff_);
  message::DriverMsgHelper::CreateMsg<message::DriverMsgExecForLoop>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();
  message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  master_recv_temp_buff_.ClearOneMsg();
}

jl_value_t*
Driver::GetAccumulatorValue(
    const char *symbol,
    const char *combiner) {
  LOG(INFO) << __func__;
  task::GetAccumulatorValue get_accumulator_value_task;
  get_accumulator_value_task.set_symbol(symbol);
  get_accumulator_value_task.set_combiner(combiner);
  get_accumulator_value_task.SerializeToString(&msg_buff_);

  message::DriverMsgHelper::CreateMsg<message::DriverMsgGetAccumulatorValue>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();
  auto *msg = message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  size_t result_size = msg->result_bytes;
  CHECK_GT(result_size, 0);

  uint8_t *cursor = result_buff_.GetBytes();
  jl_value_t *bytes_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_uint8_type), 1);

  jl_function_t *io_buffer_func
      = GetFunction(jl_base_module, "IOBuffer");
  jl_function_t *deserialize_func
      = GetFunction(jl_base_module, "deserialize");

  jl_value_t *serialized_result_array = nullptr,
              *serialized_result_buff = nullptr,
                 *deserialized_result = nullptr;
  JL_GC_PUSH3(&serialized_result_array,
              &serialized_result_buff,
              &deserialized_result);
  serialized_result_array = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(
          bytes_array_type,
          cursor,
          result_size, 0));
  serialized_result_buff = jl_call1(io_buffer_func,
                                    reinterpret_cast<jl_value_t*>(serialized_result_array));
  JuliaEvaluator::AbortIfException();
  deserialized_result = jl_call1(deserialize_func, serialized_result_buff);

  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  master_recv_temp_buff_.ClearOneAndNextMsg();
  return deserialized_result;
}

void Driver::RandomRemapPartialKeys(
    int32_t dist_array_id,
    size_t* dim_indices,
    size_t num_dim_indices) {
  task::RandomRemapPartialKeys random_remap_partial_keys_task;
  random_remap_partial_keys_task.set_dist_array_id(dist_array_id);
  for (size_t i = 0; i < num_dim_indices; i++) {
    random_remap_partial_keys_task.add_dim_indices(dim_indices[i]);
  }
  random_remap_partial_keys_task.SerializeToString(&msg_buff_);
  message::DriverMsgHelper::CreateMsg<message::DriverMsgRandomRemapPartialKeys>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();
  expected_msg_type_ = message::DriverMsgType::kMasterResponse;

  received_from_master_ = false;
  BlockRecvFromMaster();
  auto *msg = message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  size_t result_size = msg->result_bytes;
  CHECK_EQ(result_size, 0);

  master_recv_temp_buff_.ClearOneMsg();
}

jl_value_t* Driver::ComputeHistogram(
      int32_t dist_array_id,
      size_t dim_index,
      size_t num_bins) {
  task::ComputeHistogram compute_histogram_task;
  compute_histogram_task.set_dist_array_id(dist_array_id);
  compute_histogram_task.set_dim_index(dim_index);
  compute_histogram_task.set_num_bins(num_bins);
  compute_histogram_task.SerializeToString(&msg_buff_);
  message::DriverMsgHelper::CreateMsg<message::DriverMsgComputeHistogram>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();

  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();

  auto *msg = message::DriverMsgHelper::get_msg<message::DriverMsgMasterResponse>(
      master_recv_temp_buff_);
  size_t result_size = msg->result_bytes;
  size_t replied_num_bins = result_size / (sizeof(int64_t) + sizeof(size_t));
  std::vector<int64_t> bin_keys(replied_num_bins);
  std::vector<size_t> bin_sizes(replied_num_bins);

  uint8_t *cursor = result_buff_.GetBytes();
  memcpy(bin_keys.data(), cursor, replied_num_bins * sizeof(int64_t));
  memcpy(bin_sizes.data(), cursor + replied_num_bins * sizeof(int64_t),
         replied_num_bins * sizeof(size_t));

  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 7);
  jl_value_t *&key_array_type_jl = jl_values[0];
  jl_value_t *&size_array_type_jl = jl_values[1];
  jl_value_t *&key_array_jl = jl_values[2];
  jl_value_t *&size_array_jl = jl_values[3];
  jl_value_t *&ret_tuple_jl = jl_values[4];
  jl_value_t *&key_jl = jl_values[5];
  jl_value_t *&size_jl = jl_values[6];

  key_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  key_array_jl = reinterpret_cast<jl_value_t*>(
      jl_alloc_array_1d(key_array_type_jl, replied_num_bins));
  size_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_uint64_type), 1);
  size_array_jl = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(size_array_type_jl,
                                                                  replied_num_bins));

  for (size_t i = 0; i < replied_num_bins; i++) {
    auto bin_key = bin_keys[i];
    auto bin_size = bin_sizes[i];
    key_jl = jl_box_int64(bin_key);
    size_jl = jl_box_uint64(bin_size);
    jl_arrayset(reinterpret_cast<jl_array_t*>(key_array_jl), key_jl, i);
    jl_arrayset(reinterpret_cast<jl_array_t*>(size_array_jl), size_jl, i);
  }
  jl_function_t *tuple_func = JuliaEvaluator::GetFunction(jl_base_module, "tuple");
  ret_tuple_jl = jl_call2(tuple_func, key_array_jl, size_array_jl);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  master_recv_temp_buff_.ClearOneAndNextMsg();
  return ret_tuple_jl;
}

void
Driver::SaveAsTextFile(
    int32_t dist_array_id,
    const char* to_string_func_name,
    const char* file_path) {

  task::SaveAsTextFile save_as_text_file_task;
  save_as_text_file_task.set_dist_array_id(dist_array_id);
  save_as_text_file_task.set_to_string_func_name(to_string_func_name);
  save_as_text_file_task.set_file_path(file_path);
  save_as_text_file_task.SerializeToString(&msg_buff_);
  message::DriverMsgHelper::CreateMsg<message::DriverMsgSaveAsTextFile>(
      &master_.send_buff, msg_buff_.size());
  master_.send_buff.set_next_to_send(msg_buff_.data(), msg_buff_.size());
  BlockSendToMaster();
  master_.send_buff.clear_to_send();
  master_.send_buff.reset_sent_sizes();

  expected_msg_type_ = message::DriverMsgType::kMasterResponse;
  received_from_master_ = false;
  BlockRecvFromMaster();
  master_recv_temp_buff_.ClearOneAndNextMsg();
}

}
}
