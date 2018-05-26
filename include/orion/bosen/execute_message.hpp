#pragma once

#include <orion/bosen/message.hpp>
#include <orion/bosen/julia_task.hpp>

namespace orion {
namespace bosen {
namespace message {

struct ExecuteMsgJuliaEvalAck {
  JuliaTask *task;
 private:
  ExecuteMsgJuliaEvalAck() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(JuliaTask *_task) {
    task = _task;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kJuliaEvalAck; }
};

struct ExecuteMsgCreateDistArray {
  size_t task_size;
 private:
  ExecuteMsgCreateDistArray() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _task_size) {
    task_size = _task_size;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kCreateDistArray;
  }
};

struct ExecuteMsgExecutorAck {
  size_t result_size;
 private:
  ExecuteMsgExecutorAck() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _result_size) {
    result_size = _result_size;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kExecutorAck; }
};

struct ExecuteMsgTextFileLoadAck {
  size_t num_dims;
  int32_t dist_array_id;
 private:
  ExecuteMsgTextFileLoadAck() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _num_dims,
            int32_t _dist_array_id) {
    num_dims = _num_dims;
    dist_array_id = _dist_array_id;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kTextFileLoadAck;
  }
};

struct ExecuteMsgDistArrayDims {
  size_t num_dims;
  int32_t dist_array_id;
 private:
  ExecuteMsgDistArrayDims() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _num_dims,
            int32_t _dist_array_id) {
    num_dims = _num_dims;
    dist_array_id = _dist_array_id;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kDistArrayDims;
  }
};

struct ExecuteMsgCreateDistArrayAck {
  int32_t dist_array_id;
 private:
  ExecuteMsgCreateDistArrayAck() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(int32_t _dist_array_id) {
    dist_array_id = _dist_array_id;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kCreateDistArrayAck;
  }
};

struct ExecuteMsgPeerRecvStop {
 private:
  ExecuteMsgPeerRecvStop() = default;
  friend class DefaultMsgCreator;
 public:
  void Init() { }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kPeerRecvStop;
  }
};

struct ExecuteMsgRepartitionDistArrayData {
  int32_t dist_array_id;
  size_t data_size;
  bool from_server;
 private:
  ExecuteMsgRepartitionDistArrayData() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(int32_t _dist_array_id, size_t _data_size,
            bool _from_server) {
    dist_array_id = _dist_array_id;
    data_size = _data_size;
    from_server = _from_server;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kRepartitionDistArrayData;
  }
};

struct ExecuteMsgRepartitionDistArrayRecved {
 public:
  void *data_buff;
 private:
  ExecuteMsgRepartitionDistArrayRecved() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(void *_data_buff) {
    data_buff = _data_buff;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kRepartitionDistArrayRecved;
  }
};

struct ExecuteMsgRepartitionDistArrayAck {
 public:
  int32_t dist_array_id;
  size_t num_dims;
  int32_t max_ids[2];
 private:
  ExecuteMsgRepartitionDistArrayAck() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(int32_t _dist_array_id,
            size_t _num_dims) {
    dist_array_id = _dist_array_id;
    num_dims = _num_dims;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kRepartitionDistArrayAck;
  }
};

struct ExecuteMsgRepartitionDistArrayMaxPartitionIds {
 public:
  int32_t dist_array_id;
  size_t num_dims;
  int32_t max_ids[2];
 private:
  ExecuteMsgRepartitionDistArrayMaxPartitionIds() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(int32_t _dist_array_id,
            size_t _num_dims) {
    dist_array_id = _dist_array_id;
    num_dims = _num_dims;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kRepartitionDistArrayMaxPartitionIds;
  }
};

struct ExecuteMsgPipelinedTimePartitions {
 public:
  size_t data_size;
  uint64_t pred_notice;
 private:
  ExecuteMsgPipelinedTimePartitions() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _data_size,
            uint64_t _pred_notice) {
    data_size = _data_size;
    pred_notice = _pred_notice;
  }

  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kPipelinedTimePartitions;
  }
};

struct ExecuteMsgRequestExecForLoopGlobalIndexedDistArrays {
 private:
  ExecuteMsgRequestExecForLoopGlobalIndexedDistArrays() = default;
  friend class DefaultMsgCreator;
 public:
  void Init() {  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kRequestExecForLoopGlobalIndexedDistArrays;
  }
};

static_assert(std::is_pod<ExecuteMsgRequestExecForLoopGlobalIndexedDistArrays>::value,
              "ExecuteMsgRequestExecForLoopGlobalIndexedDistArrays must be POD!");

struct ExecuteMsgRequestExecForLoopPipelinedTimePartitions {
 private:
  ExecuteMsgRequestExecForLoopPipelinedTimePartitions() = default;
  friend class DefaultMsgCreator;
 public:
  void Init() { }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kRequestExecForLoopPipelinedTimePartitions;
  }
};

struct ExecuteMsgRequestExecForLoopPredecessorCompletion {
 private:
  ExecuteMsgRequestExecForLoopPredecessorCompletion() = default;
  friend class DefaultMsgCreator;
 public:
  void Init() { }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kRequestExecForLoopPredecessorCompletion;
  }
};

struct ExecuteMsgReplyExecForLoopPipelinedTimePartitions {
 public:
  void *data_buff_vec;
  size_t num_data_buffs;
 private:
  ExecuteMsgReplyExecForLoopPipelinedTimePartitions() = default;
  friend class DefaultMsgCreator;

 public:
  void Init(void *_data_buff_vec,
            size_t _num_data_buffs) {
    data_buff_vec = _data_buff_vec;
    num_data_buffs = _num_data_buffs;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kReplyExecForLoopPipelinedTimePartitions;
  }
};

struct ExecuteMsgReplyExecForLoopGlobalIndexedDistArrayData {
 public:
  void *data_buff_vec;
  size_t num_data_buffs;
 private:
  ExecuteMsgReplyExecForLoopGlobalIndexedDistArrayData() = default;
  friend class DefaultMsgCreator;

 public:
  void Init(void *_data_buff_vec,
            size_t _num_data_buffs) {
    data_buff_vec = _data_buff_vec;
    num_data_buffs = _num_data_buffs;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kReplyExecForLoopGlobalIndexedDistArrayData;
  }
};

struct ExecuteMsgReplyGetAccumulatorValue {
 public:
  size_t result_size;
 private:
  ExecuteMsgReplyGetAccumulatorValue() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _result_size) {
    result_size = _result_size;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kReplyGetAccumulatorValue;
  }
};

struct ExecuteMsgPartitionNumLines {
 public:
  int32_t dist_array_id;
  size_t num_partitions;
 private:
  ExecuteMsgPartitionNumLines() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(int32_t _dist_array_id,
            size_t _num_partitions) {
    dist_array_id = _dist_array_id;
    num_partitions = _num_partitions;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kPartitionNumLines;
  }
};

struct ExecuteMsgCreateDistArrayBufferAck {
  int32_t dist_array_buffer_id;
 private:
  ExecuteMsgCreateDistArrayBufferAck() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(int32_t _dist_array_buffer_id) {
    dist_array_buffer_id = _dist_array_buffer_id;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kCreateDistArrayBufferAck;
  }
};

struct ExecuteMsgRequestDistArrayValue {
  int32_t dist_array_id;
  int64_t key;
  int32_t requester_id; // executor id or server id
  bool is_requester_executor;
 private:
  ExecuteMsgRequestDistArrayValue() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(int32_t _dist_array_id,
            int64_t _key,
            int32_t _requester_id,
            bool _is_requester_executor) {
    dist_array_id = _dist_array_id;
    key = _key;
    requester_id = _requester_id;
    is_requester_executor = _is_requester_executor;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kRequestDistArrayValue;
  }
};

struct ExecuteMsgRequestDistArrayValues {
  size_t request_size;
  int32_t requester_id;
  bool is_requester_executor;
 private:
  ExecuteMsgRequestDistArrayValues() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _request_size,
            int32_t _requester_id,
            bool _is_requester_executor) {
    request_size = _request_size;
    requester_id = _requester_id;
    is_requester_executor = _is_requester_executor;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kRequestDistArrayValues;
  }
};

struct ExecuteMsgReplyDistArrayValues {
  size_t reply_size;
 private:
  ExecuteMsgReplyDistArrayValues() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _reply_size) {
    reply_size = _reply_size;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kReplyDistArrayValues;
  }
};

struct ExecuteMsgReplyExecForLoopPredecessorCompletion {
 public:
  void *data_buff_vec;
  size_t num_data_buffs;

 private:
  ExecuteMsgReplyExecForLoopPredecessorCompletion() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(void *_data_buff_vec,
            size_t _num_data_buffs) {
    data_buff_vec = _data_buff_vec;
    num_data_buffs = _num_data_buffs;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kReplyExecForLoopPredecessorCompletion;
  }
};

struct ExecuteMsgExecForLoopAck {
 public:
  int32_t executor_id;
 private:
  ExecuteMsgExecForLoopAck() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(int32_t _executor_id) {
    executor_id = _executor_id;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kExecForLoopAck;
  }
};

struct ExecuteMsgExecForLoopDone {
 public:
  int32_t executor_id;
 private:
  ExecuteMsgExecForLoopDone() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(int32_t _executor_id) {
    executor_id = _executor_id;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kExecForLoopDone;
  }
};

struct ExecuteMsgExecForLoopDistArrayBufferData {
  size_t num_bytes;
 private:
  ExecuteMsgExecForLoopDistArrayBufferData() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _num_bytes) {
    num_bytes = _num_bytes;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kExecForLoopDistArrayBufferData;
  }
};

struct ExecuteMsgExecForLoopDistArrayCacheData {
  size_t num_bytes;
 private:
  ExecuteMsgExecForLoopDistArrayCacheData() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _num_bytes) {
    num_bytes = _num_bytes;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kExecForLoopDistArrayCacheData;
  }
};

struct ExecuteMsgExecForLoopDistArrayBufferDataPtr {
  uint8_t* bytes;
 private:
  ExecuteMsgExecForLoopDistArrayBufferDataPtr() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(uint8_t* _bytes) {
    bytes = _bytes;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kExecForLoopDistArrayBufferDataPtr;
  }
};

struct ExecuteMsgExecForLoopDistArrayCacheDataPtr {
  uint8_t *bytes;
 private:
  ExecuteMsgExecForLoopDistArrayCacheDataPtr() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(uint8_t* _bytes) {
    bytes = _bytes;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kExecForLoopDistArrayCacheDataPtr;
  }
};

class ExecuteMsgHelper {
 public:
  template<typename Msg,
           typename MsgCreator = DefaultMsgCreator,
           typename... Args>
  static Msg* CreateMsg(conn::SendBuffer *send_buff,
                        Args... args) {
    ExecuteMsg* execute_msg = Helper::template CreateMsg<ExecuteMsg>(send_buff);
    execute_msg->execute_msg_type = Msg::get_type();

    uint8_t *payload_mem = send_buff->get_avai_payload_mem();
    uint8_t *aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(payload_mem, alignof(Msg)));

    Msg* msg = MsgCreator::template CreateMsg<Msg, Args...>(
        aligned_mem, args...);
    send_buff->inc_payload_size(aligned_mem + sizeof(Msg) - payload_mem);
    return msg;
  }

  static ExecuteMsgType get_type(conn::RecvBuffer& recv_buff) {
    return Helper::get_msg<ExecuteMsg>(recv_buff)->execute_msg_type;
  }

  template<typename Msg>
  static Msg* get_msg(conn::RecvBuffer &recv_buff) {
    uint8_t *mem = Helper::get_remaining_mem<ExecuteMsg>(recv_buff);
    uint8_t* aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(mem, alignof(Msg)));

    return reinterpret_cast<Msg*>(aligned_mem);
  }

  template<typename Msg>
  static uint8_t* get_remaining_mem(conn::RecvBuffer &recv_buff) {
    uint8_t *mem = Helper::get_remaining_mem<ExecuteMsg>(recv_buff);
    uint8_t* aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(mem, alignof(Msg)));
    return aligned_mem + sizeof(Msg);
  }

  template<typename Msg>
  static size_t get_remaining_size(conn::RecvBuffer &recv_buff) {
    uint8_t *mem = Helper::get_remaining_mem<ExecuteMsg>(recv_buff);
    uint8_t* aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(mem, alignof(Msg)));

    return recv_buff.get_payload_capacity()
        - (aligned_mem + sizeof(Msg) - recv_buff.get_payload_mem());
  }
};

}
}
}
