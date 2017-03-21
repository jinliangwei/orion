#pragma once

#include <orion/bosen/message.hpp>

namespace orion {
namespace bosen {
namespace message {

struct ExecuteMsgExecuteCode {
  size_t task_size;
 private:
  ExecuteMsgExecuteCode() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _task_size) {
    task_size = _task_size;
  }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kExecuteCode; }
};

struct ExecuteMsgJuliaEvalAck {
 private:
  ExecuteMsgJuliaEvalAck() = default;
  friend class DefaultMsgCreator;
 public:
  void Init() { }
  static constexpr ExecuteMsgType get_type() {
    return ExecuteMsgType::kJuliaEvalAck; }
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
