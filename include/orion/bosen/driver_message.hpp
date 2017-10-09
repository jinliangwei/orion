#pragma once

#include <orion/bosen/message.hpp>
#include <orion/bosen/type.hpp>

namespace orion {
namespace bosen {

namespace message {

struct DriverMsgStop {
 private:
  DriverMsgStop() = default;
  friend class DefaultMsgCreator;
 public:
  void Init() { }
  static constexpr DriverMsgType get_type() { return DriverMsgType::kStop; }
};

struct DriverMsgMasterResponse {
  size_t result_bytes;
 private:
  DriverMsgMasterResponse() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _result_size) {
    result_bytes = _result_size;
  }
  static constexpr DriverMsgType get_type() {
    return DriverMsgType::kMasterResponse;
  }
};

struct DriverMsgCreateDistArray {
  size_t task_size;
 private:
  DriverMsgCreateDistArray() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _task_size) {
    task_size = _task_size;
  }
  static constexpr DriverMsgType get_type() {
    return DriverMsgType::kCreateDistArray;
  }
};

struct DriverMsgEvalExpr {
  size_t ast_size;
 private:
  DriverMsgEvalExpr() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _ast_size) {
    ast_size = _ast_size;
  }
  static constexpr DriverMsgType get_type() {
    return DriverMsgType::kEvalExpr;
  }
};

struct DriverMsgDefineVar {
  size_t var_info_size;
 private:
  DriverMsgDefineVar() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _var_info_size) {
    var_info_size = _var_info_size;
  }
  static constexpr DriverMsgType get_type() {
    return DriverMsgType::kDefineVar;
  }
};

struct DriverMsgRepartitionDistArray {
  size_t task_size;
 private:
  DriverMsgRepartitionDistArray() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _task_size) { task_size = _task_size; }
  static constexpr DriverMsgType get_type() {
    return DriverMsgType::kRepartitionDistArray;
  }
};

class DriverMsgHelper {
 public:
  template<typename Msg,
           typename MsgCreator = DefaultMsgCreator,
           typename... Args>
  static Msg* CreateMsg(conn::SendBuffer *send_buff,
                        Args... args) {
    // create driver msg
    DriverMsg* driver_msg = Helper::template CreateMsg<DriverMsg>(send_buff);
    driver_msg->driver_msg_type = Msg::get_type();

    // create msg
    uint8_t *payload_mem = send_buff->get_avai_payload_mem();
    uint8_t *aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(payload_mem, alignof(Msg)));
    Msg* msg = MsgCreator::template CreateMsg<Msg, Args...>(
        aligned_mem, args...);
    send_buff->inc_payload_size(aligned_mem + sizeof(Msg) - payload_mem);
    return msg;
  }

  static DriverMsgType get_type(conn::RecvBuffer& recv_buff) {
    return Helper::get_msg<DriverMsg>(recv_buff)->driver_msg_type;
  }

  template<typename Msg>
  static Msg* get_msg(conn::RecvBuffer &recv_buff) {
    uint8_t* mem = Helper::get_remaining_mem<DriverMsg>(recv_buff);
    uint8_t* aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(mem, alignof(Msg)));

    return reinterpret_cast<Msg*>(aligned_mem);
  }

  template<typename Msg>
  static uint8_t* get_remaining_mem(conn::RecvBuffer &recv_buff) {
    uint8_t *mem = Helper::get_remaining_mem<DriverMsg>(recv_buff);
    uint8_t* aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(mem, alignof(Msg)));
    return aligned_mem + sizeof(Msg);
  }

  template<typename Msg>
  static size_t get_remaining_size(conn::RecvBuffer &recv_buff) {
    uint8_t* mem = Helper::get_remaining_mem<DriverMsg>(recv_buff);
    uint8_t* aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(mem, alignof(Msg)));

    return recv_buff.get_payload_capacity()
        - (aligned_mem + sizeof(Msg) - recv_buff.get_payload_mem());
  }
};

}
}
}
