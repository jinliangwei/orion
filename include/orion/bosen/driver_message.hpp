#pragma once

#include <orion/bosen/message.hpp>
#include <orion/bosen/types.hpp>

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

struct DriverMsgExecuteOnAny {
  size_t cmd_size;
  type::PrimitiveType result_type;
 private:
  DriverMsgExecuteOnAny() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _cmd_size, size_t _result_size) { }
  static constexpr DriverMsgType get_type() { return DriverMsgType::kExecuteOnAny; }
};

class DriverMsgHelper {
 public:
  template<typename Msg,
           typename MsgCreator = DefaultMsgCreator,
           typename... Args>
  static Msg* CreateMsg(conn::SendBuffer *send_buff,
                        Args... args) {
    DriverMsg* driver_msg = Helper::template CreateMsg<DriverMsg>(send_buff);
    driver_msg->driver_msg_type = Msg::get_type();
    uint8_t *payload_mem = send_buff->get_avai_payload_mem();
    Msg* msg = MsgCreator::template CreateMsg<Msg, Args...>(
        payload_mem, args...);
    send_buff->inc_payload_size(sizeof(Msg));
    return msg;
  }

  static DriverMsgType get_type(conn::RecvBuffer& recv_buff) {
    return Helper::get_msg<DriverMsg>(recv_buff)->driver_msg_type;
  }

  template<typename Msg>
  static Msg* get_msg(conn::RecvBuffer &recv_buff) {
    return reinterpret_cast<Msg*>(
        Helper::get_remaining_mem<DriverMsg>(recv_buff));
  }

  template<typename Msg>
  static uint8_t* get_remaining_mem(conn::RecvBuffer &recv_buff) {
    return Helper::get_remaining_mem<DriverMsg>(recv_buff) + sizeof(Msg);
  }

  template<typename Msg>
  static size_t get_remaining_size(const conn::RecvBuffer &recv_buff) {
    return Helper::get_remaining_size<DriverMsg>(recv_buff) - sizeof(Msg);
  }
};

}
}
}
