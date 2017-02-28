#pragma once
#include <type_traits>
#include <orion/noncopyable.hpp>
#include <orion/bosen/host_info.hpp>
#include <orion/bosen/conn.hpp>

namespace orion {
namespace bosen {

namespace message {

static constexpr size_t kMaxSize = 1024 - sizeof(conn::beacon_t);

enum class Type {
  kDriverMsg = 0,
    kExecutorConnectToPeers = 1,
    kExecutorConnectToPeersAck = 2,
    kExecutorIdentity = 3,
    kExecutorStop = 4
};

class Helper;

template<Type header_type>
class DefaultPayloadCreator;

struct Header {
  Type type;
  size_t payload_size;
  uint64_t seq, ack;
 private:
  Header() = default;
  friend class Helper;
 public:
  Header (Type _type):
      type(_type),
      payload_size(0) { }
};

static_assert(std::is_pod<Header>::value, "Header must be POD!");
static_assert(sizeof(Header) < kMaxSize, "Message size is beyond limit.");

struct ExecutorConnectToPeers {
 private:
  ExecutorConnectToPeers() = default;
  friend class DefaultPayloadCreator<Type::kExecutorConnectToPeers>;
 public:
  void Init() { }
  static constexpr Type get_type() { return Type::kExecutorConnectToPeers; }
  size_t get_payload_size() const { return 0; }
};

static_assert(std::is_pod<ExecutorConnectToPeers>::value,
              "ExecutorConnectToPeers must be POD!");
static_assert(sizeof(ExecutorConnectToPeers) < kMaxSize,
              "Message size is beyond limit.");

struct ExecutorConnectToPeersAck {
 private:
  ExecutorConnectToPeersAck() = default;
  friend class DefaultPayloadCreator<Type::kExecutorConnectToPeersAck>;
 public:
  void Init() { }
  static constexpr Type get_type() { return Type::kExecutorConnectToPeersAck; }
  size_t get_payload_size() const { return 0; }
};

static_assert(std::is_pod<ExecutorConnectToPeersAck>::value,
                "ExecutorConnectToPeersAck must be POD!");
static_assert(sizeof(ExecutorConnectToPeersAck) < kMaxSize,
              "Message size is beyond limit.");
struct ExecutorStop {
 private:
  ExecutorStop() = default;
  friend class DefaultPayloadCreator<Type::kExecutorStop>;
 public:
  void Init() { }
  static constexpr Type get_type() { return Type::kExecutorStop; }
  size_t get_payload_size() const { return 0; }
};
static_assert(std::is_pod<ExecutorStop>::value,
              "ExecutorStop must be POD!");
static_assert(sizeof(ExecutorStop) < kMaxSize,
              "Message size is beyond limit.");

struct ExecutorIdentity {
  int32_t executor_id;
  HostInfo host_info;
 private:
  ExecutorIdentity() = default;
  friend class DefaultPayloadCreator<Type::kExecutorIdentity>;
 public:
  void Init(int32_t _executor_id, HostInfo _host_info) {
    executor_id = _executor_id;
    host_info = _host_info;
  }
  static constexpr Type get_type() { return Type::kExecutorIdentity; }
  size_t get_payload_size() const { return 0; }
};

static_assert(std::is_pod<ExecutorIdentity>::value, "ExecutorIdentity must be POD!");
static_assert(sizeof(ExecutorIdentity) < kMaxSize,
              "Message size is beyond limit.");

template<Type header_type>
class DefaultPayloadCreator {
 public:
  template<typename Msg, typename... Args>
  static Msg* CreateMsg(uint8_t* mem, Args... args) {
    auto msg = new (mem) Msg();
    msg->Init(args...);
    return msg;
  }
  static Type get_header_type() {
    return header_type;
  }
};

class Helper {
 public:
  template<typename Msg, typename PayloadCreator = DefaultPayloadCreator<Msg::get_type()>,
           typename... Args>
  static Msg* CreateMsg(conn::SendBuffer *send_buff,
                        Args... args) {
    auto header = new (send_buff->get_payload_mem()) Header();
    header->type = PayloadCreator::get_header_type();
    Msg* msg = PayloadCreator::template CreateMsg<Msg, Args...>(send_buff->get_payload_mem()
                                                       + sizeof(Header), args...);
    header->payload_size = sizeof(Msg) + msg->get_payload_size();
    send_buff->set_payload_size(sizeof(Header) + header->payload_size);
    return msg;
  }

  static Type get_type(conn::RecvBuffer& recv_buff) {
    return reinterpret_cast<const Header*>(recv_buff.get_payload_mem())->type;
  }

  static Type get_type(void* payload_mem) {
    return reinterpret_cast<const Header*>(payload_mem)->type;
  }

  static size_t get_payload_capacity(const conn::SendBuffer &send_buff) {
    return send_buff.get_payload_capacity() - sizeof(Header);
  }

  template<typename Msg>
  static Msg* get_msg(conn::RecvBuffer &recv_buff) {
    return reinterpret_cast<Msg*>(recv_buff.get_payload_mem() + sizeof(Header));
  }

  static uint8_t* get_payload_mem(conn::RecvBuffer &recv_buff) {
    return recv_buff.get_payload_mem() + sizeof(Header);
  }
};

} // end of namespace message

} // end of namespace bosen
}
