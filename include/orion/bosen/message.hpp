#pragma once
#include <type_traits>
#include <orion/noncopyable.hpp>
#include <orion/bosen/host_info.h>
#include <orion/bosen/conn.hpp>

namespace orion {
namespace bosen {

namespace message {

enum class Type {
  kDriverMsg = 0,
    kWorkerConnectToPeers = 1,
    kWorkerConnectToPeersAck = 2,
    kWorkerIdentity = 3,
    kWorkerStop = 4
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

struct WorkerConnectToPeers {
 private:
  WorkerConnectToPeers() = default;
  friend class DefaultPayloadCreator<Type::kWorkerConnectToPeers>;
 public:
  void Init() { }
  static Type get_type() { return Type::kWorkerConnectToPeers; }
  size_t get_payload_size() const { return 0; }
};

static_assert(std::is_pod<WorkerConnectToPeers>::value,
              "WorkerConnectToPeers must be POD!");

struct WorkerConnectToPeersAck {
 private:
  WorkerConnectToPeersAck() = default;
  friend class DefaultPayloadCreator<Type::kWorkerConnectToPeersAck>;
 public:
  void Init() { }
  static Type get_type() { return Type::kWorkerConnectToPeersAck; }
  size_t get_payload_size() const { return 0; }
};

static_assert(std::is_pod<WorkerConnectToPeersAck>::value,
                "WorkerConnectToPeersAck must be POD!");

struct WorkerStop {
 private:
  WorkerStop() = default;
  friend class DefaultPayloadCreator<Type::kWorkerStop>;
 public:
  void Init() { }
  static Type get_type() { return Type::kWorkerStop; }
  size_t get_payload_size() const { return 0; }
};
static_assert(std::is_pod<WorkerStop>::value,
              "WorkerStop must be POD!");

struct WorkerIdentity {
  int32_t worker_id;
 private:
  WorkerIdentity() = default;
  friend class DefaultPayloadCreator<Type::kWorkerIdentity>;
 public:
  void Init(int32_t _worker_id) {
    worker_id = _worker_id;
  }
  static Type get_type() { return Type::kWorkerIdentity; }
  size_t get_payload_size() const { return 0; }
};

static_assert(std::is_pod<WorkerIdentity>::value, "WorkerIdentity must be POD!");

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
  template<typename Msg, typename PayloadCreator, typename... Args>
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
