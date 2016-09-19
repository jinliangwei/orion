#pragma once
#include <type_traits>
#include <orion/noncopyable.hpp>
#include <orion/bosen/host_info.hpp>
#include <orion/bosen/conn.hpp>

namespace orion {
namespace bosen {

namespace message {

enum class Type {
  kExecutorHostInfo = 0,
    kExecutorInfo = 1,
    kExecutorReady = 2,
    kExecutorIdentity = 3,
    kExecutorConnectToPeers = 4,
    kExecutorConnectToPeersAck = 5,
    kExecuteTask = 6,
    kExecutorStop = 7
};

class Helper;

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

static_assert(std::is_pod<Header>::value, "Base must be POD!");

struct ExecutorHostInfo {
  HostInfo host_info;
  int32_t executor_id;
 private:
  ExecutorHostInfo() = default;
  friend class Helper;
 public:
  void Init(int32_t _executor_id) {
    executor_id = _executor_id;
  }
  static Type get_type() { return Type::kExecutorHostInfo; }
  size_t get_payload_size() const { return 0; }
};

static_assert(std::is_pod<ExecutorHostInfo>::value,
              "ExecutorHostInfo must be POD!");

struct ExecutorInfo {
  size_t num_total_executors;
  size_t num_executors;
 private:
  ExecutorInfo() = default;
  friend class Helper;
 public:
  void Init(size_t _num_total_executors,
            size_t _num_executors) {
    num_total_executors = _num_total_executors;
    num_executors = _num_executors;
  }
  HostInfo &get_host_info(size_t idx) {
    HostInfo *host_infos = reinterpret_cast<HostInfo*>(
        reinterpret_cast<uint8_t*>(this) + sizeof(ExecutorInfo));
    return host_infos[idx];
  }

  uint8_t *get_host_info_mem() {
    return reinterpret_cast<uint8_t*>(this) + sizeof(ExecutorInfo);
  }

  size_t get_payload_size() const {
    return sizeof(HostInfo) * num_executors;
  }
  static Type get_type() { return Type::kExecutorInfo; }
};

static_assert(std::is_pod<ExecutorInfo>::value, "ExecutorInfo must be POD!");

struct ExecutorReady {
 private:
  ExecutorReady() = default;
  friend class Helper;
 public:
  void Init() { }
  static Type get_type() { return Type::kExecutorReady; }
  size_t get_payload_size() const { return 0; }
};

static_assert(std::is_pod<ExecutorReady>::value, "ExecutorReady must be POD!");

struct ExecutorConnectToPeers {
 private:
  ExecutorConnectToPeers() = default;
  friend class Helper;
 public:
  void Init() { }
  static Type get_type() { return Type::kExecutorConnectToPeers; }
  size_t get_payload_size() const { return 0; }
};

static_assert(std::is_pod<ExecutorConnectToPeers>::value,
              "ExecutorConnectToPeers must be POD!");

struct ExecutorConnectToPeersAck {
 private:
  ExecutorConnectToPeersAck() = default;
  friend class Helper;
 public:
  void Init() { }
  static Type get_type() { return Type::kExecutorConnectToPeersAck; }
  size_t get_payload_size() const { return 0; }
};

static_assert(std::is_pod<ExecutorConnectToPeersAck>::value,
              "ExecutorConnectToPeersAck must be POD!");

struct ExecutorIdentity {
  int32_t executor_id;
 private:
  ExecutorIdentity() = default;
  friend class Helper;
 public:
  void Init(int32_t _executor_id) {
    executor_id = _executor_id;
  }
  static Type get_type() { return Type::kExecutorIdentity; }
  size_t get_payload_size() const { return 0; }
};

static_assert(std::is_pod<ExecutorIdentity>::value, "ExecutorIdentity must be POD!");

struct ExecutorStop {
 private:
  ExecutorStop() = default;
  friend class Helper;
 public:
  void Init() { }
  static Type get_type() { return Type::kExecutorStop; }
  size_t get_payload_size() const { return 0; }
};
static_assert(std::is_pod<ExecutorStop>::value,
              "ExecutorStop must be POD!");

struct ExecuteTask {
 private:
  ExecuteTask() = default;
  friend class Helper;
 public:
  void Init() { }
  static Type get_type() { return Type::kExecuteTask; }
  size_t get_payload_size() const { return 0; }
};
static_assert(std::is_pod<ExecuteTask>::value,
              "ExecuteTask must be POD!");

class Helper {
 public:
  template<typename Msg, typename... Args>
  static Msg* CreateMsg(conn::SendBuffer *send_buff,
                        Args... args) {
    auto header = new (send_buff->get_payload_mem()) Header();
    header->type = Msg::get_type();
    auto* msg = new (send_buff->get_payload_mem() + sizeof(Header)) Msg();
    msg->Init(args...);
    header->payload_size = msg->get_payload_size();
    send_buff->set_payload_size(sizeof(Header) + sizeof(Msg)
                                + msg->get_payload_size());
    return msg;
  }

  static Type get_type(conn::RecvBuffer& recv_buff) {
    return reinterpret_cast<const Header*>(recv_buff.get_payload_mem())->type;
  }

  static Type get_type(void* payload_mem) {
    return reinterpret_cast<const Header*>(payload_mem)->type;
  }

  template<typename Msg>
  static size_t get_msg_payload_capacity(const conn::SendBuffer &send_buff) {
    return send_buff.get_payload_capacity() - sizeof(Header) - sizeof(Msg);
  }

  template<typename Msg>
  static Msg* get_msg(conn::RecvBuffer &recv_buff) {
    return reinterpret_cast<Msg*>(recv_buff.get_payload_mem() + sizeof(Header));
  }

  template<typename Msg>
  static uint8_t* get_msg_payload_mem(conn::RecvBuffer &recv_buff) {
    return recv_buff.get_payload_mem() + sizeof(Header) + sizeof(Msg);
  }
};


} // end of namespace message

} // end of namespace bosen
}
