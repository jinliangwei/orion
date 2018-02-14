#pragma once
#include <type_traits>
#include <orion/noncopyable.hpp>
#include <orion/bosen/host_info.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/util.hpp>

namespace orion {
namespace bosen {

namespace message {

enum class Type {
  kMasterMsg = 0,
    kDriverMsg = 1,
    kExecutorConnectToPeers = 2,
    kExecutorConnectToPeersAck = 3,
    kExecutorIdentity = 4,
    kExecutorReady = 5,
    kExecuteMsg = 6,
    kServerIdentity = 7,
    kExecutorStop = 10
};

class Helper;
class DefaultMsgCreator;

struct Header {
  Type type;
  uint64_t seq, ack;
 private:
  Header() = default;
  friend class Helper;
 public:
  Header (Type _type):
      type(_type) { }
};

static_assert(std::is_pod<Header>::value, "Header must be POD!");

struct ExecutorConnectToPeers {
 public:
  size_t num_executors;
  size_t num_servers;
 private:
  ExecutorConnectToPeers() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(size_t _num_executors,
            size_t _num_servers) {
    num_executors = _num_executors;
    num_servers = _num_servers;
  }
  static constexpr Type get_type() { return Type::kExecutorConnectToPeers; }
};

static_assert(std::is_pod<ExecutorConnectToPeers>::value,
              "ExecutorConnectToPeers must be POD!");

struct ExecutorConnectToPeersAck {
 private:
  ExecutorConnectToPeersAck() = default;
  friend class DefaultMsgCreator;
 public:
  void Init() { }
  static constexpr Type get_type() { return Type::kExecutorConnectToPeersAck; }
};

static_assert(std::is_pod<ExecutorConnectToPeersAck>::value,
                "ExecutorConnectToPeersAck must be POD!");

struct ExecutorStop {
 private:
  ExecutorStop() = default;
  friend class DefaultMsgCreator;
 public:
  void Init() { }
  static constexpr Type get_type() { return Type::kExecutorStop; }
};
static_assert(std::is_pod<ExecutorStop>::value,
              "ExecutorStop must be POD!");

struct ExecutorIdentity {
  int32_t executor_id;
  HostInfo host_info;
 private:
  ExecutorIdentity() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(int32_t _executor_id, HostInfo _host_info) {
    executor_id = _executor_id;
    host_info = _host_info;
  }
  static constexpr Type get_type() { return Type::kExecutorIdentity; }
};

static_assert(std::is_pod<ExecutorIdentity>::value, "ExecutorIdentity must be POD!");

struct ServerIdentity {
  int32_t server_id;
  HostInfo host_info;
 private:
  ServerIdentity() = default;
  friend class DefaultMsgCreator;
 public:
  void Init(int32_t _server_id, HostInfo _host_info) {
    server_id = _server_id;
    host_info = _host_info;
  }
  static constexpr Type get_type() { return Type::kServerIdentity; }
};

static_assert(std::is_pod<ServerIdentity>::value, "ServerIdentity must be POD!");

enum class DriverMsgType {
  kStop = 0,
    kCallFuncOnOne = 3,
    kCallFuncOnAll = 4,
    kEvalExpr = 5,
    kCreateDistArray = 6,
    kMap = 7,
    kShuffle = 8,
    kMasterResponse = 9,
    kDefineVar = 10,
    kRepartitionDistArray = 11,
    kExecForLoop = 12,
    kGetAccumulatorValue = 13,
    kCreateDistArrayBuffer = 14,
    kSetDistArrayBufferInfo = 15,
    kDeleteDistArrayBufferInfo = 16,
    kUpdateDistArrayIndex = 17
};

struct DriverMsg {
  DriverMsgType driver_msg_type;
 private:
  DriverMsg() = default;
  friend class DefaultMsgCreator;
 public:
  void Init() { }
  static constexpr Type get_type() { return Type::kDriverMsg; }
};

static_assert(std::is_pod<DriverMsg>::value,
              "DriverMsg must be POD!");

enum class ExecuteMsgType {
  kExecutorAck = 1,
    kPeerRecvStop = 2,
    kJuliaEvalAck = 3,
    kCreateDistArray = 4,
    kTextFileLoadAck = 5,
    kDistArrayDims = 6,
    kPartitionNumLines = 7,
    kCreateDistArrayAck = 8,
    kCreateDistArrayBufferAck = 9,
    kRepartitionDistArrayData = 10,
    kRepartitionDistArrayRecved = 11,
    kRepartitionDistArrayMaxPartitionIds = 12,
    kRepartitionDistArrayAck = 13,
    kRequestExecForLoopGlobalIndexedDistArrays = 14,
    kReplyExecForLoopGlobalIndexedDistArrayData = 15,
    kRequestExecForLoopPipelinedTimePartitions = 16,
    kReplyExecForLoopPipelinedTimePartitions = 17,
    kRequestExecForLoopPredecessorCompletion = 18,
    kReplyExecForLoopPredecessorCompletion = 19,
    kPipelinedTimePartitions = 20,
    kReplyGetAccumulatorValue = 21,
    kRequestDistArrayValue = 22,
    kRequestDistArrayValues = 23,
    kReplyDistArrayValues = 24,
    kExecForLoopDone = 25,
    kExecForLoopAck = 26,
    kExecForLoopDistArrayBufferData = 27,
    kExecForLoopDistArrayCacheData = 28,
    kExecForLoopDistArrayBufferDataPtr = 29,
    kExecForLoopDistArrayCacheDataPtr = 30
};

struct ExecuteMsg {
  ExecuteMsgType execute_msg_type;
 private:
  ExecuteMsg() = default;
  friend class DefaultMsgCreator;
 public:
  void Init() { }
  static constexpr Type get_type() { return Type::kExecuteMsg; }
};

class DefaultMsgCreator {
 public:
  template<typename Msg, typename... Args>
  static Msg* CreateMsg(uint8_t* mem, Args... args) {
    auto msg = new (mem) Msg();
    msg->Init(args...);
    return msg;
  }
};

class Helper {
 public:
  template<typename Msg,
           typename MsgCreator = DefaultMsgCreator,
           typename... Args>
  static Msg* CreateMsg(conn::SendBuffer *send_buff,
                        Args... args) {
    // create Header
    uint8_t *payload_mem = send_buff->get_payload_mem();
    uint8_t *aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(payload_mem, alignof(Header)));
    auto header = new (aligned_mem) Header();
    header->type = Msg::get_type();
    send_buff->set_payload_size(aligned_mem + sizeof(Header) - payload_mem);

    // create Msg
    uint8_t *avai_mem = send_buff->get_avai_payload_mem();
    aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(avai_mem, alignof(Msg)));
    Msg* msg = MsgCreator::template CreateMsg<Msg, Args...>(aligned_mem,
                                                            args...);
    CHECK_GE(send_buff->get_payload_capacity(),
             aligned_mem - payload_mem + sizeof(Msg));
    send_buff->set_payload_size(aligned_mem - payload_mem + sizeof(Msg));
    return msg;
  }

  static Type get_type(conn::RecvBuffer& recv_buff) {
    uint8_t* payload_mem = recv_buff.get_payload_mem();
    uint8_t *aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(payload_mem, alignof(Header)));
    return reinterpret_cast<const Header*>(aligned_mem)->type;
  }

  template<typename Msg>
  static Msg* get_msg(conn::RecvBuffer &recv_buff) {
    uint8_t* payload_mem = recv_buff.get_payload_mem();
    uint8_t *aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(payload_mem, alignof(Header)));
    aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(aligned_mem + sizeof(Header), alignof(Msg)));
    return reinterpret_cast<Msg*>(aligned_mem);
  }

  template<typename Msg>
  static uint8_t* get_remaining_mem(conn::RecvBuffer &recv_buff) {
    uint8_t* payload_mem = recv_buff.get_payload_mem();
    uint8_t *aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(payload_mem, alignof(Header)));
    aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(aligned_mem + sizeof(Header), alignof(Msg)));

    return aligned_mem + sizeof(Msg);
  }

  template<typename Msg>
  static size_t get_remaining_size(conn::RecvBuffer &recv_buff) {
    uint8_t* payload_mem = recv_buff.get_payload_mem();
    uint8_t *aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(payload_mem, alignof(Header)));
    aligned_mem = reinterpret_cast<uint8_t*>(
        get_aligned(aligned_mem + sizeof(Header), alignof(Msg)));
    return recv_buff.get_payload_capacity()
        - (aligned_mem + sizeof(Msg) - recv_buff.get_payload_mem());
  }
};

} // end of namespace message

} // end of namespace bosen
}
