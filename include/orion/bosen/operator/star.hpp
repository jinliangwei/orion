#include <orion/bosen/operators/headers.hpp>

namespace orion {
namespace bosen {
namespace operators {

struct StarTask {
  size_t task_size_;
 private:
  StarTask() = default;
  static void CreateStarTask(uint8_t* mem) {
    *reinterpret_cast<Header*>(mem) = Header::kStar;
    new (mem + sizeof(Header)) StarTask;
  }
};

class Star {
 private:
  conn::SocketConn& driver_;
  conn::SendBuffer& send_buff_;
  const int32_t& my_id_;
  WorkerRuntime* worker_runtime_;

 public:
  Star(conn::SocketConn& driver,
       conn::SendBuffer& send_buff,
       const int32_t& my_id,
       WorkerRuntime* worker_runtime):
      driver_(driver),
      send_buff_(send_buff),
      my_id_(my_id),
      worker_runtime_(worker_runtime) { }

  void ParseExecuteTask(const void* msg);
};

void
Star::ParseExecuteTask(const void* msg) {

}

}
}
}
