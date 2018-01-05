#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <orion/bosen/type.hpp>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/blob.hpp>
#include <julia.h>

namespace orion {
namespace bosen {

class JuliaThreadRequester {

 private:
  conn::Pipe &write_pipe_;
  std::vector<uint8_t> &send_mem_;
  conn::SendBuffer &send_buff_;
  conn::Poll &poll_;
  const size_t kNumEvents;
  epoll_event *es_;
  bool request_replied_ { false };
  Blob requested_value_;
  std::mutex request_mtx_;
  std::condition_variable request_cv_;
 public:
  JuliaThreadRequester(conn::Pipe &write_pipe,
                       std::vector<uint8_t> &send_mem,
                       conn::SendBuffer &send_buff,
                       conn::Poll &poll,
                       size_t num_events,
                       epoll_event* es): write_pipe_(write_pipe),
                                         send_mem_(send_mem),
                                         send_buff_(send_buff),
                                         poll_(poll),
                                         kNumEvents(num_events),
                                         es_(es) { }
  ~JuliaThreadRequester() { }

  void RequestDistArrayData(
      int32_t dist_array_id,
      int64_t key,
      type::PrimitiveType value_type,
      jl_value_t **value);

  void ReplyDistArrayData(
      const uint8_t* bytes,
      size_t num_bytes);
};

}
}
