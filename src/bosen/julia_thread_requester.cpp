#include <orion/bosen/julia_thread_requester.hpp>
#include <orion/bosen/julia_evaluator.hpp>
#include <orion/bosen/execute_message.hpp>
#include <julia.h>

namespace orion {
namespace bosen {

void
JuliaThreadRequester::RequestDistArrayData(
    int32_t dist_array_id,
    int64_t key,
    type::PrimitiveType value_type,
    jl_value_t **value) {
  message::ExecuteMsgHelper::CreateMsg<
    message::ExecuteMsgRequestDistArrayValue>(&send_buff_, dist_array_id, key,
                                              kMyExecutorOrServerId,
                                              kIsExecutor);
  bool sent = write_pipe_.Send(&send_buff_);
  if (!sent) {
    int ret = poll_.Add(write_pipe_.get_write_fd(), &write_pipe_, EPOLLOUT);
    CHECK_EQ(ret, 0);
    while (!sent) {
      int num_events = poll_.Wait(es_, kNumEvents);
      CHECK(num_events > 0);
      CHECK(es_[0].events & EPOLLOUT);
      CHECK(conn::Poll::EventConn<conn::Pipe>(es_, 0) == &write_pipe_);
      sent = write_pipe_.Send(&send_buff_);
    }
    poll_.Remove(write_pipe_.get_write_fd());
  }
  send_buff_.reset_sent_sizes();
  send_buff_.clear_to_send();

  std::unique_lock<std::mutex> lock(request_mtx_);
  request_cv_.wait(lock, [this]{ return this->request_replied_; });

  if (requested_value_.size() == 0) {
    *value = jl_nothing;
    return;
  }

  if (value_type == type::PrimitiveType::kVoid) {
    jl_value_t* buff_jl = nullptr;
    jl_value_t* serialized_value_array = nullptr;
    jl_value_t *serialized_value_array_type = nullptr;
    JL_GC_PUSH3(&buff_jl, &serialized_value_array,
                &serialized_value_array_type);
    serialized_value_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_uint8_type), 1);
    jl_function_t *io_buffer_func
        = JuliaEvaluator::GetFunction(jl_base_module, "IOBuffer");
    jl_function_t *deserialize_func
        = JuliaEvaluator::GetFunction(jl_base_module, "deserialize");

    serialized_value_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
        serialized_value_array_type,
        requested_value_.data(),
        requested_value_.size(), 0));
    buff_jl = jl_call1(io_buffer_func, serialized_value_array);
    *value = jl_call1(deserialize_func, buff_jl);
    JL_GC_POP();
  } else if (value_type == type::PrimitiveType::kString) {
    *value = jl_cstr_to_string(reinterpret_cast<char*>(requested_value_.data()));
  } else {
    JuliaEvaluator::BoxValue(value_type, requested_value_.data(), value);
  }
  requested_value_.resize(0);
}

void
JuliaThreadRequester::ReplyDistArrayData(
    const uint8_t* bytes,
    size_t num_bytes) {
  std::unique_lock<std::mutex> lock(request_mtx_);
  requested_value_.resize(num_bytes);
  memcpy(requested_value_.data(), bytes, num_bytes);
  request_replied_ = true;
  request_cv_.notify_one();
}

}
}
