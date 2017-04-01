#pragma once
#include <string>
#include <julia.h>
#include <glog/logging.h>

#include <orion/bosen/type.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/julia_task.hpp>

namespace orion {
namespace bosen {

class JuliaEvaluator {
 private:
  void UnboxResult(jl_value_t* value,
                   type::PrimitiveType result_type,
                   Blob *result_buff);
 public:
  JuliaEvaluator() { }
  ~JuliaEvaluator() { }
  void Init() { jl_init(NULL); }
  void AtExitHook() { jl_atexit_hook(0); }
  void ExecuteTask(JuliaTask* task);
};

void
JuliaEvaluator::UnboxResult(jl_value_t* value,
                            type::PrimitiveType result_type,
                            Blob *result_buff) {
  switch (result_type) {
    case type::PrimitiveType::kVoid:
      break;
    case type::PrimitiveType::kInt8:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_int8(value));
        int8_t ret = jl_unbox_int8(value);
        memcpy(result_buff->data(), &ret, sizeof(ret));
      }
      break;
    case type::PrimitiveType::kUInt8:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_uint8(value));
        uint8_t ret = jl_unbox_uint8(value);
        memcpy(result_buff->data(), &ret, sizeof(ret));
      }
      break;
    case type::PrimitiveType::kInt16:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_int16(value));
        int16_t ret = jl_unbox_int16(value);
        memcpy(result_buff->data(), &ret, sizeof(ret));
      }
      break;
    case type::PrimitiveType::kUInt16:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_uint16(value));
        uint16_t ret = jl_unbox_uint16(value);
        memcpy(result_buff->data(), &ret, sizeof(ret));
      }
      break;
    case type::PrimitiveType::kInt32:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_int32(value));
        int32_t ret = jl_unbox_int32(value);
        memcpy(result_buff->data(), &ret, sizeof(ret));
      }
      break;
    case type::PrimitiveType::kUInt32:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_uint32(value));
        uint32_t ret = jl_unbox_uint32(value);
        memcpy(result_buff->data(), &ret, sizeof(ret));
      }
      break;
    case type::PrimitiveType::kInt64:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_int64(value));
        int64_t ret = jl_unbox_int64(value);
        memcpy(result_buff->data(), &ret, sizeof(ret));
      }
      break;
    case type::PrimitiveType::kUInt64:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_uint64(value));
        uint64_t ret = jl_unbox_uint64(value);
        memcpy(result_buff->data(), &ret, sizeof(ret));
      }
      break;
    case type::PrimitiveType::kFloat32:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_float32(value));
        float ret = jl_unbox_float32(value);
        memcpy(result_buff->data(), &ret, sizeof(ret));
      }
      break;
    case type::PrimitiveType::kFloat64:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_float64(value));
        double ret = jl_unbox_float64(value);
        LOG(INFO) << "unboxed float64 " << ret;
        memcpy(result_buff->data(), &ret, sizeof(ret));
      }
      break;
    case type::PrimitiveType::kString:
      {
        LOG(FATAL) << "Returning string type is currently not supported";
      }
      break;
    default:
      LOG(FATAL) << "Unknown primitive type";
  }
}

void
JuliaEvaluator::ExecuteTask(JuliaTask* task) {
  jl_value_t* ret = nullptr;
  type::PrimitiveType result_type = type::PrimitiveType::kVoid;
  Blob *result_buff = nullptr;
  if (auto execute_code_task = dynamic_cast<ExecJuliaCodeTask*>(task)) {
    ret = jl_eval_string(execute_code_task->code.c_str());
    result_type = execute_code_task->result_type;
    result_buff = &execute_code_task->result_buff;
    //} else if (auto call_func_task = dynamic_cast<JuliaCallFuncTask*>(task)) {

  } else {
    LOG(FATAL) << "Unknown task type!";
  }

  UnboxResult(ret, result_type, result_buff);
}


}
}
