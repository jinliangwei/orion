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
  jl_value_t* EvalString(const std::string &code);
  void ExecuteTask(JuliaTask* task);
  inline jl_function_t* GetFunction(const std::string &func_name);
  void ParseString(
      const char* str,
      jl_function_t *parser_func,
      type::PrimitiveType result_type,
      std::vector<int64_t> *key,
      Blob *value);
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

jl_function_t*
JuliaEvaluator::GetFunction(const std::string &func_name) {
  auto* func = jl_get_function(jl_main_module, func_name.c_str());
  CHECK(func != nullptr) << "func_name = " << func_name;
  return func;
}

jl_value_t*
JuliaEvaluator::EvalString(const std::string &code) {
  return jl_eval_string(code.c_str());
}

void
JuliaEvaluator::ExecuteTask(JuliaTask* task) {
  jl_value_t* ret = nullptr;
  type::PrimitiveType result_type = type::PrimitiveType::kVoid;
  Blob* result_buff = nullptr;
  if (auto exec_code_task = dynamic_cast<ExecJuliaCodeTask*>(task)) {
    ret = EvalString(exec_code_task->code);
    result_type = exec_code_task->result_type;
    result_buff = &exec_code_task->result_buff;
    //} else if (auto call_func_task = dynamic_cast<JuliaCallFuncTask*>(task)) {
  } else if (auto exec_cpp_func_task = dynamic_cast<ExecCppFuncTask*>(task)) {
    exec_cpp_func_task->func(this);
  } else {
    LOG(FATAL) << "Unknown task type!";
  }

  UnboxResult(ret, result_type, result_buff);
}


void
JuliaEvaluator::ParseString(
    const char *str,
    jl_function_t *parser_func,
    type::PrimitiveType result_type,
    std::vector<int64_t> *key,
    Blob *value_buff) {
  jl_value_t *str_jl = nullptr;
  jl_value_t *ret_tuple = nullptr;
  jl_value_t *key_tuple = nullptr;
  jl_value_t *key_ith = nullptr;
  jl_value_t *value = nullptr;
  //JL_GC_PUSH5(&str_jl, &ret_tuple, &key_tuple,
  //            &key_ith, &value);


  str_jl = jl_cstr_to_string(str);
  CHECK(jl_is_string(str_jl));
  ret_tuple = jl_call1(parser_func, str_jl);
  CHECK(jl_is_tuple(ret_tuple));
  key_tuple = jl_get_nth_field(ret_tuple, 1);
  CHECK(jl_is_tuple(key_tuple));
  size_t num_dims = key->size();
  for (size_t i = 0; i < num_dims; i++) {
    key_ith = jl_get_nth_field(key_tuple, i + 1);
    (*key)[i] = jl_unbox_int64(key_ith);
  }
  value = jl_get_nth_field(ret_tuple, 2);
  UnboxResult(value, result_type, value_buff);
  //JL_GC_POP();
  //JL_GC_POP();
  //JL_GC_POP();
  //JL_GC_POP();
  //JL_GC_POP();
}

}
}
