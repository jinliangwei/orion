#pragma once
#include <string>
#include <julia.h>
#include <glog/logging.h>

#include <orion/bosen/type.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/julia_task.hpp>
#include <orion/bosen/julia_module.hpp>

namespace orion {
namespace bosen {

class JuliaEvaluator {
 private:
  void UnboxResult(jl_value_t* value,
                   type::PrimitiveType result_type,
                   Blob *result_buff);

  jl_value_t* EvalExpr(const std::string &serialized_expr,
                       JuliaModule module);
  jl_module_t* orion_gen_module_;
  jl_module_t* orion_worker_module_;
 public:
  JuliaEvaluator() { }
  ~JuliaEvaluator() { }
  void Init(const std::string &orion_home);
  void AtExitHook() { jl_atexit_hook(0); }
  jl_value_t* EvalString(const std::string &code);
  void ExecuteTask(JuliaTask* task);
  inline jl_function_t* GetFunction(jl_module_t* module,
                                    const char* func_name);
  void ParseString(
      const char* str,
      jl_function_t *parser_func,
      type::PrimitiveType result_type,
      std::vector<int64_t> *key,
      Blob *value);

  void ParseStringValueOnly(
      const char* str,
      jl_function_t *parser_func,
      type::PrimitiveType result_type,
      Blob *value);
};

void
JuliaEvaluator::Init(const std::string &orion_home) {
  jl_init(NULL);

  jl_load((orion_home + "/src/julia/orion_gen.jl").c_str());
  orion_gen_module_ = reinterpret_cast<jl_module_t*>(
      jl_eval_string("OrionGen"));
  CHECK(orion_gen_module_ != nullptr);
  SetOrionGenModule(orion_gen_module_);

  jl_load((orion_home + "/src/julia/orion_worker.jl").c_str());
  orion_worker_module_ = reinterpret_cast<jl_module_t*>(
      jl_eval_string("OrionWorker"));
  CHECK(orion_worker_module_ != nullptr);
  SetOrionWorkerModule(orion_worker_module_);
}

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
        *reinterpret_cast<int8_t*>(result_buff->data()) = ret;
      }
      break;
    case type::PrimitiveType::kUInt8:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_uint8(value));
        uint8_t ret = jl_unbox_uint8(value);
        *reinterpret_cast<uint8_t*>(result_buff->data()) = ret;
      }
      break;
    case type::PrimitiveType::kInt16:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_int16(value));
        int16_t ret = jl_unbox_int16(value);
        *reinterpret_cast<int16_t*>(result_buff->data()) = ret;
      }
      break;
    case type::PrimitiveType::kUInt16:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_uint16(value));
        uint16_t ret = jl_unbox_uint16(value);
        *reinterpret_cast<uint16_t*>(result_buff->data()) = ret;
      }
      break;
    case type::PrimitiveType::kInt32:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_int32(value));
        int32_t ret = jl_unbox_int32(value);
        *reinterpret_cast<int32_t*>(result_buff->data()) = ret;
      }
      break;
    case type::PrimitiveType::kUInt32:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_uint32(value));
        uint32_t ret = jl_unbox_uint32(value);
        *reinterpret_cast<uint32_t*>(result_buff->data()) = ret;
      }
      break;
    case type::PrimitiveType::kInt64:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_int64(value));
        int64_t ret = jl_unbox_int64(value);
        *reinterpret_cast<int64_t*>(result_buff->data()) = ret;
      }
      break;
    case type::PrimitiveType::kUInt64:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_uint64(value));
        uint64_t ret = jl_unbox_uint64(value);
        *reinterpret_cast<uint64_t*>(result_buff->data()) = ret;
      }
      break;
    case type::PrimitiveType::kFloat32:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_float32(value));
        float ret = jl_unbox_float32(value);
        *reinterpret_cast<float*>(result_buff->data()) = ret;
      }
      break;
    case type::PrimitiveType::kFloat64:
      {
        result_buff->reserve(type::SizeOf(result_type));
        CHECK(jl_is_float64(value));
        double ret = jl_unbox_float64(value);
        *reinterpret_cast<double*>(result_buff->data()) = ret;
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

jl_value_t*
JuliaEvaluator::EvalExpr(const std::string &serialized_expr,
                         JuliaModule module) {
  jl_value_t *array_type, *serialized_expr_buff, *expr, *ret;
  jl_array_t *serialized_expr_array;
  JL_GC_PUSH5(&array_type, &serialized_expr_array, &serialized_expr_buff, &expr,
              &ret);

  array_type = jl_apply_array_type(jl_uint8_type, 1);
  std::vector<uint8_t> temp_serialized_expr(serialized_expr.size());
  memcpy(temp_serialized_expr.data(), serialized_expr.data(),
         serialized_expr.size());
  serialized_expr_array = jl_ptr_to_array_1d(array_type,
                                             temp_serialized_expr.data(),
                                             serialized_expr.size(), 0);
  jl_function_t *io_buffer_func
      = GetFunction(jl_base_module, "IOBuffer");
  serialized_expr_buff = jl_call1(io_buffer_func,
                                 reinterpret_cast<jl_value_t*>(serialized_expr_array));

  jl_function_t *deserialize_func
      = GetFunction(jl_base_module, "deserialize");
  expr = jl_call1(deserialize_func, serialized_expr_buff);

  jl_function_t *eval_func
      = GetFunction(jl_core_module, "eval");
  jl_module_t *jl_module = GetJlModule(module);
  ret = jl_call2(eval_func, reinterpret_cast<jl_value_t*>(jl_module), expr);
  JL_GC_POP();
  return ret;
}

jl_function_t*
JuliaEvaluator::GetFunction(jl_module_t* module,
                            const char* func_name) {
  auto* func = jl_get_function(module, func_name);
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
  } else if (auto exec_cpp_func_task = dynamic_cast<ExecCppFuncTask*>(task)) {
    exec_cpp_func_task->func(this);
  } else if (auto eval_expr_task = dynamic_cast<EvalJuliaExprTask*>(task)) {
    EvalExpr(eval_expr_task->serialized_expr,
             eval_expr_task->module);
    result_type = eval_expr_task->result_type,
    result_buff = &eval_expr_task->result_buff;
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
  JL_GC_PUSH5(&str_jl, &ret_tuple, &key_tuple,
              &key_ith, &value);

  str_jl = jl_cstr_to_string(str);
  CHECK(jl_is_string(str_jl));
  ret_tuple = jl_call1(parser_func, str_jl);
  CHECK(jl_is_tuple(ret_tuple));
  key_tuple = jl_get_nth_field(ret_tuple, 0);
  CHECK(jl_is_tuple(key_tuple)) << "key tuple is " << (void*) key_tuple;
  size_t num_dims = key->size();
  for (size_t i = 0; i < num_dims; i++) {
    key_ith = jl_get_nth_field(key_tuple, i);
    (*key)[i] = jl_unbox_int64(key_ith);
  }
  value = jl_get_nth_field(ret_tuple, 1);
  CHECK(jl_is_float64(value)) << "value ptr is " << (void*) value;
  UnboxResult(value, result_type, value_buff);
  JL_GC_POP();
}

void
JuliaEvaluator::ParseStringValueOnly(
    const char *str,
    jl_function_t *parser_func,
    type::PrimitiveType result_type,
    Blob *value_buff) {
  jl_value_t *str_jl = nullptr;
  jl_value_t *ret_tuple = nullptr;
  jl_value_t *value = nullptr;
  JL_GC_PUSH3(&str_jl, &ret_tuple, &value);

  str_jl = jl_cstr_to_string(str);
  CHECK(jl_is_string(str_jl));
  ret_tuple = jl_call1(parser_func, str_jl);
  CHECK(jl_is_tuple(ret_tuple));
  value = jl_get_nth_field(ret_tuple, 0);
  CHECK(jl_is_float64(value)) << "value ptr is " << (void*) value;
  UnboxResult(value, result_type, value_buff);
  JL_GC_POP();
}

}
}
