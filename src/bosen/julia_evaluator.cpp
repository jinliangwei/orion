#include <orion/bosen/julia_evaluator.hpp>
#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/julia_module.hpp>
namespace orion {
namespace bosen {

jl_module_t *JuliaEvaluator::orion_worker_module_ = nullptr;
std::string JuliaEvaluator::lib_path_;
std::string JuliaEvaluator::orion_home_;

void
JuliaEvaluator::Init(const std::string &orion_home,
                     size_t num_servers,
                     size_t num_executors) {
  jl_init(NULL);
  orion_home_ = orion_home;
  lib_path_ = orion_home + "/lib/liborion.so";
  jl_load((orion_home + "/src/julia/orion_worker.jl").c_str());
  orion_worker_module_ = reinterpret_cast<jl_module_t*>(
      jl_eval_string("OrionWorker"));
  CHECK(orion_worker_module_ != nullptr);
  SetOrionWorkerModule(orion_worker_module_);

  jl_value_t *lib_path_str = nullptr;
  jl_value_t *num_servers_jl = nullptr;
  jl_value_t *num_executors_jl = nullptr;
  JL_GC_PUSH1(&lib_path_str);
  lib_path_str = jl_cstr_to_string(lib_path_.c_str());
  jl_function_t *set_lib_path_func
      = GetFunction(orion_worker_module_, "set_lib_path");
  jl_call1(set_lib_path_func, lib_path_str);
  jl_function_t *worker_init_func
      = GetFunction(orion_worker_module_, "worker_init");
  num_servers_jl = jl_box_uint64(num_servers);
  num_executors_jl = jl_box_uint64(num_executors);
  jl_call2(worker_init_func, num_servers_jl, num_executors_jl);
  JL_GC_POP();
}

void
JuliaEvaluator::BoxValue(
    type::PrimitiveType result_type,
    const uint8_t* value,
    jl_value_t **value_jl) {
  switch (result_type) {
    case type::PrimitiveType::kVoid:
      break;
    case type::PrimitiveType::kInt8:
      {
        *value_jl = jl_box_int8(*reinterpret_cast<const int8_t*>(value));
      }
      break;
    case type::PrimitiveType::kUInt8:
      {
        *value_jl = jl_box_uint8(*reinterpret_cast<const uint8_t*>(value));
      }
      break;
    case type::PrimitiveType::kInt16:
      {
        *value_jl = jl_box_int16(*reinterpret_cast<const int16_t*>(value));
      }
      break;
    case type::PrimitiveType::kUInt16:
      {
        *value_jl = jl_box_uint16(*reinterpret_cast<const uint16_t*>(value));
      }
      break;
    case type::PrimitiveType::kInt32:
      {
        *value_jl = jl_box_int32(*reinterpret_cast<const int32_t*>(value));
      }
      break;
    case type::PrimitiveType::kUInt32:
      {
        *value_jl = jl_box_uint32(*reinterpret_cast<const uint32_t*>(value));
      }
      break;
    case type::PrimitiveType::kInt64:
      {
        *value_jl = jl_box_int64(*reinterpret_cast<const int64_t*>(value));
      }
      break;
    case type::PrimitiveType::kUInt64:
      {
        *value_jl = jl_box_uint64(*reinterpret_cast<const uint64_t*>(value));
      }
      break;
    case type::PrimitiveType::kFloat32:
      {
        *value_jl = jl_box_float32(*reinterpret_cast<const float*>(value));
      }
      break;
    case type::PrimitiveType::kFloat64:
      {
        *value_jl = jl_box_float32(*reinterpret_cast<const double*>(value));
      }
      break;
    case type::PrimitiveType::kString:
      {
        *value_jl = jl_cstr_to_string(reinterpret_cast<const char*>(value));
      }
      break;
    default:
      LOG(FATAL) << "Unknown primitive type";
  }
}

void
JuliaEvaluator::UnboxValue(jl_value_t* value,
                           type::PrimitiveType result_type,
                           uint8_t* value_buff) {
  switch (result_type) {
    case type::PrimitiveType::kVoid:
      break;
    case type::PrimitiveType::kInt8:
      {
        CHECK(jl_is_int8(value));
        int8_t ret = jl_unbox_int8(value);
        *reinterpret_cast<int8_t*>(value_buff) = ret;
      }
      break;
    case type::PrimitiveType::kUInt8:
      {
        CHECK(jl_is_uint8(value));
        uint8_t ret = jl_unbox_uint8(value);
        *reinterpret_cast<uint8_t*>(value_buff) = ret;
      }
      break;
    case type::PrimitiveType::kInt16:
      {
        CHECK(jl_is_int16(value));
        int16_t ret = jl_unbox_int16(value);
        *reinterpret_cast<int16_t*>(value_buff) = ret;
      }
      break;
    case type::PrimitiveType::kUInt16:
      {
        CHECK(jl_is_uint16(value));
        uint16_t ret = jl_unbox_uint16(value);
        *reinterpret_cast<uint16_t*>(value_buff) = ret;
      }
      break;
    case type::PrimitiveType::kInt32:
      {
        CHECK(jl_is_int32(value));
        int32_t ret = jl_unbox_int32(value);
        *reinterpret_cast<int32_t*>(value_buff) = ret;
      }
      break;
    case type::PrimitiveType::kUInt32:
      {
        CHECK(jl_is_uint32(value));
        uint32_t ret = jl_unbox_uint32(value);
        *reinterpret_cast<uint32_t*>(value_buff) = ret;
      }
      break;
    case type::PrimitiveType::kInt64:
      {
        CHECK(jl_is_int64(value));
        int64_t ret = jl_unbox_int64(value);
        *reinterpret_cast<int64_t*>(value_buff) = ret;
      }
      break;
    case type::PrimitiveType::kUInt64:
      {
        CHECK(jl_is_uint64(value));
        uint64_t ret = jl_unbox_uint64(value);
        *reinterpret_cast<uint64_t*>(value_buff) = ret;
      }
      break;
    case type::PrimitiveType::kFloat32:
      {
        CHECK(jl_is_float32(value));
        float ret = jl_unbox_float32(value);
        *reinterpret_cast<float*>(value_buff) = ret;
      }
      break;
    case type::PrimitiveType::kFloat64:
      {
        CHECK(jl_is_float64(value));
        double ret = jl_unbox_float64(value);
        *reinterpret_cast<double*>(value_buff) = ret;
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
JuliaEvaluator::EvalExpr(const std::string &serialized_expr,
                         JuliaModule module,
                         Blob *result_buff) {
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 7);
  jl_value_t* &array_type = jl_values[0];
  jl_value_t* &serialized_expr_buff = jl_values[1];
  jl_value_t* &expr = jl_values[2];
  jl_value_t* &ret = jl_values[3];
  jl_value_t* &serialized_expr_array = jl_values[4];
  jl_value_t* &buff = jl_values[5];
  jl_value_t* &serialized_result_array = jl_values[6];

  array_type = jl_apply_array_type(jl_uint8_type, 1);
  std::vector<uint8_t> temp_serialized_expr(serialized_expr.size());
  memcpy(temp_serialized_expr.data(), serialized_expr.data(),
         serialized_expr.size());
  serialized_expr_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      array_type,
      temp_serialized_expr.data(),
      serialized_expr.size(), 0));

  jl_function_t *io_buffer_func
      = GetFunction(jl_base_module, "IOBuffer");
  CHECK(io_buffer_func != nullptr);
  serialized_expr_buff = jl_call1(io_buffer_func,
                                  reinterpret_cast<jl_value_t*>(serialized_expr_array));
  jl_function_t *deserialize_func
      = GetFunction(jl_base_module, "deserialize");
  CHECK(deserialize_func != nullptr);
  expr = jl_call1(deserialize_func, serialized_expr_buff);
  AbortIfException();
  jl_function_t *eval_func = GetFunction(jl_core_module, "eval");
  jl_module_t *jl_module = GetJlModule(module);
  CHECK(jl_module != nullptr);
  ret = jl_call2(eval_func, reinterpret_cast<jl_value_t*>(jl_module), expr);
  AbortIfException();
  buff = jl_call0(io_buffer_func);

  jl_function_t *serialize_func
      = GetFunction(jl_base_module, "serialize");
  CHECK(serialize_func != nullptr);
  jl_call2(serialize_func, buff, ret);

  jl_function_t *takebuff_array_func
      = GetFunction(jl_base_module, "takebuf_array");
  serialized_result_array = jl_call1(takebuff_array_func, buff);
  size_t result_array_length = jl_array_len(serialized_result_array);
  uint8_t* array_bytes = reinterpret_cast<uint8_t*>(jl_array_data(serialized_result_array));
  result_buff->resize(result_array_length);
  memcpy(result_buff->data(), array_bytes, result_array_length);
  JL_GC_POP();
  AbortIfException();
}

jl_function_t*
JuliaEvaluator::GetOrionWorkerFunction(const char* func_name) {
  auto* func = jl_get_function(orion_worker_module_, func_name);
  CHECK(func != nullptr) << "func_name = " << func_name;
  return func;
}

jl_function_t*
JuliaEvaluator::GetFunction(jl_module_t* module,
                            const char* func_name) {
  auto* func = jl_get_function(module, func_name);
  CHECK(func != nullptr) << "func_name = " << func_name;
  return func;
}

void
JuliaEvaluator::AbortIfException() {
  jl_value_t *exception_jl = jl_exception_occurred();
  JL_GC_PUSH1(&exception_jl);
  if (exception_jl) {
    jl_function_t *show_error_func
        = GetFunction(jl_base_module, "showerror");
    jl_call2(show_error_func, jl_stdout_obj(), exception_jl);
    LOG(FATAL) << "julia exception occurs: " << jl_typeof_str(exception_jl)
               << " " << jl_typename_str(exception_jl);
  }
  JL_GC_POP();
}

void
JuliaEvaluator::ExecuteTask(JuliaTask* task) {
  if (auto exec_cpp_func_task = dynamic_cast<ExecCppFuncTask*>(task)) {
    exec_cpp_func_task->func();
  } else if (auto eval_expr_task = dynamic_cast<EvalJuliaExprTask*>(task)) {
    Blob *result_buff = &eval_expr_task->result_buff;
    EvalExpr(eval_expr_task->serialized_expr,
             eval_expr_task->module,
             result_buff);
  } else {
    LOG(FATAL) << "Unknown task type!";
  }
}

void
JuliaEvaluator::ParseString(
    const char *str,
    JuliaModule parser_func_module,
    const char *parser_func_name,
    std::vector<int64_t> *key, // key memory is pre-allocated
    size_t num_dims,
    jl_value_t **value_ptr) {
  jl_function_t* parser_func = GetFunction(GetJlModule(parser_func_module),
                                          parser_func_name);
  jl_value_t *str_jl = nullptr;
  jl_value_t *ret_tuple = nullptr;
  jl_value_t *key_tuple = nullptr;
  jl_value_t *key_ith = nullptr;
  JL_GC_PUSH4(&str_jl, &ret_tuple, &key_tuple, &key_ith);
  str_jl = jl_cstr_to_string(str);
  ret_tuple = jl_call1(parser_func, str_jl);
  CHECK(jl_is_tuple(ret_tuple));
  key_tuple = jl_get_nth_field(ret_tuple, 0);
  CHECK_EQ(key->size(), num_dims);
  for (size_t i = 0; i < num_dims; i++) {
    key_ith = jl_get_nth_field(key_tuple, i);
    (*key)[i] = jl_unbox_int64(key_ith);
    }
  *value_ptr = jl_get_nth_field(ret_tuple, 1);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::ParseStringFlatten(
    const char *str,
    JuliaModule parser_func_module,
    const char *parser_func_name,
    std::vector<int64_t> *key,
    size_t num_dims,
    jl_value_t *value_type,
    jl_value_t **value_ptr) {
  jl_function_t* parser_func = GetFunction(GetJlModule(parser_func_module),
                                           parser_func_name);
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 7);
  jl_value_t* &str_jl = jl_values[0];
  jl_value_t* &ret_array = jl_values[1];
  jl_value_t* &value_array_type = jl_values[2];
  jl_value_t* &key_value = jl_values[3];
  jl_value_t* &key_tuple = jl_values[4];
  jl_value_t* &key_ith = jl_values[5];
  jl_value_t* &value_element = jl_values[6];
  jl_value_t* &value_array = *value_ptr;

  str_jl = jl_cstr_to_string(str);
  CHECK(jl_is_string(str_jl));
  ret_array = jl_call1(parser_func, str_jl);
  CHECK(jl_is_array(ret_array));

  value_array_type = jl_apply_array_type(reinterpret_cast<jl_datatype_t*>(value_type), 1);

  size_t num_output_values = jl_array_len(ret_array);
  value_array = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(
      value_array_type,
      num_output_values));

  key->clear();
  for (size_t i = 0; i < num_output_values; i++) {
    key_value = jl_arrayref(reinterpret_cast<jl_array_t*>(ret_array), i);
    key_tuple = jl_get_nth_field(key_value, 0);
    for (size_t i = 0; i < num_dims; i++) {
      key_ith = jl_get_nth_field(key_tuple, i);
      int64_t key_int64 = jl_unbox_int64(key_ith);
      key->push_back(key_int64);
    }
    value_element = jl_get_nth_field(key_value, 1);
    jl_arrayset(reinterpret_cast<jl_array_t*>(value_array), value_element, i);
  }
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::ParseStringWithLineNumber(
    size_t line_number,
    const char *str,
    JuliaModule parser_func_module,
    const char *parser_func_name,
    std::vector<int64_t> *key, // key memory is pre-allocated
    size_t num_dims,
    jl_value_t **value_ptr) {
  jl_function_t* parser_func = GetFunction(GetJlModule(parser_func_module),
                                           parser_func_name);
  jl_value_t *str_jl = nullptr;
  jl_value_t *ret_tuple = nullptr;
  jl_value_t *key_tuple = nullptr;
  jl_value_t *key_ith = nullptr;
  jl_value_t *line_number_jl = nullptr;
  JL_GC_PUSH5(&str_jl, &ret_tuple, &key_tuple, &key_ith,
              &line_number_jl);
  line_number_jl = jl_box_int64(line_number);

  str_jl = jl_cstr_to_string(str);
  CHECK(jl_is_string(str_jl));
  ret_tuple = jl_call2(parser_func, line_number_jl, str_jl);
  AbortIfException();

  CHECK(jl_is_tuple(ret_tuple));
  key_tuple = jl_get_nth_field(ret_tuple, 0);
  CHECK_EQ(key->size(), num_dims);
  for (size_t i = 0; i < num_dims; i++) {
    key_ith = jl_get_nth_field(key_tuple, i);
    (*key)[i] = jl_unbox_int64(key_ith);
  }
  *value_ptr = jl_get_nth_field(ret_tuple, 1);
  JL_GC_POP();
}

void
JuliaEvaluator::ParseStringWithLineNumberFlatten(
    size_t line_number,
    const char *str,
    JuliaModule parser_func_module,
    const char *parser_func_name,
    std::vector<int64_t> *key,
    size_t num_dims,
    jl_value_t *value_type,
    jl_value_t **value_ptr) {
  jl_function_t* parser_func = GetFunction(GetJlModule(parser_func_module),
                                           parser_func_name);
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 8);
  jl_value_t* &str_jl = jl_values[0];
  jl_value_t* &ret_array = jl_values[1];
  jl_value_t* &value_array_type = jl_values[2];
  jl_value_t* &key_value = jl_values[3];
  jl_value_t* &key_tuple = jl_values[4];
  jl_value_t* &key_ith = jl_values[5];
  jl_value_t* &value_element = jl_values[6];
  jl_value_t* &value_array = *value_ptr;
  jl_value_t* &line_number_jl = jl_values[7];
  line_number_jl = jl_box_int64(line_number);

  str_jl = jl_cstr_to_string(str);
  CHECK(jl_is_string(str_jl));
  ret_array = jl_call2(parser_func, line_number_jl, str_jl);
  CHECK(jl_is_array(ret_array));

  value_array_type = jl_apply_array_type(reinterpret_cast<jl_datatype_t*>(value_type), 1);

  size_t num_output_values = jl_array_len(ret_array);
  value_array = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(
      value_array_type,
      num_output_values));

  key->clear();
  for (size_t i = 0; i < num_output_values; i++) {
    key_value = jl_arrayref(reinterpret_cast<jl_array_t*>(ret_array), i);
    key_tuple = jl_get_nth_field(key_value, 0);
    for (size_t i = 0; i < num_dims; i++) {
      key_ith = jl_get_nth_field(key_tuple, i);
      int64_t key_int64 = jl_unbox_int64(key_ith);
      key->push_back(key_int64);
    }
    value_element = jl_get_nth_field(key_value, 1);
    jl_arrayset(reinterpret_cast<jl_array_t*>(ret_array), value_element, i);
  }
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::ParseStringValueOnly(
    const char *str,
    JuliaModule parser_func_module,
    const char *parser_func_name,
    size_t num_dims,
    jl_value_t **value_ptr) {
  jl_function_t* parser_func = GetFunction(GetJlModule(parser_func_module),
                                           parser_func_name);

  jl_value_t *str_jl = nullptr;
  JL_GC_PUSH1(&str_jl);

  str_jl = jl_cstr_to_string(str);
  CHECK(jl_is_string(str_jl));
  *value_ptr = jl_call1(parser_func, str_jl);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::ParseStringValueOnlyWithLineNumber(
    size_t line_number,
    const char *str,
    JuliaModule parser_func_module,
    const char *parser_func_name,
    size_t num_dims,
    jl_value_t **value_ptr) {
  jl_function_t* parser_func = GetFunction(GetJlModule(parser_func_module),
                                           parser_func_name);
  jl_value_t* str_jl = nullptr;
  jl_value_t* line_number_jl = nullptr;
  JL_GC_PUSH2(&str_jl, &line_number_jl);
  line_number_jl = jl_box_int64(line_number);

  str_jl = jl_cstr_to_string(str);
  CHECK(jl_is_string(str_jl));
  *value_ptr = jl_call2(parser_func, line_number_jl, str_jl);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::DefineDistArray(
    int32_t id,
    const std::string &symbol,
    const std::string &serialized_value_type,
    const std::vector<int64_t> &dims,
    bool is_dense,
    bool is_buffer,
    const std::vector<uint8_t> &serialized_init_value) {

  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 13);
  jl_value_t* &symbol_jl = jl_values[0];
  jl_value_t* &value_type_jl = jl_values[1];
  jl_value_t* &dims_vec_jl = jl_values[2];
  jl_value_t* &is_dense_jl = jl_values[3];
  jl_value_t* &dims_vec_array_type_jl = jl_values[4];
  jl_value_t* &serialized_value_type_buff = jl_values[5];
  jl_value_t* &serialized_value_type_array = jl_values[6];
  jl_value_t* &serialized_value_type_array_type = jl_values[7];
  jl_value_t* &is_buffer_jl = jl_values[8];
  jl_value_t* &serialized_init_value_buff = jl_values[9];
  jl_value_t* &serialized_init_value_array = jl_values[10];
  jl_value_t* &init_value_jl = jl_values[11];
  jl_value_t* &id_jl = jl_values[12];

  serialized_value_type_array_type = jl_apply_array_type(jl_uint8_type, 1);
  std::vector<uint8_t> temp_serialized_value_type(serialized_value_type.size());
  memcpy(temp_serialized_value_type.data(),
         serialized_value_type.data(),
         serialized_value_type.size());
  serialized_value_type_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      serialized_value_type_array_type,
      temp_serialized_value_type.data(),
      temp_serialized_value_type.size(), 0));

  jl_function_t *io_buffer_func = GetFunction(jl_base_module, "IOBuffer");
  CHECK(io_buffer_func != nullptr);
  serialized_value_type_buff = jl_call1(io_buffer_func,
                                        reinterpret_cast<jl_value_t*>(serialized_value_type_array));
  jl_function_t *deserialize_func
      = GetFunction(jl_base_module, "deserialize");
  value_type_jl = jl_call1(deserialize_func, serialized_value_type_buff);
  AbortIfException();

  symbol_jl = jl_cstr_to_string(symbol.c_str());
  dims_vec_array_type_jl = jl_apply_array_type(jl_int64_type, 1);
  std::vector<int64_t> tmp_dims = dims;
  dims_vec_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(dims_vec_array_type_jl,
                         tmp_dims.data(), tmp_dims.size(), 0));
  is_dense_jl = jl_box_bool(is_dense);

  if (is_buffer) {
    auto temp_serialized_init_value = serialized_init_value;
    serialized_init_value_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
        serialized_value_type_array_type,
        temp_serialized_init_value.data(),
        temp_serialized_init_value.size(), 0));
    serialized_init_value_buff = jl_call1(io_buffer_func,
                                          reinterpret_cast<jl_value_t*>(serialized_init_value_array));
    init_value_jl = jl_call1(deserialize_func, serialized_init_value_buff);
  } else {
    init_value_jl = jl_nothing;
  }
  AbortIfException();
  is_buffer_jl = jl_box_bool(is_buffer);
  id_jl = jl_box_int32(id);
  jl_function_t *create_dist_array_func = GetFunction(
      jl_main_module, "orionres_define_dist_array");
  jl_value_t* args[7];
  args[0] = id_jl;
  args[1] = value_type_jl;
  args[2] = symbol_jl;
  args[3] = dims_vec_jl;
  args[4] = is_dense_jl;
  args[5] = is_buffer_jl;
  args[6] = init_value_jl;

  jl_call(create_dist_array_func, args, 7);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::RandNormal(
    type::PrimitiveType value_type,
    jl_value_t **values_ptr,
    size_t num_values) {
  jl_value_t *rand_seed_jl = nullptr;
  jl_value_t *num_values_jl = nullptr;
  JL_GC_PUSH2(&rand_seed_jl, &num_values_jl);
  rand_seed_jl = jl_box_int32(1);
  num_values_jl = jl_box_uint64(num_values);

  jl_function_t *srand_func = GetFunction(jl_base_module, "srand");
  jl_call1(srand_func, rand_seed_jl);

  jl_function_t *rand_func = GetFunction(jl_base_module, "randn");
  jl_datatype_t *value_type_jl = type::GetJlDataType(value_type);
  *values_ptr = jl_call2(rand_func, reinterpret_cast<jl_value_t*>(value_type_jl),
                         num_values_jl);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::RandUniform(
    type::PrimitiveType value_type,
    jl_value_t **values_ptr,
    size_t num_values) {
  jl_value_t *rand_seed_jl = nullptr;
  jl_value_t *num_values_jl = nullptr;
  JL_GC_PUSH2(&rand_seed_jl, &num_values_jl);
  rand_seed_jl = jl_box_int32(1);
  num_values_jl = jl_box_uint64(num_values);

  jl_function_t *srand_func = GetFunction(jl_base_module, "srand");
  jl_call1(srand_func, rand_seed_jl);

  jl_function_t *rand_func = GetFunction(jl_base_module, "rand");
  jl_datatype_t *value_type_jl = type::GetJlDataType(value_type);
  *values_ptr = jl_call2(rand_func, reinterpret_cast<jl_value_t*>(value_type_jl),
                         num_values_jl);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::Fill(
    std::vector<uint8_t> serialized_init_value,
    jl_value_t **values_ptr,
    size_t num_values) {
  jl_value_t *init_value_jl = nullptr;
  jl_value_t *num_values_jl = nullptr;
  jl_value_t *serialized_value_array = nullptr,
              *serialized_value_buff = nullptr,
        *serialized_value_array_type = nullptr;

  JL_GC_PUSH5(&init_value_jl,
              &num_values_jl,
              &serialized_value_array,
              &serialized_value_buff,
              &serialized_value_array_type);
  num_values_jl = jl_box_uint64(num_values);
  LOG(INFO) << __func__ << " serialized_init_value size = "
            << serialized_init_value.size();
  serialized_value_array_type = jl_apply_array_type(jl_uint8_type, 1);
  serialized_value_array = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(serialized_value_array_type,
                         serialized_init_value.data(),
                         serialized_init_value.size(), 0));
  jl_function_t *io_buffer_func
      = GetFunction(jl_base_module, "IOBuffer");
  CHECK(io_buffer_func != nullptr);
  serialized_value_buff = jl_call1(io_buffer_func,
                                   reinterpret_cast<jl_value_t*>(serialized_value_array));
  jl_function_t *deserialize_func
      = GetFunction(jl_base_module, "deserialize");
  CHECK(deserialize_func != nullptr);
  init_value_jl = jl_call1(deserialize_func, serialized_value_buff);

  jl_function_t *fill_func = GetFunction(jl_base_module, "fill");
  *values_ptr = jl_call2(fill_func, init_value_jl,
                         num_values_jl);

  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::RunMapGeneric(
    DistArrayMapType map_type,
    const std::vector<int64_t> &parent_dims,
    const std::vector<int64_t> &child_dims,
    size_t num_keys,
    int64_t *keys,
    jl_value_t* input_values,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name,
    std::vector<int64_t>* output_keys,
    jl_value_t *output_value_type,
    jl_value_t **output_values_ptr) {

  switch (map_type) {
    case DistArrayMapType::kNoMap:
      break;
    case DistArrayMapType::kMap:
      {
        RunMap(parent_dims,
               child_dims,
               num_keys,
               keys,
               input_values,
               mapper_func_module,
               mapper_func_name,
               output_keys,
               output_value_type,
               output_values_ptr);

        break;
      }
    case DistArrayMapType::kMapFixedKeys:
      {
        RunMapFixedKeys(parent_dims,
                        num_keys,
                        keys,
                        input_values,
                        mapper_func_module,
                        mapper_func_name,
                        output_value_type,
                        output_values_ptr);
        output_keys->resize(num_keys);
        memcpy(output_keys->data(), keys, sizeof(int64_t) * num_keys);
        break;
      }
    case DistArrayMapType::kMapValues:
      {
        RunMapValues(child_dims,
                     num_keys,
                     input_values,
                     mapper_func_module,
                     mapper_func_name,
                     output_value_type,
                     output_values_ptr);
        output_keys->resize(num_keys);
        memcpy(output_keys->data(), keys, sizeof(int64_t) * num_keys);
        break;
      }
    case DistArrayMapType::kMapValuesNewKeys:
      {
        RunMapValuesNewKeys(child_dims,
                            num_keys,
                            input_values,
                            mapper_func_module,
                            mapper_func_name,
                            output_keys,
                            output_value_type,
                            output_values_ptr);

        break;
      }
    default:
      LOG(FATAL) << "unrecognized map type = " << static_cast<int>(map_type);
  }
}

void
JuliaEvaluator::RunMap(
    const std::vector<int64_t> &parent_dims,
    const std::vector<int64_t> &child_dims,
    size_t num_keys,
    int64_t *keys,
    jl_value_t *input_values,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name,
    std::vector<int64_t>* output_keys,
    jl_value_t *output_value_type,
    jl_value_t **output_values_ptr) {

  jl_value_t *key_array_type = nullptr,
               *key_array_jl = nullptr,
         *parent_dim_array_jl = nullptr,
         *child_dim_array_jl = nullptr,
            *output_tuple_jl = nullptr;

  JL_GC_PUSH5(&key_array_type,
              &key_array_jl,
              &parent_dim_array_jl,
              &child_dim_array_jl,
              &output_tuple_jl);

  auto temp_parent_dims = parent_dims;
  auto temp_child_dims = child_dims;

  key_array_type = jl_apply_array_type(jl_int64_type, 1);
  parent_dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, temp_parent_dims.data(), temp_parent_dims.size(), 0));
  child_dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, temp_child_dims.data(), temp_child_dims.size(), 0));
  key_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, keys, num_keys, 0));
  jl_function_t *mapper_func = GetFunction(
      GetJlModule(mapper_func_module), mapper_func_name.c_str());
  {
    jl_value_t *args[5];
    args[0] = parent_dim_array_jl;
    args[1] = child_dim_array_jl;
    args[2] = key_array_jl;
    args[3] = input_values;
    args[4] = output_value_type;
    output_tuple_jl = jl_call(mapper_func, args, 5);
  }

  jl_value_t *output_key_array_jl = jl_get_nth_field(output_tuple_jl, 0);
  *output_values_ptr = jl_get_nth_field(output_tuple_jl, 1);

  size_t num_output_keys = jl_array_len(output_key_array_jl);
  output_keys->resize(num_output_keys);
  uint8_t *output_key_array = reinterpret_cast<uint8_t*>(jl_array_data(output_key_array_jl));
  memcpy(output_keys->data(), output_key_array, num_output_keys * sizeof(int64_t));

  size_t num_output_values = jl_array_len(*output_values_ptr);
  CHECK_EQ(num_output_values, num_output_keys);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::RunMapFixedKeys(
    const std::vector<int64_t> &parent_dims,
    size_t num_keys,
    int64_t *keys,
    jl_value_t *input_values,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name,
    jl_value_t *output_value_type,
    jl_value_t **output_values_ptr) {
  jl_value_t *key_array_type = nullptr,
               *key_array_jl = nullptr,
               *dim_array_jl = nullptr,
            *output_tuple_jl = nullptr;

  JL_GC_PUSH4(&key_array_type,
              &key_array_jl,
              &dim_array_jl,
              &output_tuple_jl);
  auto temp_parent_dims = parent_dims;
  key_array_type = jl_apply_array_type(jl_int64_type, 1);
  dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, temp_parent_dims.data(), temp_parent_dims.size(), 0));
  key_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, keys, num_keys, 0));
  jl_function_t *mapper_func = GetFunction(
      GetJlModule(mapper_func_module), mapper_func_name.c_str());

  {
    jl_value_t *args[4];
    args[0] = dim_array_jl;
    args[1] = key_array_jl;
    args[2] = input_values;
    args[3] = output_value_type;
    output_tuple_jl = jl_call(mapper_func, args, 4);
  }
  *output_values_ptr = jl_get_nth_field(output_tuple_jl, 1);

  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::RunMapValues(
    const std::vector<int64_t> &child_dims,
    size_t num_values,
    jl_value_t *input_values,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name,
    jl_value_t *output_value_type,
    jl_value_t **output_values_ptr) {
  jl_value_t *key_array_type = nullptr,
            *output_tuple_jl = nullptr,
                *dim_array_jl = nullptr;
  JL_GC_PUSH3(&key_array_type,
              &output_tuple_jl,
              &dim_array_jl);
  auto temp_child_dims = child_dims;
  key_array_type = jl_apply_array_type(jl_int64_type, 1);
  dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, temp_child_dims.data(), temp_child_dims.size(), 0));
  jl_function_t *mapper_func = GetFunction(
      GetJlModule(mapper_func_module), mapper_func_name.c_str());
  output_tuple_jl = jl_call3(mapper_func, dim_array_jl, input_values,
                             reinterpret_cast<jl_value_t*>(output_value_type));
  CHECK(jl_is_tuple(output_tuple_jl));

  *output_values_ptr = jl_get_nth_field(output_tuple_jl, 1);
  CHECK(jl_is_array(*output_values_ptr));
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::RunMapValuesNewKeys(
    const std::vector<int64_t> &child_dims,
    size_t num_values,
    jl_value_t *input_values,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name,
    std::vector<int64_t>* output_keys,
    jl_value_t *output_value_type,
    jl_value_t **output_values_ptr) {

  jl_value_t *key_array_type = nullptr,
               *dim_array_jl = nullptr,
          *output_key_array_jl = nullptr,
            *output_tuple_jl = nullptr;

  JL_GC_PUSH4(&key_array_type,
              &dim_array_jl,
              &output_key_array_jl,
              &output_tuple_jl);
  auto temp_child_dims = child_dims;
  key_array_type = jl_apply_array_type(jl_int64_type, 1);

  dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, temp_child_dims.data(), temp_child_dims.size(), 0));

  jl_function_t *mapper_func = GetFunction(
      GetJlModule(mapper_func_module), mapper_func_name.c_str());

  output_tuple_jl = jl_call3(mapper_func, dim_array_jl, input_values,
                             output_value_type);
  output_key_array_jl = jl_get_nth_field(output_tuple_jl, 0);
  *output_values_ptr = jl_get_nth_field(output_tuple_jl, 1);

  size_t num_output_keys = jl_array_len(output_key_array_jl);
  output_keys->resize(num_output_keys);
  uint8_t *output_key_array = reinterpret_cast<uint8_t*>(jl_array_data(output_key_array_jl));
  memcpy(output_keys->data(), output_key_array, num_output_keys * sizeof(int64_t));
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::GetVarValue(
    const std::string &symbol,
    Blob *result_buff) {
  jl_sym_t *symbol_jl = nullptr;
  jl_value_t *value_jl = nullptr,
                 *buff = nullptr,
*serialized_result_array = nullptr;

  JL_GC_PUSH4(reinterpret_cast<jl_value_t**>(&symbol_jl),
              &value_jl,
              &buff,
              &serialized_result_array);
  symbol_jl = jl_symbol(symbol.c_str());
  value_jl = jl_get_global(jl_main_module, symbol_jl);

  jl_function_t *io_buffer_func
      = GetFunction(jl_base_module, "IOBuffer");
  CHECK(io_buffer_func != nullptr);
  buff = jl_call0(io_buffer_func);

  jl_function_t *serialize_func
      = GetFunction(jl_base_module, "serialize");
  CHECK(serialize_func != nullptr);
  jl_call2(serialize_func, buff, value_jl);

  jl_function_t *takebuff_array_func
      = GetFunction(jl_base_module, "takebuf_array");
  serialized_result_array = jl_call1(takebuff_array_func, buff);
  size_t result_array_length = jl_array_len(serialized_result_array);
  uint8_t* array_bytes = reinterpret_cast<uint8_t*>(jl_array_data(serialized_result_array));
  result_buff->resize(result_array_length);
  memcpy(result_buff->data(), array_bytes, result_array_length);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::SetVarValue(
    const std::string &symbol,
    uint8_t *serialized_value,
    size_t value_size) {

  jl_sym_t *symbol_jl = nullptr;
  jl_value_t *serialized_value_array = nullptr,
        *serialized_value_array_type = nullptr,
                           *value_jl = nullptr,
              *serialized_value_buff = nullptr;
  JL_GC_PUSH5(reinterpret_cast<jl_value_t**>(&symbol_jl),
              &value_jl,
              &serialized_value_buff,
              &serialized_value_array,
              &serialized_value_array_type);
  serialized_value_array_type = jl_apply_array_type(jl_uint8_type, 1);

  serialized_value_array = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(serialized_value_array_type,
                         serialized_value,
                         value_size, 0));
  jl_function_t *io_buffer_func
      = GetFunction(jl_base_module, "IOBuffer");
  CHECK(io_buffer_func != nullptr);
  serialized_value_buff = jl_call1(io_buffer_func,
                                   reinterpret_cast<jl_value_t*>(serialized_value_array));

  jl_function_t *deserialize_func
      = GetFunction(jl_base_module, "deserialize");
  CHECK(deserialize_func != nullptr);
  value_jl = jl_call1(deserialize_func, serialized_value_buff);
  symbol_jl = jl_symbol(symbol.c_str());
  jl_set_global(jl_main_module, symbol_jl, value_jl);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::CombineVarValue(
    const std::string &symbol,
    uint8_t *serialized_value_to_combine,
    size_t value_size,
    const std::string &combiner) {
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 7);
  jl_sym_t *&symbol_jl = reinterpret_cast<jl_sym_t*&>(jl_values[0]);
  jl_value_t *&serialized_value_array = jl_values[1],
        *&serialized_value_array_type = jl_values[2],
                           *&value_to_combine_jl = jl_values[3],
              *&serialized_value_buff = jl_values[4],
                  *&original_value_jl = jl_values[5],
                       *&new_value_jl = jl_values[6];
  serialized_value_array_type = jl_apply_array_type(jl_uint8_type, 1);
  serialized_value_array = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(serialized_value_array_type,
                         serialized_value_to_combine,
                         value_size, 0));
  jl_function_t *io_buffer_func
      = GetFunction(jl_base_module, "IOBuffer");
  CHECK(io_buffer_func != nullptr);
  serialized_value_buff = jl_call1(io_buffer_func,
                                   reinterpret_cast<jl_value_t*>(serialized_value_array));

  jl_function_t *deserialize_func
      = GetFunction(jl_base_module, "deserialize");
  CHECK(deserialize_func != nullptr);
  value_to_combine_jl = jl_call1(deserialize_func, serialized_value_buff);
  symbol_jl = jl_symbol(symbol.c_str());
  original_value_jl = jl_get_global(jl_main_module, symbol_jl);
  jl_function_t *combiner_func = GetFunction(jl_base_module, combiner.c_str());
  new_value_jl = jl_call2(combiner_func, original_value_jl, value_to_combine_jl);
  jl_set_global(jl_main_module, symbol_jl, new_value_jl);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::SetDistArrayDims(
    const std::string &dist_array_sym,
    const std::vector<int64_t> &dims) {
  jl_sym_t *dist_array_sym_jl = nullptr;
  jl_value_t *dist_array_jl = nullptr;
  jl_value_t* dims_vec_jl = nullptr;
  jl_value_t* dims_vec_array_type_jl = nullptr;
  auto temp_dims = dims;
  JL_GC_PUSH4(reinterpret_cast<jl_value_t**>(&dist_array_sym_jl),
              &dist_array_jl,
              &dims_vec_jl,
              &dims_vec_array_type_jl);
  dist_array_sym_jl = jl_symbol(dist_array_sym.c_str());
  dist_array_jl = jl_get_global(jl_main_module, dist_array_sym_jl);

  dims_vec_array_type_jl = jl_apply_array_type(jl_int64_type, 1);
  dims_vec_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(dims_vec_array_type_jl,
                         temp_dims.data(), temp_dims.size(), 0));

  jl_function_t *set_dist_array_dims_func = GetFunction(
      jl_main_module, "orionres_set_dist_array_dims");
  jl_call2(set_dist_array_dims_func, dist_array_jl, dims_vec_jl);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::GetDistArray(
    const std::string &dist_array_sym,
    jl_value_t** dist_array_ptr) {
  jl_sym_t *dist_array_sym_jl = nullptr;
  JL_GC_PUSH1(&dist_array_sym_jl);

  dist_array_sym_jl = jl_symbol(dist_array_sym.c_str());
  *dist_array_ptr = jl_get_global(jl_main_module, dist_array_sym_jl);
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::GetDistArrayValueType(
    const std::string &dist_array_sym,
    jl_datatype_t **value_type_ptr) {
  jl_sym_t *dist_array_sym_jl = nullptr;
  jl_value_t *dist_array_jl = nullptr;
  JL_GC_PUSH2(reinterpret_cast<jl_value_t*>(&dist_array_sym_jl),
              &dist_array_jl);

  dist_array_sym_jl = jl_symbol(dist_array_sym.c_str());
  dist_array_jl = jl_get_global(jl_main_module, dist_array_sym_jl);

  jl_function_t *get_dist_array_value_type_func = GetFunction(
      jl_main_module, "orionres_get_dist_array_value_type");
  *value_type_ptr = reinterpret_cast<jl_datatype_t*>(jl_call1(get_dist_array_value_type_func,
                                                              dist_array_jl));
  JL_GC_POP();
  AbortIfException();
}

void
JuliaEvaluator::GetDistArrayValueType(
    jl_value_t *dist_array_jl,
    jl_datatype_t **value_type_ptr) {

  jl_function_t *get_dist_array_value_type_func = GetFunction(
      jl_main_module, "orionres_get_dist_array_value_type");
  *value_type_ptr = reinterpret_cast<jl_datatype_t*>(
      jl_call1(get_dist_array_value_type_func,
               dist_array_jl));
  AbortIfException();
}

void
JuliaEvaluator::GetAndSerializeValue(DistArray *dist_array,
                                     int64_t key,
                                     Blob *bytes_buff) {
  return dist_array->GetAndSerializeValue(key, bytes_buff);
}

void
JuliaEvaluator::GetAndSerializeValues(std::unordered_map<int32_t, DistArray> *dist_arrays,
                                      const uint8_t *request,
                                      Blob *bytes_buff) {
  const auto *cursor = request;
  size_t num_dist_arrays = *(reinterpret_cast<const size_t*>(cursor));
  cursor += sizeof(size_t);
  std::vector<Blob> buff_vec(num_dist_arrays);
  std::vector<int32_t> dist_array_ids(num_dist_arrays);
  size_t accum_size = 0;
  for (size_t i = 0; i < num_dist_arrays; i++) {
    int32_t dist_array_id = *reinterpret_cast<const int32_t*>(cursor);
    cursor += sizeof(int32_t);
    size_t num_keys = *reinterpret_cast<const size_t*>(cursor);
    cursor += sizeof(size_t);
    const int64_t *keys = reinterpret_cast<const int64_t*>(cursor);
    cursor += sizeof(int64_t) * num_keys;
    auto dist_array_iter = dist_arrays->find(dist_array_id);
    CHECK(dist_array_iter != dist_arrays->end());
    auto *dist_array_ptr = &dist_array_iter->second;
    dist_array_ptr->GetAndSerializeValues(keys, num_keys, &buff_vec[i]);
    accum_size += buff_vec[i].size() + sizeof(int32_t);
    dist_array_ids[i] = dist_array_id;
  }

  bytes_buff->resize(accum_size + sizeof(size_t));
  auto *write_cursor = bytes_buff->data();
  *reinterpret_cast<size_t*>(write_cursor) = num_dist_arrays;
  write_cursor += sizeof(size_t);
  for (size_t i = 0; i < num_dist_arrays; i++) {
    auto &buff = buff_vec[i];
    int32_t dist_array_id = dist_array_ids[i];
    *reinterpret_cast<int32_t*>(write_cursor) = dist_array_id;
    write_cursor += sizeof(int32_t);
    memcpy(write_cursor, buff.data(), buff.size());
    write_cursor += buff.size();
  }
}

}
}
