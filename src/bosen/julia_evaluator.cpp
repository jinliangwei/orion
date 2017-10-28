#include <orion/bosen/julia_evaluator.hpp>
#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/julia_module.hpp>
namespace orion {
namespace bosen {

void
JuliaEvaluator::Init(const std::string &orion_home) {
  jl_init(NULL);
  orion_home_ = orion_home;
  lib_path_ = orion_home + "/lib/liborion.so";
  jl_load((orion_home + "/src/julia/orion_worker.jl").c_str());
  orion_worker_module_ = reinterpret_cast<jl_module_t*>(
      jl_eval_string("OrionWorker"));
  CHECK(orion_worker_module_ != nullptr);
  SetOrionWorkerModule(orion_worker_module_);

  jl_value_t *lib_path_str = nullptr;
  JL_GC_PUSH1(&lib_path_str);
  lib_path_str = jl_cstr_to_string(lib_path_.c_str());
  jl_function_t *set_lib_path_func
      = GetFunction(orion_worker_module_, "set_lib_path");
  jl_call1(set_lib_path_func, lib_path_str);
  jl_function_t *helloworld_func
      = GetFunction(orion_worker_module_, "helloworld");
  jl_call0(helloworld_func);
  JL_GC_POP();
}


jl_value_t*
JuliaEvaluator::BoxValue(
    type::PrimitiveType result_type,
    const uint8_t* value) {
  jl_value_t* ret = nullptr;
  switch (result_type) {
    case type::PrimitiveType::kVoid:
      break;
    case type::PrimitiveType::kInt8:
      {
        ret = jl_box_int8(*reinterpret_cast<const int8_t*>(value));
      }
      break;
    case type::PrimitiveType::kUInt8:
      {
        ret = jl_box_uint8(*reinterpret_cast<const uint8_t*>(value));
      }
      break;
    case type::PrimitiveType::kInt16:
      {
        ret = jl_box_int16(*reinterpret_cast<const int16_t*>(value));
      }
      break;
    case type::PrimitiveType::kUInt16:
      {
        ret = jl_box_uint16(*reinterpret_cast<const uint16_t*>(value));
      }
      break;
    case type::PrimitiveType::kInt32:
      {
        ret = jl_box_int32(*reinterpret_cast<const int32_t*>(value));
      }
      break;
    case type::PrimitiveType::kUInt32:
      {
        ret = jl_box_uint32(*reinterpret_cast<const uint32_t*>(value));
      }
      break;
    case type::PrimitiveType::kInt64:
      {
        ret = jl_box_int64(*reinterpret_cast<const int64_t*>(value));
      }
      break;
    case type::PrimitiveType::kUInt64:
      {
        ret = jl_box_uint64(*reinterpret_cast<const uint64_t*>(value));
      }
      break;
    case type::PrimitiveType::kFloat32:
      {
        ret = jl_box_float32(*reinterpret_cast<const float*>(value));
      }
      break;
    case type::PrimitiveType::kFloat64:
      {
        ret = jl_box_float32(*reinterpret_cast<const double*>(value));
      }
      break;
    case type::PrimitiveType::kString:
      {
        ret = jl_cstr_to_string(reinterpret_cast<const char*>(value));
      }
      break;
    default:
      LOG(FATAL) << "Unknown primitive type";
  }
  return ret;
}

void
JuliaEvaluator::UnboxValue(jl_value_t* value,
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

void
JuliaEvaluator::EvalExpr(const std::string &serialized_expr,
                         JuliaModule module,
                         Blob *result_buff) {
  LOG(INFO) << __func__ << " started!";
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
  if (jl_exception_occurred())
        LOG(INFO) << "julia exception occurs: " << jl_typeof_str(jl_exception_occurred());
  jl_function_t *eval_func
      = GetFunction(jl_core_module, "eval");
  CHECK(eval_func != nullptr);

  jl_module_t *jl_module = GetJlModule(module);
  CHECK(jl_module != nullptr);
  ret = jl_call2(eval_func, reinterpret_cast<jl_value_t*>(jl_module), expr);
  if (jl_exception_occurred())
    LOG(INFO) << "julia exception occurs: " << jl_typeof_str(jl_exception_occurred());
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
  LOG(INFO) << __func__ << " result_array_length = " << result_array_length;
  memcpy(result_buff->data(), array_bytes, result_array_length);
  JL_GC_POP();
  CHECK(!jl_exception_occurred()) << "julia exception occurs: "
                                  << jl_typeof_str(jl_exception_occurred());
}

jl_function_t*
JuliaEvaluator::GetFunction(jl_module_t* module,
                            const char* func_name) {
  auto* func = jl_get_function(module, func_name);
  CHECK(func != nullptr) << "func_name = " << func_name;
  return func;
}

void
JuliaEvaluator::ExecuteTask(JuliaTask* task) {
  if (auto exec_cpp_func_task = dynamic_cast<ExecCppFuncTask*>(task)) {
    exec_cpp_func_task->func(this);
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
  UnboxValue(value, result_type, value_buff);
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
  UnboxValue(value, result_type, value_buff);
  JL_GC_POP();
}

void
JuliaEvaluator::DefineVar(std::string var_name,
                          std::string var_value) {
  LOG(INFO) << __func__ << " var_name = " << var_name
            << " var_value.size = " << var_value.size();

  jl_value_t *array_type = nullptr,
  *serialized_value_buff = nullptr,
                           *var_value_jl = nullptr,
                           *var_name_jl = nullptr;
  jl_array_t *serialized_value_array = nullptr;
  JL_GC_PUSH5(&array_type, &serialized_value_buff, &var_name_jl,
              &var_value_jl, &serialized_value_array);

  jl_function_t *define_setter_func
      = GetFunction(jl_main_module, "orion_define_setter");
  var_name_jl = jl_cstr_to_string(var_name.c_str());
  jl_call1(define_setter_func, var_name_jl);

  array_type = jl_apply_array_type(jl_uint8_type, 1);
  std::vector<uint8_t> temp_serialized_value(var_value.size());
  memcpy(temp_serialized_value.data(), var_value.data(),
         var_value.size());
  serialized_value_array = jl_ptr_to_array_1d(array_type,
                                              temp_serialized_value.data(),
                                              temp_serialized_value.size(), 0);
  jl_function_t *io_buffer_func
      = GetFunction(jl_base_module, "IOBuffer");
  serialized_value_buff = jl_call1(io_buffer_func,
                                   reinterpret_cast<jl_value_t*>(serialized_value_array));

  jl_function_t *deserialize_func
      = GetFunction(jl_base_module, "deserialize");
  var_value_jl = jl_call1(deserialize_func, serialized_value_buff);

  jl_function_t *setter_func
      = GetFunction(jl_main_module, (std::string("orion_set_") + var_name).c_str());
  jl_call1(setter_func, var_value_jl);
  JL_GC_POP();
}

void
JuliaEvaluator::StaticDefineVar(
    JuliaEvaluator *julia_eval,
    std::string var_name,
    std::string var_value) {
  julia_eval->DefineVar(var_name, var_value);
}

void
JuliaEvaluator::DefineDistArray(
    std::string *symbol,
    type::PrimitiveType value_type,
    std::vector<int64_t> *dims,
    bool is_dense,
    void* access_ptr) {
  LOG(INFO) << __func__ << " dist_array: " << *symbol;
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 6);
  jl_value_t* &symbol_jl = jl_values[0];
  jl_value_t* &value_type_jl = jl_values[1];
  jl_value_t* &dims_vec_jl = jl_values[2];
  jl_value_t* &is_dense_jl = jl_values[3];
  jl_value_t* &access_ptr_jl = jl_values[4];
  jl_value_t* &dims_vec_array_type_jl = jl_values[5];

  symbol_jl = jl_cstr_to_string(symbol->c_str());
  value_type_jl = reinterpret_cast<jl_value_t*>(type::GetJlDataType(value_type));
  dims_vec_array_type_jl = jl_apply_array_type(jl_int64_type, 1);
  dims_vec_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(dims_vec_array_type_jl,
                         dims->data(), dims->size(), 0));
  is_dense_jl = jl_box_bool(is_dense);
  access_ptr_jl = jl_box_voidpointer(access_ptr);

  jl_function_t *create_dist_array_func = GetFunction(
      jl_main_module, "orionres_define_dist_array");
  jl_value_t* args[5];
  args[0] = value_type_jl;
  args[1] = symbol_jl;
  args[2] = dims_vec_jl;
  args[3] = is_dense_jl;
  args[4] = access_ptr_jl;
  jl_call(create_dist_array_func, args, 5);
  if (jl_exception_occurred())
      LOG(FATAL) << "julia exception occurs: " << jl_typeof_str(jl_exception_occurred());
  JL_GC_POP();
}

void
JuliaEvaluator::StaticDefineDistArray(
    JuliaEvaluator *julia_eval,
    std::string *symbol,
    type::PrimitiveType value_type,
    std::vector<int64_t> *dims,
    bool is_dense,
    void* access_ptr) {
  julia_eval->DefineDistArray(symbol,
                              value_type,
                              dims,
                              is_dense,
                              access_ptr);
}

void
JuliaEvaluator::ComputeRepartition(
    std::string repartition_func_name,
    DistArray *dist_array) {
  LOG(INFO) << __func__;
  jl_value_t *array_type = nullptr,
            *keys_vec_jl = nullptr,
            *dims_vec_jl = nullptr,
 *repartition_ids_vec_jl = nullptr;
  JL_GC_PUSH4(&array_type,
              &keys_vec_jl, &dims_vec_jl, &repartition_ids_vec_jl);
  auto &dims = dist_array->GetDims();
  std::vector<AbstractDistArrayPartition*> partition_buff;
  dist_array->GetAndClearLocalPartitions(&partition_buff);
  for (auto dist_array_partition : partition_buff) {
    auto &keys = dist_array_partition->GetKeys();
    array_type = jl_apply_array_type(jl_int64_type, 1);
    keys_vec_jl = reinterpret_cast<jl_value_t*>(
        jl_ptr_to_array_1d(array_type, keys.data(), keys.size(), 0));
    dims_vec_jl = reinterpret_cast<jl_value_t*>(
        jl_ptr_to_array_1d(array_type, dims.data(), dims.size(), 0));
    jl_function_t *repartition_func = GetFunction(jl_main_module, repartition_func_name.c_str());
    repartition_ids_vec_jl = jl_call2(repartition_func, keys_vec_jl, dims_vec_jl);
    CHECK(!jl_exception_occurred()) << jl_typeof_str(jl_exception_occurred());
    int32_t *repartition_ids = reinterpret_cast<int32_t*>(jl_array_data(repartition_ids_vec_jl));
    dist_array_partition->Repartition(repartition_ids);
    delete dist_array_partition;
  }
  JL_GC_POP();
}

void
JuliaEvaluator::StaticComputeRepartition(
    JuliaEvaluator *julia_eval,
    std::string repartition_func_name,
    DistArray *dist_array) {
  julia_eval->ComputeRepartition(
      repartition_func_name,
      dist_array);
}

void
JuliaEvaluator::RandNormal(
    type::PrimitiveType value_type,
    uint8_t *buff,
    size_t array_len) {
  jl_value_t *array_type = nullptr,
               *array_jl = nullptr,
           *rand_seed_jl = nullptr;
  JL_GC_PUSH3(&array_type, &array_jl, &rand_seed_jl);
  rand_seed_jl = jl_box_int32(1);
  jl_function_t *srand_func = GetFunction(jl_base_module, "srand");
  jl_call1(srand_func, rand_seed_jl);

  array_type = jl_apply_array_type(
      type::GetJlDataType(value_type), 1);

  array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      array_type, buff, array_len, 0));

  jl_function_t *rand_func = GetFunction(jl_base_module, "randn!");
  jl_call1(rand_func, array_jl);

  CHECK(!jl_exception_occurred());
  JL_GC_POP();
}

void
JuliaEvaluator::RunMapGeneric(
    task::DistArrayMapType map_type,
    std::vector<int64_t> dims,
    size_t num_keys,
    int64_t *keys,
    type::PrimitiveType input_value_type,
    uint8_t *input_values,
    std::vector<int64_t>* output_keys,
    type::PrimitiveType output_value_type,
    Blob *output_values,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name) {
  switch (map_type) {
    case task::NO_MAP:
      break;
    case task::MAP:
      {
        RunMap(dims, num_keys, keys,
                  input_value_type,
                  input_values,
                  output_keys,
                  output_value_type,
                  output_values,
                  mapper_func_module,
                  mapper_func_name);
        break;
      }
    case task::MAP_FIXED_KEYS:
      {
        RunMapFixedKeys(dims, num_keys, keys,
                  input_value_type,
                  input_values,
                  output_value_type,
                  output_values,
                  mapper_func_module,
                  mapper_func_name);
        break;
      }
    case task::MAP_VALUES:
      {
        RunMapValues(dims, num_keys,
                  input_value_type,
                  input_values,
                  output_value_type,
                  output_values,
                  mapper_func_module,
                  mapper_func_name);
        break;
      }
    case task::MAP_VALUES_NEW_KEYS:
      {
        RunMapValuesNewKeys(dims, num_keys,
                  input_value_type,
                  input_values,
                  output_keys,
                  output_value_type,
                  output_values,
                  mapper_func_module,
                  mapper_func_name);
        break;
      }
    default:
      LOG(FATAL) << "unrecognized map type = " << static_cast<int>(map_type);
  }
}

void
JuliaEvaluator::RunMap(
    std::vector<int64_t> dims,
    size_t num_keys,
    int64_t *keys,
    type::PrimitiveType input_value_type,
    uint8_t *input_values,
    std::vector<int64_t>* output_keys,
    type::PrimitiveType output_value_type,
    Blob *output_values,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name) {
  jl_value_t *input_array_type = nullptr,
               *input_array_jl = nullptr,
               *key_array_type = nullptr,
                 *key_array_jl = nullptr,
              *output_tuple_jl = nullptr,
                 *dim_array_jl = nullptr;

  JL_GC_PUSH4(&input_array_jl,
              &dim_array_jl,
              &key_array_jl,
              &output_tuple_jl);

  input_array_type = jl_apply_array_type(
      type::GetJlDataType(input_value_type), 1);
  key_array_type = jl_apply_array_type(jl_int64_type, 1);
  input_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(input_array_type, input_values,
                         num_keys, 0));
  dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, dims.data(),
                         num_keys, 0));
  key_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, keys,
                         num_keys, 0));
  jl_function_t *mapper_func = GetFunction(
      GetJlModule(mapper_func_module), mapper_func_name.c_str());
  jl_datatype_t *output_value_type_jl = type::GetJlDataType(output_value_type);
  {
    jl_value_t *args[4];
    args[0] = dim_array_jl;
    args[1] = key_array_jl;
    args[2] = input_array_jl;
    args[3] = reinterpret_cast<jl_value_t*>(output_value_type_jl);
    output_tuple_jl = jl_call(mapper_func, args, 4);
  }

  jl_value_t *output_key_array_jl = jl_get_nth_field(output_tuple_jl, 0);
  jl_value_t *output_array_jl = jl_get_nth_field(output_tuple_jl, 1);

  size_t num_output_keys = jl_array_len(output_key_array_jl);
  output_keys->resize(num_output_keys);
  uint8_t *output_key_array = reinterpret_cast<uint8_t*>(jl_array_data(output_key_array_jl));
  memcpy(output_keys->data(), output_key_array, num_output_keys * sizeof(int64_t));

  size_t num_output_values = jl_array_len(output_array_jl);
  CHECK_EQ(num_output_values, num_output_keys);
  uint8_t *output_array = reinterpret_cast<uint8_t*>(jl_array_data(output_array_jl));
  output_values->resize(num_output_values * type::SizeOf(output_value_type));
  memcpy(output_values->data(), output_array, num_output_values * type::SizeOf(output_value_type));
  JL_GC_POP();
  CHECK(!jl_exception_occurred());
}

void
JuliaEvaluator::RunMapFixedKeys(
    std::vector<int64_t> dims,
    size_t num_keys,
    int64_t *keys,
    type::PrimitiveType input_value_type,
    uint8_t *input_values,
    type::PrimitiveType output_value_type,
    Blob *output_values,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name) {
  jl_value_t *input_array_type = nullptr,
               *input_array_jl = nullptr,
               *key_array_type = nullptr,
                 *key_array_jl = nullptr,
              *output_tuple_jl = nullptr,
                 *dim_array_jl = nullptr;

  JL_GC_PUSH4(&input_array_jl,
              &dim_array_jl,
              &key_array_jl,
              &output_tuple_jl);

  input_array_type = jl_apply_array_type(
      type::GetJlDataType(input_value_type), 1);
  key_array_type = jl_apply_array_type(jl_int64_type, 1);
  input_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(input_array_type, input_values,
                         num_keys, 0));
  dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, dims.data(),
                         num_keys, 0));
  key_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, keys,
                         num_keys, 0));
  jl_function_t *mapper_func = GetFunction(
      GetJlModule(mapper_func_module), mapper_func_name.c_str());

  jl_datatype_t *output_value_type_jl = type::GetJlDataType(output_value_type);
  {
    jl_value_t *args[4];
    args[0] = dim_array_jl;
    args[1] = key_array_jl;
    args[2] = input_array_jl;
    args[3] = reinterpret_cast<jl_value_t*>(output_value_type_jl);
    output_tuple_jl = jl_call(mapper_func, args, 4);
  }
  jl_value_t *output_array_jl = jl_get_nth_field(output_tuple_jl, 1);

  size_t num_output_values = jl_array_len(output_array_jl);
  uint8_t *output_array = reinterpret_cast<uint8_t*>(jl_array_data(output_array_jl));
  output_values->resize(num_output_values * type::SizeOf(output_value_type));
  memcpy(output_values->data(), output_array, num_output_values * type::SizeOf(output_value_type));
  JL_GC_POP();
  CHECK(!jl_exception_occurred());
}

void
JuliaEvaluator::RunMapValues(
    std::vector<int64_t> dims,
    size_t num_keys,
    type::PrimitiveType input_value_type,
    uint8_t *input_values,
    type::PrimitiveType output_value_type,
    Blob *output_values,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name) {
  LOG(INFO) << __func__;

  jl_value_t *input_array_jl = nullptr,
      *output_tuple_jl = nullptr,
      *output_array_jl = nullptr,
      *dim_array_jl = nullptr;

  JL_GC_PUSH4(&input_array_jl,
              &dim_array_jl,
              &output_tuple_jl,
              &output_array_jl);

  LOG(INFO) << "input_value_type = " << static_cast<int>(input_value_type);

  jl_value_t* input_array_type = jl_apply_array_type(
      type::GetJlDataType(input_value_type), 1);
  jl_value_t* key_array_type = jl_apply_array_type(jl_int64_type, 1);

  input_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(input_array_type, input_values, num_keys, 0));

  dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, dims.data(), dims.size(), 0));

  jl_function_t *mapper_func = GetFunction(
      GetJlModule(mapper_func_module), mapper_func_name.c_str());
  jl_datatype_t *output_value_type_jl = type::GetJlDataType(output_value_type);
  LOG(INFO) << "before call! "
            << " num_keys = " << num_keys << " "
            << (void*) mapper_func << " "
            << (void*) dim_array_jl << " "
            << (void*) input_array_jl << " "
            << (void*) output_value_type_jl;

  output_tuple_jl = jl_call3(mapper_func, dim_array_jl, input_array_jl,
                             reinterpret_cast<jl_value_t*>(output_value_type_jl));
  CHECK(jl_is_tuple(output_tuple_jl));
  output_array_jl = jl_get_nth_field(output_tuple_jl, 1);
  CHECK(jl_is_array(output_array_jl));
  size_t num_output_values = jl_array_len(output_array_jl);
  uint8_t *output_array = reinterpret_cast<uint8_t*>(jl_array_data(output_array_jl));
  output_values->resize(num_output_values * type::SizeOf(output_value_type));
  memcpy(output_values->data(), output_array, num_output_values * type::SizeOf(output_value_type));
  JL_GC_POP();
  CHECK(!jl_exception_occurred());
}

void
JuliaEvaluator::RunMapValuesNewKeys(
    std::vector<int64_t> dims,
    size_t num_keys,
    type::PrimitiveType input_value_type,
    uint8_t *input_values,
    std::vector<int64_t> *output_keys,
    type::PrimitiveType output_value_type,
    Blob *output_values,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name) {
  jl_value_t *input_array_type = nullptr,
      *key_array_type = nullptr,
      *input_array_jl = nullptr,
      *output_tuple_jl = nullptr,
      *dim_array_jl = nullptr;

  JL_GC_PUSH3(&input_array_jl,
              &dim_array_jl,
              &output_tuple_jl);

  input_array_type = jl_apply_array_type(
      type::GetJlDataType(input_value_type), 1);
  key_array_type = jl_apply_array_type(jl_int64_type, 1);
  input_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(input_array_type, input_values,
                         num_keys, 0));
  dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, dims.data(),
                         num_keys, 0));
  jl_function_t *mapper_func = GetFunction(
      GetJlModule(mapper_func_module), mapper_func_name.c_str());
  jl_datatype_t *output_value_type_jl = type::GetJlDataType(output_value_type);
  output_tuple_jl = jl_call3(mapper_func, dim_array_jl, input_array_jl,
                             reinterpret_cast<jl_value_t*>(output_value_type_jl));
  jl_value_t *output_key_array_jl = jl_get_nth_field(output_tuple_jl, 0);
  jl_value_t *output_array_jl = jl_get_nth_field(output_tuple_jl, 1);

  size_t num_output_keys = jl_array_len(output_key_array_jl);
  output_keys->resize(num_output_keys);
  uint8_t *output_key_array = reinterpret_cast<uint8_t*>(jl_array_data(output_key_array_jl));
  memcpy(output_keys->data(), output_key_array, num_output_keys * sizeof(int64_t));

  size_t num_output_values = jl_array_len(output_array_jl);
  CHECK_EQ(num_output_values, num_output_keys);
  uint8_t *output_array = reinterpret_cast<uint8_t*>(jl_array_data(output_array_jl));
  output_values->resize(num_output_values * type::SizeOf(output_value_type));
  memcpy(output_values->data(), output_array, num_output_values * type::SizeOf(output_value_type));
  JL_GC_POP();
  CHECK(!jl_exception_occurred());
}

void
JuliaEvaluator::ExecForLoopTile(
    AbstractDistArrayPartition *iteration_space_partition,
    std::string exec_loop_func_name) {
  auto& dims = iteration_space_partition->GetDims();
  auto& keys = iteration_space_partition->GetKeys();
  void* values = iteration_space_partition->GetValues();

  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 6);

  jl_value_t* &dims_vec_jl = jl_values[0];
  jl_value_t* &keys_vec_jl = jl_values[1];
  jl_value_t* &values_vec_jl = jl_values[2];
  jl_value_t* &dims_array_type_jl = jl_values[3];
  jl_value_t* &keys_array_type_jl = jl_values[4];
  jl_value_t* &values_array_type_jl = jl_values[5];
  dims_array_type_jl = jl_apply_array_type(jl_int64_type, 1);
  keys_array_type_jl = jl_apply_array_type(jl_int64_type, 1);
  jl_datatype_t* value_type_jl =type::GetJlDataType(
      iteration_space_partition->GetValueType());
  values_array_type_jl = reinterpret_cast<jl_value_t*>(jl_apply_array_type(
      value_type_jl, 1));
  dims_vec_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      dims_array_type_jl, dims.data(), dims.size(), 0));
  keys_vec_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl, keys.data(), keys.size(), 0));
  values_vec_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      values_array_type_jl, values, keys.size(), 0));

  //LOG(INFO) << __func__;

  jl_function_t *exec_loop_func
      = GetFunction(jl_main_module, exec_loop_func_name.c_str());
  jl_call3(exec_loop_func, keys_vec_jl, values_vec_jl, dims_vec_jl);
  jl_value_t* exception_jl = jl_exception_occurred();
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
JuliaEvaluator::StaticExecForLoopTile(
    JuliaEvaluator *julia_eval,
    AbstractDistArrayPartition *iteration_space_partition,
    std::string exec_loop_func_name) {
  julia_eval->ExecForLoopTile(iteration_space_partition,
                          exec_loop_func_name);

}

}
}
