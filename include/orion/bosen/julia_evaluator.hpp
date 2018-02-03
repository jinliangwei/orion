#pragma once
#include <string>
#include <julia.h>
#include <glog/logging.h>
#include <unordered_map>
#include <mutex>
#include <condition_variable>

#include <orion/bosen/type.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/julia_task.hpp>
#include <orion/bosen/julia_module.hpp>
#include <orion/bosen/task.pb.h>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/peer_recv_buffer.hpp>
#include <orion/bosen/dist_array_meta.hpp>

namespace orion {
namespace bosen {

class DistArray;
class AbstractDistArrayPartition;

class JuliaEvaluator {
 private:
  static void EvalExpr(const std::string &serialized_expr,
                       JuliaModule module,
                       Blob *result_buff);
  static jl_module_t* orion_worker_module_;
  static std::string lib_path_;
  static std::string orion_home_;
 public:
  JuliaEvaluator() { }
  ~JuliaEvaluator() { }
  static void Init(const std::string &orion_home,
                   size_t num_servers,
                   size_t num_executors);
  static void AtExitHook() { jl_atexit_hook(0); }
  static void ExecuteTask(JuliaTask* task);
  static jl_function_t* GetOrionWorkerFunction(const char* func_name);
  static jl_function_t* GetFunction(jl_module_t* module,
                                    const char* func_name);

  static void AbortIfException();
  static void BoxValue(
      type::PrimitiveType result_type,
      const uint8_t* value,
      jl_value_t **value_jl);

  static void UnboxValue(jl_value_t* value,
                         type::PrimitiveType result_type,
                         uint8_t* value_buff);
  static void ParseString(
    const char *str,
    JuliaModule parser_func_module,
    const char *parser_func_name,
    std::vector<int64_t> *key, // key memory is pre-allocated
    size_t num_dims,
    jl_value_t **value_ptr);

  static void ParseStringFlatten(
      const char *str,
      JuliaModule parser_func_module,
      const char *parser_func_name,
      std::vector<int64_t> *key,
      size_t num_dims,
      jl_value_t *value_type,
      jl_value_t **value_ptr);

  static void ParseStringWithLineNumber(
      size_t line_number,
      const char *str,
      JuliaModule parser_func_module,
      const char *parser_func_name,
      std::vector<int64_t> *key, // key memory is pre-allocated
      size_t num_dims,
      jl_value_t **value_ptr);

  static void ParseStringWithLineNumberFlatten(
      size_t line_numebr,
      const char *str,
      JuliaModule parser_func_module,
      const char *parser_func_name,
      std::vector<int64_t> *key,
      size_t num_dims,
      jl_value_t *value_type,
      jl_value_t **value_ptr);

  static void ParseStringValueOnly(
      const char *str,
      JuliaModule parser_func_module,
      const char *parser_func_name,
      size_t num_dims,
      jl_value_t **value_ptr);

  static void ParseStringValueOnlyWithLineNumber(
      size_t line_numebr,
      const char *str,
      JuliaModule parser_func_module,
      const char *parser_func_name,
      size_t num_dims,
      jl_value_t **value_ptr);

  static void DefineDistArray(int32_t id,
                              const std::string &symbol,
                              const std::string &serialized_value_type,
                              const std::vector<int64_t> &dims,
                              bool is_dense,
                              bool is_buffer,
                              const std::vector<uint8_t> &serialized_init_value_vec);

  static void RandNormal(
      type::PrimitiveType value_type,
      jl_value_t **values_ptr,
      size_t num_values);

  static void RandUniform(
      type::PrimitiveType value_type,
      jl_value_t **values_ptr,
      size_t num_values);

  static void RunMapGeneric(
      DistArrayMapType map_type,
      const std::vector<int64_t> &parent_dims,
      const std::vector<int64_t> &child_dims,
      size_t num_keys,
      int64_t *keys,
      jl_value_t *input_values,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name,
      std::vector<int64_t>* output_keys,
      jl_value_t *output_value_type,
      jl_value_t **output_values_ptr);

  static void RunMap(
      const std::vector<int64_t> &parent_dims,
      const std::vector<int64_t> &child_dims,
      size_t num_keys,
      int64_t *keys,
      jl_value_t *input_values,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name,
      std::vector<int64_t>* output_keys,
      jl_value_t *output_value_type,
      jl_value_t **output_values_ptr);

  static void RunMapFixedKeys(
      const std::vector<int64_t> &parent_dims,
      size_t num_keys,
      int64_t *keys,
      jl_value_t *input_values,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name,
      jl_value_t *output_value_type,
      jl_value_t **output_values_ptr);

  static void RunMapValues(
      const std::vector<int64_t> &child_dims,
      size_t num_values,
      jl_value_t *input_values,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name,
      jl_value_t *output_value_type,
      jl_value_t **output_values_ptr);

  static void RunMapValuesNewKeys(
      const std::vector<int64_t> &child_dims,
      size_t num_values,
      jl_value_t *input_values,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name,
      std::vector<int64_t>* output_keys,
      jl_value_t *output_value_type,
      jl_value_t **output_values_ptr);

  static void GetVarValue(
      const std::string &symbol,
      Blob *result_buff);

  static void SetVarValue(
      const std::string &symbol,
      uint8_t *serialized_value,
      size_t value_size);

  static void CombineVarValue(
      const std::string &symbol,
      uint8_t *serialized_value_to_combine,
      size_t value_size,
      const std::string &combiner);

  static void SetDistArrayDims(
      const std::string &dist_array_sym,
      const std::vector<int64_t> &dims);

  static void GetDistArray(
      const std::string &dist_array_sym,
      jl_value_t** dist_array_ptr);

  static void GetDistArrayValueType(
      const std::string &dist_array_sym,
      jl_datatype_t **value_type_ptr);

  static void GetDistArrayValueType(
      jl_value_t *dist_array_jl,
      jl_datatype_t **value_type_ptr);

  static void GetAndSerializeValue(DistArray *dist_array,
                                   int64_t key, Blob *bytes_buff);
  static void GetAndSerializeValues(std::unordered_map<int32_t, DistArray> *dist_arrays,
                                    const uint8_t *request,
                                    Blob *bytes_buff);
};

}
}
