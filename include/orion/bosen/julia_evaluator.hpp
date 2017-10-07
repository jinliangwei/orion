#pragma once
#include <string>
#include <julia.h>
#include <glog/logging.h>
#include <unordered_map>

#include <orion/bosen/type.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/julia_task.hpp>
#include <orion/bosen/julia_module.hpp>
#include <orion/bosen/task.pb.h>

namespace orion {
namespace bosen {

class DistArray;
class AbstractDistArrayPartition;

class JuliaEvaluator {
 private:
  jl_value_t* BoxValue(
      type::PrimitiveType result_type,
      const uint8_t* value);

  void UnboxValue(jl_value_t* value,
                   type::PrimitiveType result_type,
                   Blob *result_buff);
  void EvalExpr(const std::string &serialized_expr,
                JuliaModule module,
                Blob *result_buff);
  jl_module_t* orion_worker_module_;
  std::string lib_path_;
  std::string orion_home_;
 public:
  JuliaEvaluator() { }
  ~JuliaEvaluator() { }
  void Init(const std::string &orion_home);
  void AtExitHook() { jl_atexit_hook(0); }
  void ExecuteTask(JuliaTask* task);
  jl_function_t* GetFunction(jl_module_t* module,
                             const char* func_name);
  void SetResult(jl_value_t *result,
                 Blob *result_buff);

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

  void DefineVar(std::string var_name,
                 std::string var_value);
  static void StaticDefineVar(
      JuliaEvaluator *julia_eval,
      std::string var_name,
      std::string var_value);

  void ComputeSpaceTimeRepartition(
      std::string repartition_func_name,
      DistArray *dist_array);

  static void StaticComputeSpaceTimeRepartition(
      JuliaEvaluator *julia_eval,
      std::string repartition_func_name,
      DistArray *dist_array);

  void RandNormal(
      type::PrimitiveType type,
      uint8_t *buff,
      size_t array_len);

  void RunMapGeneric(
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
      const std::string &mapper_func_name);

  void RunMap(
      std::vector<int64_t> dims,
      size_t num_keys,
      int64_t *keys,
      type::PrimitiveType input_value_type,
      uint8_t *input_values,
      std::vector<int64_t>* output_keys,
      type::PrimitiveType output_value_type,
      Blob *output_values,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name);

  void RunMapFixedKeys(
      std::vector<int64_t> dims,
      size_t num_keys,
      int64_t *keys,
      type::PrimitiveType input_value_type,
      uint8_t *input_values,
      type::PrimitiveType output_value_type,
      Blob *output_values,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name);

  void RunMapValues(
      std::vector<int64_t> dims,
      size_t num_keys,
      type::PrimitiveType input_value_type,
      uint8_t *input_values,
      type::PrimitiveType output_value_type,
      Blob *output_values,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name);

  void RunMapValuesNewKeys(
      std::vector<int64_t> dims,
      size_t num_keys,
      type::PrimitiveType input_value_type,
      uint8_t *input_values,
      std::vector<int64_t>* output_keys,
      type::PrimitiveType output_value_type,
      Blob *output_values,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name);
};

}
}
