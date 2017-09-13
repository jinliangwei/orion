#pragma once
#include <string>
#include <julia.h>
#include <glog/logging.h>
#include <unordered_map>

#include <orion/bosen/type.hpp>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/julia_task.hpp>
#include <orion/bosen/julia_module.hpp>

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
  jl_module_t* orion_gen_module_;
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

  void ReloadOrionGenModule();

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
};

}
}
