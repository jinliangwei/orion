#pragma once
#include <string>
#include <julia.h>

#include <orion/bosen/type.hpp>
#include <orion/bosen/blob.hpp>

namespace orion {
namespace bosen {

class JuliaTask {
 protected:
  JuliaTask() { }
  virtual ~JuliaTask() { }
};

class JuliaCallFuncTask : private JuliaTask {
 public:
  std::string function_name;
  type::PrimitiveType result_type;
  Blob result_buff;
};

class JuliaExecuteCodeTask : private JuliaTask {
 public:
  std::string code;
  type::PrimitiveType result_type;
  Blob result_buff;
};

class JuliaEvaluator {
 private:
  void UnboxResult(jl_value_t* value, type::PrimitiveType result_type,
                   Blob *result_buff);
 public:
  JuliaEvaluator() { }
  ~JuliaEvaluator() { }
  void Init() { jl_init(NULL); }
  void AtExitHook() { jl_atexit_hook(0); }
  void ExecuteTask(const JuliaTask* task);
};

void
JuliaEvaluator::ExecuteTask(const JuliaTask* task) {

}


}
}
