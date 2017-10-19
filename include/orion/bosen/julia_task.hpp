#pragma once

#include <functional>
#include <string>
#include <orion/bosen/julia_module.hpp>
#include <orion/bosen/type.hpp>
#include <orion/bosen/blob.hpp>

namespace orion {
namespace bosen {

enum class TaskLabel {
  kNone = 0,
    kLoadDistArrayFromTextFile = 1,
    kRandomInitDistArray = 2,
    kDefineVar = 3,
    kComputeRepartition = 4,
    kDefineJuliaDistArray = 5,
    kExecForLoopTile = 6

};

class JuliaEvaluator;

class JuliaTask {
 protected:
  JuliaTask() { }
  virtual ~JuliaTask() { }
};

class ExecJuliaFuncTask : public JuliaTask {
 public:
  std::string function_name;
  type::PrimitiveType result_type;
  Blob result_buff;
};

class ExecCppFuncTask : public JuliaTask {
 public:
  std::function<void(JuliaEvaluator*)> func;
  Blob result_buff;
  TaskLabel label {TaskLabel::kNone};
};

class EvalJuliaExprTask : public JuliaTask {
 public:
  std::string serialized_expr;
  JuliaModule module;
  Blob result_buff;
};


}
}
