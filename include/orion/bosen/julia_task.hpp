#pragma once

#include <functional>

namespace orion {
namespace bosen {

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

class ExecJuliaCodeTask : public JuliaTask {
 public:
  std::string code;
  type::PrimitiveType result_type;
  Blob result_buff;
};

class ExecCppFuncTask : public JuliaTask {
 public:
  std::function<void(JuliaEvaluator*)> func;
};


}
}
