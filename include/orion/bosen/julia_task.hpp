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
    kParseDistArrayTextBuffer = 2,
    kInitDistArray = 3,
    kMapDistArray = 4,
    kDefineVar = 5,
    kComputeRepartition = 6,
    kRepartitionSerialize = 7,
    kRepartitionDeserialize = 8,
    kDefineJuliaDistArray = 9,
    kDefineJuliaDistArrayBuffer = 10,
    kGetAccumulatorValue = 11,
    kSetDistArrayDims = 12,
    kExecForLoopPartition = 13,
    kComputePrefetchIndices = 14,
    kSerializeGlobalIndexedDistArrays = 15,
    kSerializeDistArrayTimePartitions = 16,
    kDeserializeGlobalIndexedDistArrays = 17,
    kDeserializeDistArrayTimePartitions = 18,
    kGetAndSerializeDistArrayValues = 19,
    kCachePrefetchDistArrayValues = 20
          };

class JuliaTask {
 protected:
  JuliaTask() { }
  virtual ~JuliaTask() { }
};

class ExecCppFuncTask : public JuliaTask {
 public:
  std::function<void()> func;
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
