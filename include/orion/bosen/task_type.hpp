#pragma once

namespace orion {
namespace bosen {

enum class TaskType {
  kNone = 0,
  kExecJuliaCode = 1,
    kExecJuliaFunc = 2,
    kExecCppFunc = 3,
    kEvalJuliaAst = 4
};

}
}
