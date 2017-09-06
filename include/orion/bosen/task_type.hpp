#pragma once

namespace orion {
namespace bosen {

enum class TaskType {
  kNone = 0,
    kExecJuliaFunc = 2,
    kExecCppFunc = 3,
    kEvalJuliaExpr = 4
};

}
}
