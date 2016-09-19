#pragma once

#include <functional>
#include <stdint.h>

namespace orion {
namespace bosen {
class DriverRuntime;
class DriverThread;

using DriverFunc = std::function<int(DriverRuntime*,
                                     DriverThread*)>;

struct DriverTask {
 public:
  enum class Inst {
    kExecuteFunc = 0,
      kExecuteFuncWaitReturn = 1,
      kRepeatForN = 2,
      kRepeatIfTrue = 3,
      kStall = 4
  };

  const Inst inst;
  const uint64_t param1 {0}, param2 {0};
  uint64_t count {0};
  const DriverFunc f;
  DriverTask(Inst _inst):
      inst(_inst) { }

  DriverTask(Inst _inst,
             uint64_t _param1,
             uint64_t _param2,
             DriverFunc _f):
      inst(_inst),
      param1(_param1),
      param2(_param2),
      f(_f) { }
};

}
}
