#pragma once

#include <functional>
#include <stdint.h>
#include <memory>

namespace orion {
namespace bosen {
struct WorkerTask {
 public:
  enum class Operator {
    kGetMaxIndex = 0,
      kRangePartitionTable = 1
  };

  const Operator op;
  const uint64_t param1, param2;
  uint64_t count {0};
  std::unique_ptr<uint8_t[]> custom_data;

  WorkerTask(Operator _op,
             uint64_t _param1,
             uint64_t _param2):
      op(_op),
      param1(_param1),
      param2(_param2) { }
};

}
}
