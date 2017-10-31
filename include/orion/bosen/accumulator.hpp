#pragma once

#include <string>

namespace orion {
namespace bosen {

struct Accumulator {
  std::string symbol;
  std::string combiner;
  size_t num_accumulated;
};

}
}
