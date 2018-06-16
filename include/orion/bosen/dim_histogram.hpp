#pragma once

#include <unordered_map>

namespace orion {
namespace bosen {

struct DimHistogram {
  std::unordered_map<int64_t, size_t> bin_to_counts;
};

}
}
