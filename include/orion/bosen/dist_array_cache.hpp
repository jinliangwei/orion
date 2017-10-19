#pragma once

#include <utility>

namespace orion {
namespace bosen {

class AbstractDistArrayPartition;

enum class PrefetchStatus {
  kNotPrefetched = 0,
  kPrefetchSent = 1,
  kPrefetchRecved = 2
};

using DistArrayCache = std::pair<PrefetchStatus, AbstractDistArrayPartition*>;

}
}
