#pragma once

#include <vector>
#include <unordered_map>
namespace orion {
namespace bosen {

enum QueryDelimiter {
  kPointQueryStart = 0,
  kRangeQueryStart = 1
};

using PointQueryKeyVec = std::vector<int64_t>;
using RangeQueryKeyVec = std::vector<std::pair<int64_t, size_t>>;
using PointQueryKeyDistArrayMap = std::unordered_map<int32_t, PointQueryKeyVec>;
using RangeQueryKeyDistArrayMap = std::unordered_map<int32_t, RangeQueryKeyVec>;

}
}
