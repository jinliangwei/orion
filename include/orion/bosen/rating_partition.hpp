#pragma once

#include <orion/noncopyable.hpp>
#include <stdint.h>
#include <memory>

namespace orion {
namespace bosen {
struct Rating {
  int32_t x;
  int32_t y;
  float v;
};
using RatingBuffer = std::pair<std::unique_ptr<Rating[]>, size_t>;

struct RatingPartition {
  std::unique_ptr<Rating[]> ratings;
  size_t num_ratings {0};
  int32_t y_min {0}, y_max {0};

  RatingPartition() = default;
  DISALLOW_COPY(RatingPartition);

  RatingPartition(RatingPartition && other):
      ratings(std::move(other.ratings)),
      num_ratings(other.num_ratings),
      y_min(other.y_min),
      y_max(other.y_max) { }
};
}
}
