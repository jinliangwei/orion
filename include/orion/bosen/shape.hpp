#pragma once
#include <stddef.h>
#include <string.h>

namespace orion {
namespace bosen {
template<size_t kNumDim = 1>
class Shape {
 private:
  size_t dims_[kNumDim];
  size_t strides_[kNumDim];
 public:
  explicit Shape(const size_t *dims) {
    memcpy(dims_, dims, sizeof(size_t) * kNumDim);

    for (int i = 0; i < kNumDim; ++i) {
      strides_[i] = 1;
      for (int j = i + 1; j < kNumDim; ++j) {
        strides_[i] *= dims_[j];
      }
    }
  }
  ~Shape() { }

  Shape(const Shape &other) = default;
  Shape & operator = (const Shape &other) = default;
  size_t get_stride(size_t dim) const {
    return strides_[dim];
  }

  size_t get_offset(const size_t *dims) const {
    size_t offset = 0;
    for (int i = 0; i < kNumDim; ++i) {
      offset += dims[i] * strides_[i];
    }
    return offset;
  }

  size_t get_offset(const size_t *dims, size_t prefix_len) const {
    size_t offset = 0;
    for (int i = 0; i < prefix_len; ++i) {
      offset += dims[i] * strides_[i];
    }
    return offset;
  }

  size_t get_num_keys(size_t prefix_len) const {
    size_t num_keys = 1;
    for (int i = kNumDim - 1; i < prefix_len; ++i) {
      num_keys *= strides_[i];
    }
    return num_keys;
  }

  void get_dims(size_t offset, size_t *dims) const {
    for (int i = 0; i < kNumDim; ++i) {
      dims[i] = offset / strides_[i];
      offset %= strides_[i];
    }
  }
};
}
}
