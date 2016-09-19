#pragma once

#include <memory>
#include <orion/helper.hpp>
#include <orion/noncopyable.hpp>
#include <random>

namespace orion {
namespace bosen {

class LocalTable {
 private:
  const size_t rank_;
  const size_t num_rows_;
  const int32_t x_offset_;
  std::unique_ptr<float[]> data_;
 public:
  LocalTable(size_t rank, size_t num_rows,
             int32_t x_offset):
      rank_(rank),
      num_rows_(num_rows),
      x_offset_(x_offset),
      data_(std::make_unique<float[]>(num_rows * rank)) {
    std::mt19937 gen(1);
    std::uniform_real_distribution<> dist(0, 1);
    for (size_t i = 0; i < rank_ * num_rows_; i++) {
      data_[i] = 0.1;
    }
  }

  ~LocalTable() { }
  DISALLOW_COPY(LocalTable);


  float *get_row(int32_t row_id) {
    int32_t row_index = row_id - x_offset_;
    return data_.get() + row_index * rank_;
  }
};

}
}
