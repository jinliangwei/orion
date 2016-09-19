#pragma once
#include <memory>
#include <orion/helper.hpp>
#include <vector>
#include <orion/bosen/rating_partition.hpp>

namespace orion {
namespace bosen {

struct ParamPartition {
  std::unique_ptr<float[]> params;
  int32_t y_min {0}, y_max {0};

  float *get_col(int32_t y_id, size_t rank) {
    int32_t idx = y_id - y_min;
    CHECK(idx >= 0) << "y_id = " << y_id;
    return params.get() + rank * idx;
  }

  float *get_params() {
    return params.get();
  }
};

class RingTable {
 private:
  const size_t rank_;
  const size_t num_cols_per_partition_;
  const size_t num_executors_;
  const size_t pipeline_depth_;
  std::vector<ParamPartition> partitions_;
  size_t compute_start_idx_;
  size_t compute_avai_size_;
  size_t server_start_idx_;
  size_t server_avai_size_;
  size_t server_ptr_start_idx_;
  size_t server_ptr_avai_size_;
  size_t client_start_idx_;
  size_t client_avai_size_;
  size_t server_dest_capacity_ {0};
 public:
  RingTable(size_t rank, size_t num_cols_per_partition,
            size_t num_executors, size_t pipeline_depth,
            const std::vector<RatingPartition> &data_partitions,
            size_t data_partitions_begin):
      rank_(rank),
      num_cols_per_partition_(num_cols_per_partition),
      num_executors_(num_executors),
      pipeline_depth_(pipeline_depth),
      partitions_(pipeline_depth * 2),
      compute_start_idx_(0),
      compute_avai_size_(pipeline_depth),
      server_start_idx_(0),
    server_avai_size_(0),
    server_ptr_start_idx_(0),
    server_ptr_avai_size_(0),
    client_start_idx_(pipeline_depth),
    client_avai_size_(pipeline_depth) {
    for (auto &partition : partitions_) {
      partition.params = std::make_unique<float[]>(num_cols_per_partition_ * rank_);
    }

    std::mt19937 gen(1);
    std::uniform_real_distribution<> dist(0, 1);
    CHECK_LT(data_partitions_begin, data_partitions.size());
    for (size_t i = 0; i < pipeline_depth; i++) {
      partitions_[i].y_min = data_partitions[data_partitions_begin + i].y_min;
      partitions_[i].y_max = data_partitions[data_partitions_begin + i].y_max;

      for (size_t j = 0; j < num_cols_per_partition_ * rank_; j++) {
	partitions_[i].params.get()[j] = 0.1;
      }
    }
  }
  ~RingTable() { }
  DISALLOW_COPY(RingTable);
  int32_t get_rank() {
    return rank_;
  }

  ParamPartition *compute_get_first_partition() {
    return &partitions_[0];
  }

  ParamPartition *compute_get_next_partition() {
    if (compute_avai_size_ == 0) return nullptr;
    return &partitions_[compute_start_idx_];
  }

  void compute_finish_one_partition() {
    compute_avai_size_--;
    compute_start_idx_++;
    compute_start_idx_ %= partitions_.size();
  }

  void compute_add_one_partition() {
    compute_avai_size_++;
  }

  ParamPartition *server_get_next_partition() {
    if (server_avai_size_ == 0) return nullptr;
    return &partitions_[server_start_idx_];
  }

  void server_finish_one_partition() {
    server_avai_size_--;
    server_start_idx_++;
    server_start_idx_ %= partitions_.size();
  }

  void server_add_one_partition() {
    server_avai_size_++;
  }

  ParamPartition *server_ptr_get_next_partition() {
    if (server_ptr_avai_size_ == 0) return nullptr;
    return &partitions_[server_ptr_start_idx_];
  }

  void server_ptr_finish_one_partition() {
    server_ptr_avai_size_--;
    server_ptr_start_idx_++;
    server_ptr_start_idx_ %= partitions_.size();
  }

  void server_ptr_add_one_partition() {
    server_ptr_avai_size_++;
  }

  ParamPartition *client_get_next_partition() {
    if (client_avai_size_ == 0) return nullptr;
    return &partitions_[client_start_idx_];
  }

  void client_finish_one_partition() {
    client_avai_size_--;
    client_start_idx_++;
    client_start_idx_ %= partitions_.size();
  }

  void client_add_one_partition() {
    client_avai_size_++;
  }

  size_t client_get_recv_capacity() {
    return client_avai_size_;
  }

  void server_inc_dest_capacity(size_t delta) {
    server_dest_capacity_ += delta;
  }

  void server_dec_dest_capacity() {
    CHECK_GT(server_dest_capacity_, 0);
    server_dest_capacity_--;
  }

  size_t server_get_dest_capacity() {
    return server_dest_capacity_;
  }
};

}
}
