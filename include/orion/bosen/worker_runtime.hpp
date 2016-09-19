#pragma once

#include <memory>

#include <orion/bosen/table.hpp>
#include <orion/bosen/table/data_table.hpp>
#include <orion/noncopyable.hpp>
#include <orion/helper.hpp>
#include <orion/constants.hpp>
#include <orion/bosen/table/local_table.hpp>
#include <orion/bosen/table/ring_table.hpp>
#include <orion/perf.hpp>
#include <stdint.h>
#include <list>
#include <unordered_map>
#include <vector>
#include <cblas.h>

namespace orion {
namespace bosen {

class WorkerRuntime;

class WorkerRuntime {
 private:
  std::unique_ptr<DataTable> data_table_;
  std::unique_ptr<LocalTable> x_table_;
  std::unique_ptr<RingTable> y_table_;
  size_t num_executors_;
  const size_t my_id_;
  const size_t num_local_executors_;
  int32_t sub_epoch_ {0};
  int32_t eval_sub_epoch_ {0};
  int32_t num_sub_epochs_per_iteration_ {0};
  int32_t num_iterations_ {0};
  int32_t num_iterations_completed_ {0};
  bool sgd_ {true};
  std::vector<float> error_each_iteration_;
  float step_size_ = 0.001;
  const float step_size_decay_ = 0.99;
  size_t rank_ {0};
  bool sgd_all_done_ {false};
  const bool is_edge_in_;
  const bool is_edge_out_;
  const std::vector<PerfCount::CountType> kPerfCountTypes {
    PerfCount::PERF_COUNT_TYPE_HW_CPU_CYCLES,
      PerfCount::PERF_COUNT_TYPE_HW_INSTRUCTIONS,
      PerfCount::PERF_COUNT_TYPE_HW_CACHE_REFERENCES,
      PerfCount::PERF_COUNT_TYPE_HW_CACHE_MISSES,
      PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_READ_ACCESS,
      PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_WRITE_ACCESS };
  PerfCount perf_count_;

 public:
  WorkerRuntime(size_t num_executors,
                size_t num_local_executors,
                size_t my_id):
      num_executors_(num_executors),
      my_id_(my_id),
      num_local_executors_(num_local_executors),
      is_edge_in_((num_local_executors == num_executors)
                  ? false
                  : ((num_local_executors == 1)
                     ? true
                     : ((my_id_ + 1) % num_local_executors == 0))),
      is_edge_out_((num_local_executors == num_executors)
                   ? false
                   : ((num_local_executors == 1)
                      ? true
                      : (my_id_ % num_local_executors == 0))),
    perf_count_(kPerfCountTypes) { }
  ~WorkerRuntime() { }

  bool is_edge_in() const {
    return is_edge_in_;
  }

  bool is_edge_out() const {
    return is_edge_out_;
  }

  void LoadData(const char *data_path) {
    data_table_ = std::make_unique<DataTable>(data_path, num_executors_, my_id_);
    data_table_->Init();
  }

  DataTable *get_data_table() {
    return data_table_.get();
  }

  void LogArray(const char *header, const float *array, size_t len) {
    std::string s(header);
    for (size_t i = 0; i < len; i++) {
      s += " ";
      s += std::to_string(array[i]);
    }
    LOG(INFO) << s;
  }

  void InitializeParams(size_t rank, size_t pipeline_depth) {
    x_table_ = std::make_unique<LocalTable>(rank, data_table_->get_num_rows_per_partition(),
                                       data_table_->get_local_x_min());
    y_table_ = std::make_unique<RingTable>(rank, data_table_->get_num_cols_per_partition(),
                                      num_executors_, pipeline_depth,
                                      data_table_->get_data_partitions(),
                                      data_table_->get_data_partition_local_begin());
    num_sub_epochs_per_iteration_ = num_executors_ * pipeline_depth;
    sub_epoch_ = 0;
    rank_ = rank;
  }

  LocalTable *get_x_table() {
    return x_table_.get();
  }

  RingTable *get_y_table() {
    return y_table_.get();
  }

  void BeginSGDEval() {
    data_table_->BeginIterating();
    sub_epoch_ = 0;
  }

  size_t get_num_subepochs_per_iteration() {
    return num_sub_epochs_per_iteration_;
  }

  bool sgdeval_done_one_iteration() {
    return (sub_epoch_ == num_sub_epochs_per_iteration_);
  }

  void SGDOnePartition(ParamPartition *y_partition,
		       const RatingPartition &data_partition) {
    LOG(INFO) << my_id_ << " Master " << __func__ << " start";
    perf_count_.Start();
    CHECK_EQ(data_partition.y_min, y_partition->y_min);
    CHECK_EQ(data_partition.y_max, y_partition->y_max);
    Rating *ratings = data_partition.ratings.get();
    size_t num_ratings = data_partition.num_ratings;
    for (size_t i = 0; i < num_ratings; ++i) {
      Rating &rating = ratings[i];
      int32_t x_id = rating.x;
      int32_t y_id = rating.y;
      float *W_row = x_table_->get_row(x_id);
      float *H_row = y_partition->get_col(y_id, rank_);
      float est = cblas_sdot(rank_, W_row, 1, H_row, 1);
      CHECK_EQ(est, est) << "x_id = " << x_id << " y_id = " << y_id;
      float diff = rating.v - est;
      cblas_saxpy(rank_, step_size_ * 2 * diff, H_row, 1, W_row, 1);
      cblas_saxpy(rank_, step_size_ * 2 * diff, W_row, 1, H_row, 1);
    }
    perf_count_.Stop();
    LOG(INFO) << my_id_ << " Master " << __func__ << " done; num ratings = "
              << num_ratings;
  }

  void EvalOnePartition(ParamPartition *y_partition,
		       const RatingPartition &data_partition) {
    LOG(INFO) << my_id_ << " Master " << __func__ << " start";
    perf_count_.Start();
    CHECK_EQ(data_partition.y_min, y_partition->y_min);
    CHECK_EQ(data_partition.y_max, y_partition->y_max);
    Rating *ratings = data_partition.ratings.get();
    size_t num_ratings = data_partition.num_ratings;
    for (size_t i = 0; i < num_ratings; ++i) {
      Rating &rating = ratings[i];
      int32_t x_id = rating.x;
      int32_t y_id = rating.y;
      float *W_row = x_table_->get_row(x_id);
      float *H_row = y_partition->get_col(y_id, rank_);
      float est = cblas_sdot(rank_, W_row, 1, H_row, 1);
      float diff = rating.v - est;
      error_each_iteration_[num_iterations_completed_] += diff * diff;
    }
    perf_count_.Stop();
    LOG(INFO) << my_id_ << " Master " << __func__
    	      << " done; local error = " << error_each_iteration_[num_iterations_completed_];
  }

  void SGDEvalOneIteration() {
    auto y_partition = y_table_->compute_get_first_partition();
    CHECK(y_partition != nullptr);
    auto &data_partition = data_table_->NextPartition();
    SGDOnePartition(y_partition, data_partition);
    EvalOnePartition(y_partition, data_partition);
  }

  bool SGDOneSubEpoch() {
    auto y_partition = y_table_->compute_get_next_partition();
    if (y_partition == nullptr) return false;
    auto &data_partition = data_table_->NextPartition();
    SGDOnePartition(y_partition, data_partition);

    sub_epoch_++;
    return true;
  }

  bool EvalOneSubEpoch() {
    auto y_partition = y_table_->compute_get_next_partition();
    if (y_partition == nullptr) return false;
    auto &data_partition = data_table_->NextPartition();
    EvalOnePartition(y_partition, data_partition);
    sub_epoch_++;
    return true;
  }

  void set_num_iterations(int32_t num_iterations) {
    num_iterations_ = num_iterations;
    error_each_iteration_.resize(num_iterations);
  }

  void inc_num_completed_iterations() {
    //LOG(INFO) << my_id_ << " Master " << __func__
    //	      << " iteration " << num_iterations_completed_
    //          << " error = " << error_each_iteration_[num_iterations_completed_];
    num_iterations_completed_++;
    step_size_ *= step_size_decay_;
  }

  bool completed_all_iterations() {
    return num_iterations_completed_ == num_iterations_;
  }

  void toggle_sgd() {
    sgd_ = !sgd_;
  }

  bool is_sgd() {
    return sgd_;
  }

  bool is_sgd_all_done() {
    return sgd_all_done_;
  }

  void set_sgd_all_done() {
    sgd_all_done_ = true;
  }

  float *get_errors() {
    return error_each_iteration_.data();
  }

  int32_t get_num_iterations() {
    return num_iterations_;
  }

  void PrintPerfCounts() {
    LOG(INFO) << my_id_ << " Master " << __func__ << " PerfCount "
	      << "CPU_CYCLES "
	      << perf_count_.GetByIndex(0)
	      << " HW_INSTRUCTIONS "
	      << perf_count_.GetByIndex(1)
	      << " HW_CACHE_REFERENCES "
	      << perf_count_.GetByIndex(2)
	      << " HW_CACHE_MISSES "
	      << perf_count_.GetByIndex(3)
	      << " L1D_READ_ACCESS "
	      << perf_count_.GetByIndex(4)
	      << " L1D_WRITE_ACCESS "
	      << perf_count_.GetByIndex(5);
  }
};

}
}
