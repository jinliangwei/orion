#pragma once

#include <orion/bosen/abstract_exec_for_loop.hpp>
#include <orion/bosen/dist_array.hpp>
#include <unordered_map>
#include <glog/logging.h>

namespace orion {
namespace bosen {

class ExecForLoopSpaceTime : public AbstractExecForLoop {
 private:
  std::unordered_map<int32_t, DistArray*> space_partitioned_dist_arrays_;
  std::unordered_map<int32_t, DistArray*> time_partitioned_dist_arrays_;
  std::unordered_map<int32_t, DistArray*> global_indexed_dist_arrays_;
  DistArray *iteration_space_;
  int32_t clock_;
  bool is_ordered_;
  int32_t curr_space_id_;
 public:
  enum RunnableStatus {
    kRunnable = 0,
    kMissingTimePartitionedDistArray = 1,
    kMissingGlobalIndexedDistArray = 2,
    kCompleted = 3
  };

  ExecForLoopSpaceTime(
      int32_t iteration_space_id,
      const int32_t *space_partitioned_dist_array_ids,
      size_t num_space_partitioned_dist_arrays,
      const int32_t *time_partitioned_dist_array_ids,
      size_t num_time_partitioned_dist_arrays,
      const int32_t *global_indexed_dist_array_ids,
      size_t num_gloabl_indexed_dist_arrays,
      const char* loop_batch_func_name,
      bool is_ordered,
      std::unordered_map<int32_t, DistArray> &dist_arrays);
  virtual ~ExecForLoopSpaceTime() { }

  RunnableStatus GetRunnableStatus() const;

};

ExecForLoopSpaceTime::ExecForLoopSpaceTime(
    int32_t iteration_space_id,
    const int32_t *space_partitioned_dist_array_ids,
    size_t num_space_partitioned_dist_arrays,
    const int32_t *time_partitioned_dist_array_ids,
    size_t num_time_partitioned_dist_arrays,
    const int32_t *global_indexed_dist_array_ids,
    size_t num_global_indexed_dist_arrays,
    const char* loop_batch_func_name,
    bool is_ordered,
    std::unordered_map<int32_t, DistArray> &dist_arrays):
    clock_(0),
    is_ordered_(is_ordered),
    curr_space_id_(0) {
  auto iter = dist_arrays.find(iteration_space_id);
  CHECK(iter != dist_arrays.end());
  iteration_space_ = &(iter->second);

  for (size_t i = 0; i < num_space_partitioned_dist_arrays; i++) {
    int32_t id = space_partitioned_dist_array_ids[i];
    iter = dist_arrays.find(id);
    CHECK(iter != dist_arrays.end());
    space_partitioned_dist_arrays_.emplace(std::make_pair(id, &(iter->second)));
  }

  for (size_t i = 0; i < num_time_partitioned_dist_arrays; i++) {
    int32_t id = time_partitioned_dist_array_ids[i];
    iter = dist_arrays.find(id);
    CHECK(iter != dist_arrays.end());
    time_partitioned_dist_arrays_.emplace(std::make_pair(id, &(iter->second)));
  }

  for (size_t i = 0; i < num_global_indexed_dist_arrays; i++) {
    int32_t id = global_indexed_dist_array_ids[i];
    iter = dist_arrays.find(id);
    CHECK(iter != dist_arrays.end());
    global_indexed_dist_arrays_.emplace(std::make_pair(id, &(iter->second)));
  }
  CHECK(!is_ordered_);
}

ExecForLoopSpaceTime::RunnableStatus
ExecForLoopSpaceTime::GetRunnableStatus() const {
  return ExecForLoopSpaceTime::kCompleted;
}

}
}
