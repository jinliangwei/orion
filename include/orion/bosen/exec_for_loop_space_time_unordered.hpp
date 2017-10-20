#pragma once

#include <orion/bosen/abstract_exec_for_loop.hpp>
#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_cache.hpp>
#include <unordered_map>
#include <glog/logging.h>
#include <algorithm>

namespace orion {
namespace bosen {

// Unordered:
// Number of clocks: num_clocks = max(num_executors, num_space_time_partitions)
// For each clock, execute all local partitions whose time partition is
// (ExecutorId + clock_) % num_clocks

class ExecForLoopSpaceTimeUnordered : public AbstractExecForLoop {
 private:
  std::unordered_map<int32_t, DistArray*> space_partitioned_dist_arrays_;
  std::unordered_map<int32_t, DistArray*> time_partitioned_dist_arrays_;
  std::unordered_map<int32_t, DistArray*> global_indexed_dist_arrays_;
  std::unordered_map<int32_t, DistArrayCache> dist_array_cache_;

  DistArray *iteration_space_;
  int32_t clock_;
  const int32_t kExecutorId;
  const size_t kNumExecutors;
  int32_t kNumSpacePartitions {0};
  int32_t kNumTimePartitions {0};
  int32_t num_clocks_ {0};
  const std::string kExecLoopFuncName;
  int32_t curr_space_partition_id_ {kExecutorId};
  int32_t curr_time_partition_offset_ {0};
  int32_t curr_running_time_partition_ {-1};
  std::vector<int32_t> completed_time_partitions_;
 public:
  enum class RunnableStatus {
    kRunnable = 0,
    kMissingDep = 1,
    kPrefretchGlobalIndexedDep = 2,
  };

  ExecForLoopSpaceTimeUnordered(
      int32_t executor_id,
      size_t num_executors,
      int32_t iteration_space_id,
      const int32_t *space_partitioned_dist_array_ids,
      size_t num_space_partitioned_dist_arrays,
      const int32_t *time_partitioned_dist_array_ids,
      size_t num_time_partitioned_dist_arrays,
      const int32_t *global_indexed_dist_array_ids,
      size_t num_gloabl_indexed_dist_arrays,
      const char* exec_loop_func_name,
      std::unordered_map<int32_t, DistArray> &dist_arrays);
  virtual ~ExecForLoopSpaceTimeUnordered();

  RunnableStatus GetRunnableStatus(int32_t time_partition_to_exec) const;
  bool IsCompleted() const { return clock_ == num_clocks_; }
  int32_t GetNumClocks() const { return num_clocks_; }
  int32_t GetDestExecutorId() const;
  void PrepareToExecCurrentTile(int32_t time_partition_to_exec);

  const std::string &GetExecLoopFuncName() { return kExecLoopFuncName; }

  void
  GetDistArrayTimePartitionsToSend(std::unordered_map<int32_t, AbstractDistArrayPartition*>*
                                   partitions);

  void
  GetDistArrayGlobalIndexedPartitionsToSend(std::unordered_map<int32_t, AbstractDistArrayPartition*>*
                                            partitions);
  AbstractDistArrayPartition *GetNextPartitionToExec(int32_t *space_id, int32_t *time_id);
  int32_t GetCurrRunningTimePartitionId() const { return curr_running_time_partition_; }

  void SetCurrRunningTimePartitionId(int32_t time_partition_to_exec) { curr_running_time_partition_ = time_partition_to_exec; }
  int32_t HasAllInitialTimePartitions();

  int32_t GetNumTimePartitions() const { return kNumTimePartitions; }
  const std::vector<int32_t>& GetCompletedTimePartitions() const { return completed_time_partitions_; }
  void ClearCompletedTimePartitions() { completed_time_partitions_.clear(); }
 private:
  bool HasRecvedAllTimePartitionedDistArrays(int32_t time_partition_to_exec) const;
  bool HasSentAllPrefetches() const;
  bool HasRecvedAllPrefetches() const;
  int32_t GetNextToExecTimePartitionId();
};

ExecForLoopSpaceTimeUnordered::ExecForLoopSpaceTimeUnordered(
    int32_t executor_id,
    size_t num_executors,
    int32_t iteration_space_id,
    const int32_t *space_partitioned_dist_array_ids,
    size_t num_space_partitioned_dist_arrays,
    const int32_t *time_partitioned_dist_array_ids,
    size_t num_time_partitioned_dist_arrays,
    const int32_t *global_indexed_dist_array_ids,
    size_t num_global_indexed_dist_arrays,
    const char* exec_loop_func_name,
    std::unordered_map<int32_t, DistArray> &dist_arrays):
    clock_(0),
    kExecutorId(executor_id),
    kNumExecutors(num_executors),
    kExecLoopFuncName(exec_loop_func_name) {
  auto iter = dist_arrays.find(iteration_space_id);
  CHECK(iter != dist_arrays.end());
  iteration_space_ = &(iter->second);
  {
    auto &meta = iteration_space_->GetMeta();
    auto &max_ids = meta.GetMaxPartitionIds();
    CHECK(max_ids.size() == 2) << "max_ids.size() = " << max_ids.size();
    kNumSpacePartitions = max_ids[0] + 1;
    kNumTimePartitions = max_ids[1] + 1;
  }

  for (size_t i = 0; i < num_space_partitioned_dist_arrays; i++) {
    int32_t id = space_partitioned_dist_array_ids[i];
    auto iter = dist_arrays.find(id);
    CHECK(iter != dist_arrays.end());
    space_partitioned_dist_arrays_.emplace(std::make_pair(id, &(iter->second)));
  }

  for (size_t i = 0; i < num_time_partitioned_dist_arrays; i++) {
    int32_t id = time_partitioned_dist_array_ids[i];
    auto iter = dist_arrays.find(id);
    CHECK(iter != dist_arrays.end());
    time_partitioned_dist_arrays_.emplace(std::make_pair(id, &(iter->second)));
  }

  for (size_t i = 0; i < num_global_indexed_dist_arrays; i++) {
    int32_t id = global_indexed_dist_array_ids[i];
    iter = dist_arrays.find(id);
    CHECK(iter != dist_arrays.end());
    auto *dist_array_ptr = &(iter->second);
    global_indexed_dist_arrays_.emplace(std::make_pair(id, dist_array_ptr));
    dist_array_cache_.emplace(
        std::make_pair(id, std::make_pair(PrefetchStatus::kNotPrefetched,
                                          dist_array_ptr->CreatePartition())));
  }

  num_clocks_ = kNumExecutors;
}

ExecForLoopSpaceTimeUnordered::~ExecForLoopSpaceTimeUnordered() {
  for (auto &cache_pair : dist_array_cache_) {
    delete cache_pair.second.second;
  }
}

ExecForLoopSpaceTimeUnordered::RunnableStatus
ExecForLoopSpaceTimeUnordered::GetRunnableStatus(int32_t time_partition_to_exec) const {
  bool has_sent_all_prefetches = HasSentAllPrefetches();
  if (!has_sent_all_prefetches)
    return RunnableStatus::kPrefretchGlobalIndexedDep;

  bool has_recved_all_time_partitioned_dist_arrays
      = HasRecvedAllTimePartitionedDistArrays(time_partition_to_exec);
  bool has_recved_all_prefetches = HasRecvedAllPrefetches();

  if (has_recved_all_time_partitioned_dist_arrays &&
      has_recved_all_prefetches) {
    return RunnableStatus::kRunnable;
  }

  return RunnableStatus::kMissingDep;
}

int32_t
ExecForLoopSpaceTimeUnordered::GetDestExecutorId() const {
  return (kExecutorId + kNumExecutors - 1) % kNumExecutors;
}

// return nullptr if and only if I have completed all local partitions
AbstractDistArrayPartition *
ExecForLoopSpaceTimeUnordered::GetNextPartitionToExec(
    int32_t *space_id,
    int32_t *time_id) {
  int32_t time_partition_to_exec = GetNextToExecTimePartitionId();
  auto &space_time_partition_map = iteration_space_->GetSpaceTimePartitionMap();
  AbstractDistArrayPartition *partition_to_exec = nullptr;
  while (time_partition_to_exec >= 0) {
    while (curr_space_partition_id_ < kNumSpacePartitions) {
      auto space_iter = space_time_partition_map.find(curr_space_partition_id_);
      while (space_iter == space_time_partition_map.end() &&
             curr_space_partition_id_ < kNumSpacePartitions) {
        curr_space_partition_id_ += kNumExecutors;
        space_iter = space_time_partition_map.find(curr_space_partition_id_);
      }
      if (curr_space_partition_id_ >= kNumSpacePartitions) break;

      auto& time_partition_map =  space_iter->second;
      auto partition_iter = time_partition_map.find(time_partition_to_exec);
      if (partition_iter == time_partition_map.end()) {
        continue;
      }
      partition_to_exec = partition_iter->second;
      *space_id = curr_space_partition_id_;
      *time_id = time_partition_to_exec;
      return partition_to_exec;
    }

    curr_time_partition_offset_ += 1;
    if (clock_ < num_clocks_) completed_time_partitions_.emplace_back(time_partition_to_exec);
    time_partition_to_exec = GetNextToExecTimePartitionId();
    curr_space_partition_id_ = kExecutorId;
  }
  return nullptr;
}

bool
ExecForLoopSpaceTimeUnordered::HasRecvedAllTimePartitionedDistArrays(
    int32_t time_partition_to_exec) const {
  for (auto &dist_array_pair : time_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    auto* partition = dist_array->GetLocalPartition(time_partition_to_exec);
    if (partition == nullptr) return false;
  }
  return true;
}

bool
ExecForLoopSpaceTimeUnordered::HasSentAllPrefetches() const {
  for (auto &cache_pair : dist_array_cache_) {
    if (cache_pair.second.first != PrefetchStatus::kPrefetchSent
        && cache_pair.second.first != PrefetchStatus::kPrefetchRecved)
      return false;
  }
  return true;
}

bool
ExecForLoopSpaceTimeUnordered::HasRecvedAllPrefetches() const {
  for (auto &cache_pair : dist_array_cache_) {
    if (cache_pair.second.first != PrefetchStatus::kPrefetchRecved) return false;
  }
  return true;
}

void
ExecForLoopSpaceTimeUnordered::PrepareToExecCurrentTile(int32_t time_partition_to_exec) {
  for (auto& dist_array_pair : space_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    dist_array->SetAccessPartition(curr_space_partition_id_);
  }

  for (auto& dist_array_pair : time_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    dist_array->SetAccessPartition(time_partition_to_exec);
  }

  for (auto& dist_array_pair : global_indexed_dist_arrays_) {
    auto *dist_array = dist_array_pair.second;
    auto dist_array_id = dist_array_pair.first;
    auto *cache_partition = dist_array_cache_.at(dist_array_id).second;
    dist_array->SetAccessPartition(cache_partition);
  }
  curr_space_partition_id_ += kNumExecutors;
}

void
ExecForLoopSpaceTimeUnordered::GetDistArrayTimePartitionsToSend(std::unordered_map<
                                                       int32_t,
                                                       AbstractDistArrayPartition*>*
                                                       partitions) {
  partitions->clear();
  for (auto &dist_array_pair : time_partitioned_dist_arrays_) {
    int32_t dist_array_id = dist_array_pair.first;
    auto* dist_array = dist_array_pair.second;
    (*partitions)[dist_array_id] = dist_array->GetAccessPartition();
  }
}

void
ExecForLoopSpaceTimeUnordered::GetDistArrayGlobalIndexedPartitionsToSend(std::unordered_map<
                                                                int32_t,
                                                                AbstractDistArrayPartition*>*
                                                                partitions) {
  LOG(FATAL) << "not yet supported";
}

int32_t
ExecForLoopSpaceTimeUnordered::GetNextToExecTimePartitionId() {
  int32_t time_partition_id = (kExecutorId + clock_) % kNumExecutors + curr_time_partition_offset_ * kNumExecutors;
  while (time_partition_id >= kNumTimePartitions) {
    if (kExecutorId >= kNumTimePartitions
        && clock_ < kNumExecutors - kExecutorId) {
      // fast forward to skip nonexistant partitions
      clock_ = kNumExecutors - kExecutorId;
    } else {
      clock_ += 1;
    }
    if (clock_ > num_clocks_) return -1;
    curr_time_partition_offset_ = 0;
    time_partition_id = (kExecutorId + clock_) % kNumExecutors + curr_time_partition_offset_ * kNumExecutors;
  }
  return time_partition_id;
}

}

}
