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

class ExecForLoopSpaceTime : public AbstractExecForLoop {
 private:
  std::unordered_map<int32_t, DistArray*> space_partitioned_dist_arrays_;
  std::unordered_map<int32_t, DistArray*> time_partitioned_dist_arrays_;
  std::unordered_map<int32_t, DistArray*> global_indexed_dist_arrays_;
  std::unordered_map<int32_t, DistArrayCache> dist_array_cache_;

  DistArray *iteration_space_;
  int32_t clock_;
  const bool kIsOrdered;
  const int32_t kExecutorId;
  const size_t kNumExecutors;
  int32_t kNumSpacePartitions {0};
  int32_t kNumTimePartitions {0};
  int32_t num_clocks_ {0};
  const std::string kExecLoopFuncName;
  int32_t curr_space_partition_id_ {0};

 public:
  enum class RunnableStatus {
    kRunnable = 0,
    kMissingDep = 1,
    kPrefretchGlobalIndexedDep = 2,
    kSkip = 3,
    kCompleted = 4
  };

  ExecForLoopSpaceTime(
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
      bool is_ordered,
      std::unordered_map<int32_t, DistArray> &dist_arrays);
  virtual ~ExecForLoopSpaceTime();

  RunnableStatus GetRunnableStatus() const;
  bool GetTimePartitionIdToSend(int32_t *time_partition_id,
                                int32_t *dest_executor_id) const;
  AbstractDistArrayPartition* PrepareExecCurrentTile();

  void IncClock() {
    clock_++;
    curr_space_partition_id_ = 0;
  }

  const std::string &GetExecLoopFuncName() { return kExecLoopFuncName; }

  void
  GetDistArrayTimePartitionsToSend(std::unordered_map<int32_t, AbstractDistArrayPartition*>*
                                   partitions);

  void
  GetDistArrayGlobalIndexedPartitionsToSend(std::unordered_map<int32_t, AbstractDistArrayPartition*>*
                                            partitions);

 private:
  RunnableStatus GetRunnableStatusUnordered() const;
  RunnableStatus GetRunnableStatusOrdered() const;

  bool GetTimePartitionIdToSendUnordered(int32_t *time_partition_id,
                                         int32_t *dest_executor_id) const;
  bool GetTimePartitionIdToSendOrdered(int32_t *time_partition_id,
                                       int32_t *dest_executor_id) const;

  bool HasRecvedAllTimePartitionedDistArrays(int32_t time_partition_to_exec) const;
  bool HasSentAllPrefetches() const;
  bool HasRecvedAllPrefetches() const;
};

ExecForLoopSpaceTime::ExecForLoopSpaceTime(
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
    bool is_ordered,
    std::unordered_map<int32_t, DistArray> &dist_arrays):
    clock_(0),
    kIsOrdered(is_ordered),
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
    LOG(INFO) << "space partitioned = " << id;
    auto iter = dist_arrays.find(id);
    CHECK(iter != dist_arrays.end());
    space_partitioned_dist_arrays_.emplace(std::make_pair(id, &(iter->second)));
  }

  for (size_t i = 0; i < num_time_partitioned_dist_arrays; i++) {
    int32_t id = time_partitioned_dist_array_ids[i];
    LOG(INFO) << "time partitioned = " << id;
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
  CHECK(!kIsOrdered);

  num_clocks_ = kIsOrdered ? kNumSpacePartitions + kNumTimePartitions
                : std::max(kNumExecutors, (size_t) kNumTimePartitions);
}

ExecForLoopSpaceTime::~ExecForLoopSpaceTime() {
  for (auto &cache_pair : dist_array_cache_) {
    delete cache_pair.second.second;
  }
}

ExecForLoopSpaceTime::RunnableStatus
ExecForLoopSpaceTime::GetRunnableStatus() const {
  return kIsOrdered ? GetRunnableStatusOrdered()
      : GetRunnableStatusUnordered();

}

ExecForLoopSpaceTime::RunnableStatus
ExecForLoopSpaceTime::GetRunnableStatusUnordered() const {
  if (clock_ == num_clocks_) return RunnableStatus::kCompleted;

  int32_t time_partition_to_exec = (clock_ + kExecutorId) % num_clocks_;
  if (time_partition_to_exec >= kNumTimePartitions) {
    return RunnableStatus::kSkip;
  }

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

bool
ExecForLoopSpaceTime::GetTimePartitionIdToSend(
    int32_t *time_partition_id,
    int32_t *dest_executor_id) const {
  if (kIsOrdered) return GetTimePartitionIdToSendOrdered(time_partition_id,
                                                         dest_executor_id);

  return GetTimePartitionIdToSendUnordered(time_partition_id,
                                           dest_executor_id);
}
bool
ExecForLoopSpaceTime::GetTimePartitionIdToSendUnordered(
    int32_t *time_partition_id,
    int32_t *dest_executor_id) const {
  int32_t time_partition_to_exec = (clock_ + kExecutorId) % num_clocks_;
  if (time_partition_to_exec >= kNumTimePartitions) {
    return false;
  }

  *time_partition_id = time_partition_to_exec;
  *dest_executor_id = (kExecutorId + kNumExecutors - 1) % kNumExecutors;
  return true;
}

bool
ExecForLoopSpaceTime::GetTimePartitionIdToSendOrdered(
    int32_t *time_partition_id,
    int32_t *dest_executor_id) const {
  LOG(FATAL) << "not yet supported";
  return false;
}

bool
ExecForLoopSpaceTime::HasRecvedAllTimePartitionedDistArrays(
    int32_t time_partition_to_exec) const {
  for (auto &dist_array_pair : time_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    auto* partition = dist_array->GetLocalPartition(time_partition_to_exec);
    if (partition == nullptr) return false;
  }
  return true;
}

bool
ExecForLoopSpaceTime::HasSentAllPrefetches() const {
  for (auto &cache_pair : dist_array_cache_) {
    if (cache_pair.second.first != PrefetchStatus::kPrefetchSent
        && cache_pair.second.first != PrefetchStatus::kPrefetchRecved)
      return false;
  }
  return true;
}

bool
ExecForLoopSpaceTime::HasRecvedAllPrefetches() const {
  for (auto &cache_pair : dist_array_cache_) {
    if (cache_pair.second.first != PrefetchStatus::kPrefetchRecved) return false;
  }
  return true;
}

ExecForLoopSpaceTime::RunnableStatus
ExecForLoopSpaceTime::GetRunnableStatusOrdered() const {
  LOG(FATAL) << "not yet supported";
  return RunnableStatus::kCompleted;
}

AbstractDistArrayPartition*
ExecForLoopSpaceTime::PrepareExecCurrentTile() {
  auto &space_time_partition_map = iteration_space_->GetSpaceTimePartitionMap();
  AbstractDistArrayPartition* partition_to_exec = nullptr;
  int32_t time_partition_to_exec = (clock_ + kExecutorId) % num_clocks_;
  LOG(INFO) << __func__ << " executor id = " << kExecutorId
            << " time_partition_to_exec = " << time_partition_to_exec;
  for (auto& space_pair : space_time_partition_map) {
    if (space_pair.first >= curr_space_partition_id_) {
      curr_space_partition_id_ = space_pair.first;
      auto& time_partition_map =  space_pair.second;
      auto partition_iter = time_partition_map.find(time_partition_to_exec);
      if (partition_iter == time_partition_map.end()) {
        continue;
      }
      partition_to_exec = partition_iter->second;
      break;
    }
  }
  if (partition_to_exec == nullptr) return nullptr;
  LOG(INFO) << "execute partition " << curr_space_partition_id_ << " " << time_partition_to_exec;
  for (auto& dist_array_pair : space_partitioned_dist_arrays_) {
    LOG(INFO) << "SetAccessPartition for space_partitioned dist_array "
              << dist_array_pair.first << " for space partition "
              << curr_space_partition_id_;
    auto* dist_array = dist_array_pair.second;
    dist_array->SetAccessPartition(curr_space_partition_id_);
  }

  for (auto& dist_array_pair : time_partitioned_dist_arrays_) {
    LOG(INFO) << "SetAccessPartition for time_partitioned dist_array "
              << dist_array_pair.first << " for time partition "
              << time_partition_to_exec;
    auto* dist_array = dist_array_pair.second;
    dist_array->SetAccessPartition(time_partition_to_exec);
  }

  for (auto& dist_array_pair : global_indexed_dist_arrays_) {
    auto *dist_array = dist_array_pair.second;
    auto dist_array_id = dist_array_pair.first;
    auto *cache_partition = dist_array_cache_.at(dist_array_id).second;
    dist_array->SetAccessPartition(cache_partition);
  }
  curr_space_partition_id_ += 1;
  return partition_to_exec;
}

void
ExecForLoopSpaceTime::GetDistArrayTimePartitionsToSend(std::unordered_map<
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
ExecForLoopSpaceTime::GetDistArrayGlobalIndexedPartitionsToSend(std::unordered_map<
                                                                int32_t,
                                                                AbstractDistArrayPartition*>*
                                                                partitions) {
  LOG(FATAL) << "not yet supported";
}

}
}
