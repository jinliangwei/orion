#include <set>
#include <orion/bosen/server_exec_for_loop.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>

namespace orion {
namespace bosen {

ServerExecForLoop::ServerExecForLoop(
    int32_t server_id,
    size_t num_executors,
    std::unordered_map<int32_t, DistArray> *dist_arrays,
    std::unordered_map<int32_t, DistArray> *dist_array_buffers,
    const std::unordered_map<int32_t, DistArrayBufferInfo> &dist_array_buffer_info_map,
    const int32_t *global_indexed_dist_array_ids,
    size_t num_global_indexed_dist_arrays,
    const int32_t *dist_array_buffer_ids,
    size_t num_dist_array_buffers):
    kServerId(server_id),
    kNumExecutors(num_executors),
    dist_arrays_(dist_arrays),
    dist_array_buffers_(dist_array_buffers),
    dist_array_buffer_info_map_(dist_array_buffer_info_map) {
  LOG(INFO) << __func__ << " NumExecutors = " << kNumExecutors;
  std::set<int32_t> global_indexed_dist_array_id_set;
  for (size_t i = 0; i < num_global_indexed_dist_arrays; i++) {
    int32_t dist_array_id = global_indexed_dist_array_ids[i];
    global_indexed_dist_array_id_set.emplace(dist_array_id);
  }

  std::set<int32_t> buffer_updated_dist_array_ids;

  for (size_t i = 0; i < num_dist_array_buffers; i++) {
    int32_t dist_array_buffer_id = dist_array_buffer_ids[i];
    auto dist_array_buffer_info_iter = dist_array_buffer_info_map_.find(dist_array_buffer_id);
    if (dist_array_buffer_info_iter == dist_array_buffer_info_map_.end()) continue;

    const auto &dist_array_buffer_info = dist_array_buffer_info_iter->second;
    auto dist_array_id = dist_array_buffer_info.kDistArrayId;
    buffer_updated_dist_array_ids.emplace(dist_array_id);
  }

  for (size_t i = 0; i < num_dist_array_buffers; i++) {
    int32_t dist_array_buffer_id = dist_array_buffer_ids[i];
    auto dist_array_buffer_info_iter = dist_array_buffer_info_map_.find(dist_array_buffer_id);
    if (dist_array_buffer_info_iter == dist_array_buffer_info_map_.end()) continue;

    auto &dist_array_buffer_info = dist_array_buffer_info_iter->second;
    const auto &helper_dist_array_ids = dist_array_buffer_info.kHelperDistArrayIds;

    for (auto helper_dist_array_id : helper_dist_array_ids) {
      if (helper_dist_arrays_.count(helper_dist_array_id) > 0) continue;
      if (global_indexed_dist_array_id_set.count(helper_dist_array_id) > 0) continue;
      if (buffer_updated_dist_array_ids.count(helper_dist_array_id) > 0) continue;
      auto dist_array_iter = dist_arrays->find(helper_dist_array_id);
      CHECK(dist_array_iter != dist_arrays->end());
      auto &dist_array = dist_array_iter->second;
      helper_dist_arrays_.emplace(helper_dist_array_id, &dist_array);
    }
  }
}

ServerExecForLoop::~ServerExecForLoop() { }

void
ServerExecForLoop::PrepareHelperDistArrays() {
  for (auto &dist_array_pair : helper_dist_arrays_) {
    auto *dist_array_ptr = dist_array_pair.second;
    auto *dist_array_partition = dist_array_ptr->GetLocalPartition(kServerId);
    dist_array_partition->CreateAccessor();
  }
}

void
ServerExecForLoop::RetrieveHelperDistArrays() {
  for (auto &dist_array_pair : helper_dist_arrays_) {
    auto *dist_array_ptr = dist_array_pair.second;
    auto *dist_array_partition = dist_array_ptr->GetLocalPartition(kServerId);
    dist_array_partition->ClearAccessor();
  }
}

void
ServerExecForLoop::DeserializeAndApplyDistArrayCaches(
    uint8_t* bytes) {
  const auto *cursor = bytes;
  size_t num_dist_arrays = *reinterpret_cast<const size_t*>(cursor);
  LOG(INFO) << __func__ << " num_dist_arrays = " << num_dist_arrays;
  cursor += sizeof(size_t);
  for (size_t i = 0; i < num_dist_arrays; i++) {
    auto dist_array_id = *reinterpret_cast<const int32_t*>(cursor);
    cursor += sizeof(int32_t);
    LOG(INFO) << "deserialize and apply dist array cache "
              << dist_array_id;
    LOG(INFO) << "dist_arrays = " << (void*) dist_arrays_;
    auto iter = dist_arrays_->find(dist_array_id);
    CHECK(iter != dist_arrays_->end());
    LOG(INFO) << "found dist array";
    auto &dist_array = iter->second;
    auto *dist_array_partition = dist_array.GetLocalPartition(kServerId);
    LOG(INFO) << "local partition  = " << (void*) dist_array_partition;

    cursor = dist_array_partition->DeserializeAndOverwrite(cursor);
  }
  LOG(INFO) << __func__ << " done!";
  delete[] bytes;
}

void
ServerExecForLoop::DeserializeAndApplyDistArrayBuffers(
    uint8_t* bytes) {
  LOG(INFO) << __func__;
  const auto *cursor = bytes;
  size_t num_dist_arrays = *reinterpret_cast<const size_t*>(cursor);
  cursor += sizeof(size_t);
  std::unordered_map<int32_t, AbstractDistArrayPartition*> buffer_partition_map;
  for (size_t i = 0; i < num_dist_arrays; i++) {
    auto dist_array_id = *reinterpret_cast<const int32_t*>(cursor);
    cursor += sizeof(int32_t);
    auto iter = dist_array_buffers_->find(dist_array_id);
    CHECK(iter != dist_array_buffers_->end());
    auto &dist_array = iter->second;
    auto *dist_array_partition = dist_array.GetBufferPartition();
    cursor = dist_array_partition->Deserialize(cursor);
    buffer_partition_map.emplace(dist_array_id, dist_array_partition);
  }

  for (auto &buffer_pair : buffer_partition_map) {
    auto dist_array_buffer_id = buffer_pair.first;
    auto info_iter = dist_array_buffer_info_map_.find(dist_array_buffer_id);
    if (info_iter == dist_array_buffer_info_map_.end()) continue;
    auto* updates_buffer_partition = buffer_pair.second;
    auto &buffer_info = info_iter->second;
    auto dist_array_id = buffer_info.kDistArrayId;
    const auto &apply_buffer_func_name = buffer_info.kApplyBufferFuncName;
    const auto &helper_dist_array_ids = buffer_info.kHelperDistArrayIds;
    const auto &helper_dist_array_buffer_ids = buffer_info.kHelperDistArrayBufferIds;
    auto dist_array_iter = dist_arrays_->find(dist_array_id);
    CHECK(dist_array_iter != dist_arrays_->end());
    auto &dist_array_to_update = dist_array_iter->second;
    auto *partition_to_update = dist_array_to_update.GetLocalPartition(kServerId);

    std::vector<AbstractDistArrayPartition*> my_helper_dist_arrays;
    std::vector<const AbstractDistArrayPartition*> helper_dist_array_buffers;
    for (auto helper_dist_array_id : helper_dist_array_ids) {
      if (helper_dist_arrays_.count(helper_dist_array_id) > 0) continue;
      auto helper_dist_array_iter = dist_arrays_->find(helper_dist_array_id);
      auto &helper_dist_array = helper_dist_array_iter->second;
      auto *helper_partition = helper_dist_array.GetLocalPartition(kServerId);
      helper_partition->BuildKeyValueBuffersFromSparseIndex();
      my_helper_dist_arrays.push_back(helper_partition);
    }

    for (auto buffer_id : helper_dist_array_buffer_ids) {
      auto helper_dist_array_buffer_iter = buffer_partition_map.find(buffer_id);
      auto *buffer_partition = helper_dist_array_buffer_iter->second;
      helper_dist_array_buffers.push_back(buffer_partition);
    }

    partition_to_update->ApplyBufferedUpdates(updates_buffer_partition,
                                              helper_dist_array_buffers,
                                              apply_buffer_func_name);

    for (auto *helper_dist_array_partition : my_helper_dist_arrays) {
      helper_dist_array_partition->ClearAccessor();
      helper_dist_array_partition->CheckAndBuildIndex();
    }
  }

  for (auto &buffer_partition_pair : buffer_partition_map) {
    auto *buffer_partition = buffer_partition_pair.second;
    buffer_partition->Clear();
  }

  delete[] bytes;
}

bool
ServerExecForLoop::NotifyExecForLoopDone() {
  completed_executors_++;
  LOG(INFO) << __func__ << " completed_executors = "
            << completed_executors_ << " numExecutors = " << kNumExecutors
            << " bool = " << (completed_executors_ == kNumExecutors);
  return (completed_executors_ == kNumExecutors);
}

}
}
