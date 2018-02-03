#include <orion/bosen/abstract_exec_for_loop.hpp>
#include <vector>
#include <memory>
namespace orion {
namespace bosen {

AbstractExecForLoop::AbstractExecForLoop(
    int32_t executor_id,
    size_t num_executors,
    size_t num_servers,
    int32_t iteration_space_id,
    const int32_t *space_partitioned_dist_array_ids,
    size_t num_space_partitioned_dist_arrays,
    const int32_t *time_partitioned_dist_array_ids,
    size_t num_time_partitioned_dist_arrays,
    const int32_t *global_indexed_dist_array_ids,
    size_t num_global_indexed_dist_arrays,
    const int32_t *buffered_dist_array_ids,
    size_t num_buffered_dist_arrays,
    const int32_t *dist_array_buffer_ids,
    const size_t *num_buffers_each_dist_array,
    const char* loop_batch_func_name,
    const char *prefetch_batch_func_name,
    std::unordered_map<int32_t, DistArray> *dist_arrays,
    std::unordered_map<int32_t, DistArray> *dist_array_buffers):
    kExecutorId(executor_id),
    kNumExecutors(num_executors),
    kNumServers(num_servers),
    kLoopBatchFuncName(loop_batch_func_name),
    kPrefetchBatchFuncName(prefetch_batch_func_name) {

  LOG(INFO) << __func__ << " num_buffered_dist_arrays = " << num_buffered_dist_arrays;

  auto iter = dist_arrays->find(iteration_space_id);
  CHECK(iter != dist_arrays->end());
  iteration_space_ = &(iter->second);

  for (size_t i = 0; i < num_space_partitioned_dist_arrays; i++) {
    int32_t id = space_partitioned_dist_array_ids[i];
    auto iter = dist_arrays->find(id);
    CHECK(iter != dist_arrays->end());
    space_partitioned_dist_arrays_.emplace(std::make_pair(id, &(iter->second)));
  }

  for (size_t i = 0; i < num_time_partitioned_dist_arrays; i ++) {
    int32_t id = time_partitioned_dist_array_ids[i];
    auto iter = dist_arrays->find(id);
    CHECK(iter != dist_arrays->end());
    time_partitioned_dist_arrays_.emplace(std::make_pair(id, &(iter->second)));
  }

  for (size_t i = 0; i < num_global_indexed_dist_arrays; i++) {
    int32_t id = global_indexed_dist_array_ids[i];
    iter = dist_arrays->find(id);
    CHECK(iter != dist_arrays->end());
    auto *dist_array_ptr = &(iter->second);
    global_indexed_dist_arrays_.emplace(std::make_pair(id, dist_array_ptr));
    dist_array_cache_.emplace(id,
                              std::unique_ptr<AbstractDistArrayPartition>(dist_array_ptr->CreatePartition()));
  }

  size_t dist_array_buffer_index = 0;
  for (size_t i = 0; i < num_buffered_dist_arrays; i++) {
    int32_t dist_array_id = buffered_dist_array_ids[i];
    LOG(INFO) << "dist_array_id = " << dist_array_id;
    size_t num_buffers = num_buffers_each_dist_array[i];
    LOG(INFO) << "num_buffers = " << num_buffers;
    auto iter_pair = dist_array_to_buffers_map_.emplace(dist_array_id, std::vector<int32_t>());
    auto buff_vec_iter = iter_pair.first;
    auto buff_vec = buff_vec_iter->second;
    for (size_t j = 0; j < num_buffers; j++) {
      int32_t dist_array_buffer_id = dist_array_buffer_ids[dist_array_buffer_index];
      buff_vec.push_back(dist_array_buffer_id);
      auto iter = dist_array_buffers->find(dist_array_buffer_id);
      auto *dist_array_buffer = &(iter->second);
      auto buff_iter = dist_array_buffers_.find(dist_array_buffer_id);
      if (buff_iter == dist_array_buffers_.end()) {
        LOG(INFO) << "emplace " << dist_array_buffer_id;
        dist_array_buffers_.emplace(dist_array_buffer_id, dist_array_buffer);
      }
      dist_array_buffer_index++;
    }
  }
}

AbstractExecForLoop::~AbstractExecForLoop() { }

void
AbstractExecForLoop::SentAllPrefetchRequests() {
  prefetch_status_ = PrefetchStatus::kPrefetchSent;
}

bool
AbstractExecForLoop::HasSentAllPrefetchRequests() const {
  return (prefetch_status_ != PrefetchStatus::kNotPrefetched);
}

bool
AbstractExecForLoop::HasRecvedAllPrefetches() const {
  return (prefetch_status_ == PrefetchStatus::kPrefetchRecved);
}

bool
AbstractExecForLoop::HasRecvedAllTimePartitionedDistArrays(
    int32_t time_partition_id) const {
  for (auto &dist_array_pair : time_partitioned_dist_arrays_) {
    auto* dist_array = dist_array_pair.second;
    auto* partition = dist_array->GetLocalPartition(time_partition_id);
    if (partition == nullptr) return false;
  }
  return true;
}

void
AbstractExecForLoop::ComputePrefetchIndinces() {
  PrepareToExecCurrPartition();
  std::vector<int32_t> dist_array_ids_vec;

  for (const auto &dist_array_pair : global_indexed_dist_arrays_) {
    dist_array_ids_vec.push_back(dist_array_pair.first);
  }

  curr_partition_->ComputePrefetchIndinces(
      kPrefetchBatchFuncName,
      dist_array_ids_vec,
      global_indexed_dist_arrays_,
      &point_prefetch_dist_array_map_);
}

void
AbstractExecForLoop::ExecuteForLoopPartition() {
  //LOG(INFO) << __func__;
  PrepareToExecCurrPartition();
  curr_partition_->Execute(kLoopBatchFuncName);
  ClearCurrPartition();
}

void
AbstractExecForLoop::SerializeAndClearPrefetchIds(ExecutorSendBufferMap *send_buffer_map) {
  LOG(INFO) << __func__;
  // server id -> (dist_array_id -> keys)
  std::unordered_map<int32_t, PointQueryKeyDistArrayMap> server_point_key_map;

  for (const auto &dist_array_key_pair : point_prefetch_dist_array_map_) {
    int32_t dist_array_id = dist_array_key_pair.first;
    const auto &point_key_vec = dist_array_key_pair.second;
    for (auto key : point_key_vec) {
      int32_t server_id = key % kNumServers;
      server_point_key_map[server_id][dist_array_id].push_back(key);
    }
  }

  num_pending_prefetch_requests_ = server_point_key_map.size();
  for (const auto &server_point_key_pair : server_point_key_map) {
    int32_t server_id = server_point_key_pair.first;
    const auto &dist_array_point_key_map = server_point_key_pair.second;
    size_t server_num_bytes = sizeof(size_t); // num_dist_arrays
    for (const auto &dist_array_point_key_pair : dist_array_point_key_map) {
      const auto &point_key_vec = dist_array_point_key_pair.second;
      // dist_array_id, num_keys, key_vec
      server_num_bytes += sizeof(int32_t) + sizeof(size_t)
                          + point_key_vec.size() * sizeof(int64_t);
    }
    uint8_t *server_buff = new uint8_t[server_num_bytes];
    (*send_buffer_map)[server_id] = std::make_pair(server_buff, server_num_bytes);
    uint8_t *cursor = server_buff;
    // num_dist_arrays
    *reinterpret_cast<size_t*>(cursor) = dist_array_point_key_map.size();
    cursor += sizeof(size_t);

    for (const auto &dist_array_point_key_pair : dist_array_point_key_map) {
      int32_t dist_array_id = dist_array_point_key_pair.first;
      const auto &point_key_vec = dist_array_point_key_pair.second;
      *reinterpret_cast<int32_t*>(cursor) = dist_array_id;
      cursor += sizeof(int32_t);
      *reinterpret_cast<size_t*>(cursor) = point_key_vec.size();
      cursor += sizeof(size_t);
      memcpy(cursor, point_key_vec.data(), point_key_vec.size() * sizeof(int64_t));
      cursor += sizeof(int64_t) * point_key_vec.size();
    }
  }
  point_prefetch_dist_array_map_.clear();
}

void
AbstractExecForLoop::SerializeAndClearGlobalPartitionedDistArrays() {

}

void
AbstractExecForLoop::SerializeAndClearPipelinedTimePartitions() {
  time_partitions_serialized_bytes_ = nullptr;
  time_partitions_serialized_size_ = 0;
  int32_t time_partition_id_to_send = GetTimePartitionIdToSend();
  if (time_partition_id_to_send < 0) return;
  std::unordered_map<int32_t, SendDataBuffer> data_buffers;
  for (auto &dist_array_pair : time_partitioned_dist_arrays_) {
    int32_t dist_array_id = dist_array_pair.first;

    auto *dist_array = dist_array_pair.second;
    auto *dist_array_partition = dist_array->GetLocalPartition(
        time_partition_id_to_send);
    if (dist_array_partition == nullptr) continue;
    dist_array_partition->BuildKeyValueBuffersFromSparseIndex();
    auto data_buff = dist_array_partition->Serialize();
    if (data_buff.second > 0) {
      CHECK(data_buff.first != nullptr);
      data_buffers.emplace(dist_array_id, data_buff);
    }
    dist_array->DeletePartition(time_partition_id_to_send);
    dist_array->GcPartitions();
  }

  if (data_buffers.empty()) return;
  size_t total_size = sizeof(size_t) + sizeof(int32_t);
  for (auto &data_buff : data_buffers) {
    total_size += sizeof(int32_t) + data_buff.second.second;
  }
  auto* buffer_bytes = new uint8_t[total_size];
  uint8_t* cursor = buffer_bytes;
  *reinterpret_cast<size_t*>(cursor) = data_buffers.size();
  cursor += sizeof(size_t);
  *reinterpret_cast<int32_t*>(cursor) = time_partition_id_to_send;
  cursor += sizeof(int32_t);
  for (auto &data_buff_pair : data_buffers) {
    int32_t dist_array_id = data_buff_pair.first;
    auto &data_buff = data_buff_pair.second;
    *reinterpret_cast<int32_t*>(cursor) = dist_array_id;
    cursor += sizeof(int32_t);
    memcpy(cursor, data_buff.first, data_buff.second);
    cursor += data_buff.second;
    delete[] data_buff.first;
  }
  time_partitions_serialized_bytes_ = buffer_bytes;
  time_partitions_serialized_size_ = total_size;
}

void
AbstractExecForLoop::DeserializePipelinedTimePartitions(const uint8_t* bytes) {
  const uint8_t *cursor = bytes;
  size_t num_data_buffers = *reinterpret_cast<const size_t*>(cursor);
  cursor += sizeof(size_t);
  int32_t time_partition_id = *reinterpret_cast<const int32_t*>(cursor);
  cursor += sizeof(int32_t);

  for (size_t i = 0; i < num_data_buffers; i++) {
    int32_t dist_array_id = *reinterpret_cast<const int32_t*>(cursor);
    cursor += sizeof(int32_t);
    auto iter = time_partitioned_dist_arrays_.find(dist_array_id);
    CHECK(iter != time_partitioned_dist_arrays_.end());
    auto *dist_array = iter->second;
    auto create_pair = dist_array->GetAndCreateLocalPartition(time_partition_id);
    CHECK(create_pair.second);
    auto *partition = create_pair.first;
    cursor = partition->Deserialize(cursor);
    partition->BuildIndex();
  }
}

void
AbstractExecForLoop::DeserializePipelinedTimePartitionsBuffVec(
    PeerRecvPipelinedTimePartitionsBuffer** buff_vec,
    size_t num_buffs) {
  for (size_t i = 0; i < num_buffs; i++) {
    auto *buff = buff_vec[i];
    auto &byte_buff = buff->byte_buff;
    uint64_t pred_notice = buff->pred_notice;
    ApplyPredecessorNotice(pred_notice);
    DeserializePipelinedTimePartitions(byte_buff.GetBytes());
    delete buff;
  }
  delete[] buff_vec;
}

SendDataBuffer
AbstractExecForLoop::GetAndResetSerializedTimePartitions() {
  auto send_data_buffer = std::make_pair(time_partitions_serialized_bytes_,
                                          time_partitions_serialized_size_);
  time_partitions_serialized_bytes_ = nullptr;
  time_partitions_serialized_size_ = 0;
  return send_data_buffer;
}

void
AbstractExecForLoop::CachePrefetchDistArrayValues(
    PeerRecvGlobalIndexedDistArrayDataBuffer **buff_vec,
    size_t num_buffs) {
  LOG(INFO) << __func__;
  CHECK(num_pending_prefetch_requests_ > 0);
  for (size_t i = 0; i < num_buffs; i++) {
    auto *buff = buff_vec[i];
    LOG(INFO) << "i = " << i << " " << (void*) buff;
    const auto *bytes = buff->byte_buff.data();
    LOG(INFO) << "bytes->size() = " << buff->byte_buff.size();
    const auto *cursor = bytes;
    size_t num_dist_arrays = *reinterpret_cast<const size_t*>(cursor);
    cursor += sizeof(size_t);
    for (size_t j = 0; j < num_dist_arrays; j++) {
      int32_t dist_array_id = *reinterpret_cast<const int32_t*>(cursor);
      cursor += sizeof(int32_t);
      auto iter = dist_array_cache_.find(dist_array_id);
      CHECK(iter != dist_array_cache_.end()) << " dist_array_id = " << dist_array_id;
      auto *cache_partition = iter->second.get();
      LOG(INFO) << "cursor offset = " << cursor - bytes;
      cursor = cache_partition->DeserializeAndAppend(cursor);
    }
    delete buff;
  }
  delete[] buff_vec;
  num_pending_prefetch_requests_ -= 1;
  if (num_pending_prefetch_requests_ == 0) {
    prefetch_status_ = PrefetchStatus::kPrefetchRecved;
    for (auto& dist_array_pair : global_indexed_dist_arrays_) {
      auto dist_array_id = dist_array_pair.first;
      auto *cache_partition = dist_array_cache_.at(dist_array_id).get();
      cache_partition->CreateCacheAccessor();
    }
  }
}

}
}
