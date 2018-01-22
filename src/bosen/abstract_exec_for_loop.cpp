#include <orion/bosen/abstract_exec_for_loop.hpp>
#include <vector>
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
    dist_array_cache_.emplace(
        std::make_pair(id, std::make_pair(PrefetchStatus::kNotPrefetched,
                                          dist_array_ptr->CreatePartition())));
  }

  size_t dist_array_buffer_index = 0;
  for (size_t i = 0; i < num_buffered_dist_arrays; i++) {
    int32_t dist_array_id = buffered_dist_array_ids[i];
    size_t num_buffers = num_buffers_each_dist_array[i];
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
        dist_array_buffers_.emplace(dist_array_buffer_id, dist_array_buffer);
      }
      dist_array_buffer_index++;
    }
  }
}

void
AbstractExecForLoop::SentAllPrefetchRequests() {
  for (auto &cache_pair : dist_array_cache_) {
    cache_pair.second.first = PrefetchStatus::kPrefetchSent;
  }
}

bool
AbstractExecForLoop::HasSentAllPrefetchRequests() const {
  for (auto &cache_pair : dist_array_cache_) {
    if (cache_pair.second.first != PrefetchStatus::kNotPrefetched)
      return false;
  }
  return true;
}

bool
AbstractExecForLoop::HasRecvedAllPrefetches() const {
  for (auto &cache_pair : dist_array_cache_) {
    if (cache_pair.second.first != PrefetchStatus::kPrefetchRecved) return false;
  }
  return true;
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
      &point_prefetch_dist_array_map_,
      &range_prefetch_dist_array_map_);
}

void
AbstractExecForLoop::ExecuteForLoopPartition() {
  //LOG(INFO) << __func__;
  PrepareToExecCurrPartition();
  curr_partition_->Execute(kLoopBatchFuncName);
  ClearCurrPartition();
}

void
AbstractExecForLoop::SerializeAndClearPrefetchIds() {
  send_buffer_map_.clear();
  std::unordered_map<int32_t, PointQueryKeyDistArrayMap> server_point_key_map;
  std::unordered_map<int32_t, RangeQueryKeyDistArrayMap> server_range_key_map;

  for (const auto &dist_array_key_pair : point_prefetch_dist_array_map_) {
    int32_t dist_array_id = dist_array_key_pair.first;
    const auto &point_key_vec = dist_array_key_pair.second;
    for (auto key : point_key_vec) {
      int32_t server_id = key % kNumServers;
      server_point_key_map[server_id][dist_array_id].push_back(key);
    }
  }

  for (const auto &dist_array_key_pair : range_prefetch_dist_array_map_) {
    int32_t dist_array_id = dist_array_key_pair.first;
    const auto &range_key_vec = dist_array_key_pair.second;
    for (const auto &range_key_pair : range_key_vec) {
      int64_t key = range_key_pair.first;
      size_t num_keys = range_key_pair.second;
      if (num_keys <= kNumServers) {
        for (size_t i = 0; i < num_keys; i++) {
          int64_t key_i = key + i;
          int32_t server_id = key_i % kNumServers;
          server_point_key_map[server_id][dist_array_id].push_back(key);
        }
      } else {
        for (size_t i = 0; i < kNumServers; i++) {
          server_range_key_map[i][dist_array_id].emplace_back(key, num_keys);
        }
      }
    }
  }

  std::unordered_map<int32_t, size_t> server_num_bytes;
  for (const auto &server_point_key_pair : server_point_key_map) {
    int32_t server_id = server_point_key_pair.first;
    const auto &dist_array_point_key_map = server_point_key_pair.second;
    server_num_bytes[server_id] += sizeof(QueryDelimiter) + sizeof(size_t); // num_dist_arrays
    for (const auto &dist_array_point_key_pair : dist_array_point_key_map) {
      const auto &point_key_vec = dist_array_point_key_pair.second;
      // dist_array_id, num_keys, key_vec
      server_num_bytes[server_id] += sizeof(int32_t) + sizeof(size_t) + point_key_vec.size() * sizeof(int64_t);
    }
  }

  for (const auto &server_range_key_pair : server_range_key_map) {
    int32_t server_id = server_range_key_pair.first;
    const auto &dist_array_range_key_map = server_range_key_pair.second;
    server_num_bytes[server_id] += sizeof(QueryDelimiter) + sizeof(size_t); // num_dist_arrays
    for (const auto &dist_array_range_key_pair : dist_array_range_key_map) {
      const auto &range_key_vec = dist_array_range_key_pair.second;
      // dist_array_id, num_keys, key_vec
      server_num_bytes[server_id] += sizeof(int32_t) + sizeof(size_t)
                                     + range_key_vec.size() * (sizeof(int64_t) + sizeof(size_t));
    }
  }

  for (const auto &server_num_bytes_pair : server_num_bytes) {
    int32_t server_id = server_num_bytes_pair.first;
    size_t num_bytes = server_num_bytes_pair.second;
    uint8_t *server_buff = new uint8_t[num_bytes];
    send_buffer_map_[server_id] = std::make_pair(server_buff, num_bytes);
    uint8_t *cursor = server_buff;

    auto point_key_iter = server_point_key_map.find(server_id);
    if (point_key_iter != server_point_key_map.end()) {
      *reinterpret_cast<QueryDelimiter*>(cursor) = QueryDelimiter::kPointQueryStart;
      cursor += sizeof(QueryDelimiter);
      const auto &dist_array_point_key_map = point_key_iter->second;
      *reinterpret_cast<size_t*>(cursor) = dist_array_point_key_map.size();
      cursor += sizeof(size_t);

      for (const auto &dist_array_point_key_pair : dist_array_point_key_map) {
        int32_t dist_array_id = dist_array_point_key_pair.first;
        const auto &point_key_vec = dist_array_point_key_pair.second;
        *reinterpret_cast<int32_t*>(cursor) = dist_array_id;
        cursor += sizeof(int32_t);
        *reinterpret_cast<size_t*>(cursor) = point_key_vec.size();
        cursor += sizeof(size_t);
        for (auto &key : point_key_vec) {
          *reinterpret_cast<int64_t*>(cursor) = key;
          cursor += sizeof(int64_t);
        }
      }
    }

    auto range_key_iter = server_range_key_map.find(server_id);
    if (range_key_iter != server_range_key_map.end()) {
      *reinterpret_cast<QueryDelimiter*>(cursor) = QueryDelimiter::kRangeQueryStart;
      cursor += sizeof(QueryDelimiter);
      const auto &dist_array_range_key_map = range_key_iter->second;
      *reinterpret_cast<size_t*>(cursor) = dist_array_range_key_map.size();
      cursor += sizeof(size_t);

      for (const auto &dist_array_range_key_pair : dist_array_range_key_map) {
        int32_t dist_array_id = dist_array_range_key_pair.first;
        const auto &range_key_vec = dist_array_range_key_pair.second;
        *reinterpret_cast<int32_t*>(cursor) = dist_array_id;
        cursor += sizeof(int32_t);
        *reinterpret_cast<size_t*>(cursor) = range_key_vec.size();
        cursor += sizeof(size_t);
        for (auto &key_pair : range_key_vec) {
          int64_t key = key_pair.first;
          size_t num_keys = key_pair.second;
          *reinterpret_cast<int64_t*>(cursor) = key;
          cursor += sizeof(int64_t);
          *reinterpret_cast<size_t*>(cursor) = num_keys;
          cursor += sizeof(size_t);
        }
      }
    }
  }
  point_prefetch_dist_array_map_.clear();
  range_prefetch_dist_array_map_.clear();
}

void
AbstractExecForLoop::DeserializePrefetchedData(const uint8_t* bytes) {

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
    //LOG(INFO) << __func__ << " dist_array_id = " << dist_array_id
    //<< " time_partition_id = " << time_partition_id_to_send;

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
  auto send_data_buffer =  std::make_pair(time_partitions_serialized_bytes_,
                                          time_partitions_serialized_size_);
  time_partitions_serialized_bytes_ = nullptr;
  time_partitions_serialized_size_ = 0;
  return send_data_buffer;
}

}
}
