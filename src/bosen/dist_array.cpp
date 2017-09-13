#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_partition.hpp>
#include <orion/bosen/julia_evaluator.hpp>

namespace orion {
namespace bosen {

DistArray::DistArray(
    const Config &config,
    type::PrimitiveType value_type,
    int32_t executor_id):
    kConfig(config),
    kValueType(value_type),
    kValueSize(type::SizeOf(value_type)),
    kExecutorId(executor_id) { }

DistArray::~DistArray() {
  for (auto &partition_pair : partitions_) {
    delete partition_pair.second;
  }
}

DistArray::DistArray(DistArray &&other):
    kConfig(other.kConfig),
    kValueType(other.kValueType),
    kValueSize(other.kValueSize),
    kExecutorId(other.kExecutorId),
    partitions_(other.partitions_) {
  other.partitions_.clear();
}

AbstractDistArrayPartition*
DistArray::CreatePartition() const {
  AbstractDistArrayPartition *partition_ptr = nullptr;
  switch(kValueType) {
    case type::PrimitiveType::kVoid:
      {
        LOG(FATAL) << "DistArray value type cannot be void";
        break;
      }
    case type::PrimitiveType::kInt8:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int8_t>(kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kUInt8:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint8_t>(kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kInt16:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int16_t>(kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kUInt16:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint16_t>(kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kInt32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int32_t>(kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kUInt32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint32_t>(kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kInt64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int64_t>(kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kUInt64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int64_t>(kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kFloat32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<float>(kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kFloat64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<double>(kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kString:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<const char*>(kConfig, kValueType));
        break;
      }
    default:
      LOG(FATAL) << "unknown type";
  }
  return partition_ptr;
}

void
DistArray::LoadPartitionsFromTextFile(
    JuliaEvaluator *julia_eval,
    const std::string &file_path,
    bool map,
    bool flatten_results,
    size_t num_dims,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name,
    Blob *result_buff) {
  CHECK(partitions_.empty());
  bool read = true;
  size_t partition_id = kExecutorId;
  while (read) {
    auto *dist_array_partition = CreatePartition();
    read = dist_array_partition->LoadTextFile(
        julia_eval,
        file_path,
        partition_id,
        map,
        flatten_results,
        num_dims,
        mapper_func_module,
        mapper_func_name,
        result_buff);
    if (read)
      partitions_.emplace(partition_id, dist_array_partition);
    partition_id += kConfig.kNumExecutors;
  }
}

void
DistArray::SetDims(const std::vector<int64_t> &dims) {
  dims_ = dims;
  for (auto partition : partitions_) {
    partition.second->SetDims(dims_);
  }
}

std::vector<int64_t> &
DistArray::GetDims() {
  return dims_;
}

std::unordered_map<int32_t, AbstractDistArrayPartition*>&
DistArray::GetLocalPartitions() {
  return partitions_;
}

AbstractDistArrayPartition*
DistArray::GetLocalPartition(int32_t partition_id) {
  auto partition_iter = partitions_.find(partition_id);
  if (partition_iter == partitions_.end()) return nullptr;
  return partition_iter->second;

}

std::unordered_map<int32_t, DistArray::SpacePartition>&
DistArray::GetSpaceTimePartitions() {
  return space_time_partitions_;
}

void
DistArray::AddSpaceTimePartition(int32_t space_id, int32_t time_id,
                                 AbstractDistArrayPartition* partition) {
  space_time_partitions_[space_id][time_id] = partition;
}

void
DistArray::SerializeAndClearSpaceTimePartitions(
    std::unordered_map<int32_t, Blob> *send_buff_ptr) {
  LOG(INFO) << __func__ << " id = " << kExecutorId;
  std::unordered_map<int32_t, size_t> send_buff_sizes;
  for (auto &space_partition_pair : space_time_partitions_) {
    int32_t space_id = space_partition_pair.first;
    int32_t recv_id = space_id % kConfig.kNumExecutors;
    if (recv_id == kExecutorId) continue;
    send_buff_sizes[recv_id] += sizeof(int32_t) + sizeof(size_t); // space_partition_id + num_time_partitions
    //LOG(INFO) << "send_buff_sizes[" << recv_id << "] = " << send_buff_sizes[recv_id];
    auto &space_partition = space_partition_pair.second;
    for (auto &partition_pair : space_partition) {
      auto *partition = partition_pair.second;
      // time_id + num_key_values + keys + values
      send_buff_sizes[recv_id] += sizeof(int32_t) + sizeof(size_t)
                                  + partition->GetNumKeyValues()
                                  * (sizeof(int64_t) + partition->GetValueSize());
    }
  }
  LOG(INFO) << "compute send buff sizes done";
  auto &send_buff = *send_buff_ptr;
  for (int32_t recv_id = 0; recv_id < kConfig.kNumExecutors; recv_id++) {
    if (recv_id == kExecutorId) continue;
    auto iter = send_buff_sizes.find(recv_id);
    if (iter == send_buff_sizes.end()) continue;
    size_t buff_size = iter->second;
    send_buff[recv_id].resize(buff_size);
  }

  std::unordered_map<int32_t, size_t> send_buff_offsets;
  for (size_t i = 0; i < kConfig.kNumExecutors; i++) {
    send_buff_offsets[i] = 0;
  }

  for (auto &space_partition_pair : space_time_partitions_) {
    int32_t space_id = space_partition_pair.first;

    int32_t recv_id = space_id % kConfig.kNumExecutors;
    if (recv_id == kExecutorId) continue;
    auto &space_partition = space_partition_pair.second;
    auto &buff = send_buff[recv_id];
    uint8_t *mem = buff.data() + send_buff_offsets[recv_id];
    *reinterpret_cast<int32_t*>(mem) = space_id;
    mem += sizeof(int32_t);
    *reinterpret_cast<size_t*>(mem) = space_partition.size();
    mem += sizeof(size_t);
    for (auto &partition_pair : space_partition) {
      int32_t time_id = partition_pair.first;
      auto *partition = partition_pair.second;
      *reinterpret_cast<int32_t*>(mem) = time_id;
      mem += sizeof(int32_t);
      *reinterpret_cast<size_t*>(mem) = partition->GetNumKeyValues();
      mem += sizeof(size_t);
      auto &keys = partition->GetKeys();
      memcpy(mem, keys.data(), keys.size() * sizeof(int64_t));
      mem += keys.size() * sizeof(int64_t);
      partition->CopyValues(mem);
      mem += partition->GetNumKeyValues() * partition->GetValueSize();
      delete partition;
    }
    send_buff_offsets[recv_id] = mem - buff.data();
  }

  for (size_t i = 0; i < kConfig.kNumExecutors; i++) {
    if (i == kExecutorId) continue;
    CHECK_EQ(send_buff_offsets[i], send_buff_sizes[i]);
  }
  auto iter = space_time_partitions_.begin();

  while (iter != space_time_partitions_.end()) {
    int32_t space_id = iter->first;
    int32_t recv_id = space_id % kConfig.kNumExecutors;
    if (recv_id == kExecutorId) {
      iter++;
      continue;
    }
    space_time_partitions_.erase(iter++);
  }
}

void
DistArray::DeserializeSpaceTimePartitions(
    const uint8_t *mem, size_t mem_size) {
  const uint8_t *cursor = mem;
  while (cursor - mem < mem_size) {
    int32_t space_id = *reinterpret_cast<const int32_t*>(cursor);
    cursor += sizeof(int32_t);
    size_t num_time_partitions = *reinterpret_cast<const size_t*>(cursor);
    cursor += sizeof(size_t);
    for (size_t i = 0; i < num_time_partitions; i++) {
      int32_t time_id = *reinterpret_cast<const int32_t*>(cursor);
      cursor += sizeof(int32_t);
      size_t num_key_values = *reinterpret_cast<const size_t*>(cursor);
      cursor += sizeof(size_t);
      auto *partition = space_time_partitions_[space_id][time_id];
      if (partition == nullptr) {
        LOG(INFO) << "created partition for " << space_id
                  << " " << time_id;
        partition = CreatePartition();
        space_time_partitions_[space_id][time_id] = partition;
      }
      const int64_t* keys = reinterpret_cast<const int64_t*>(cursor);
      cursor += sizeof(int64_t) * num_key_values;
      const uint8_t *values = cursor;
      cursor += kValueSize * num_key_values;
      for (size_t j = 0; j < num_key_values; j++) {
        partition->AppendKeyValue(keys[j], values + kValueSize * j);
      }
      LOG(INFO) << "appended " << num_key_values << " to "
                << space_id << " " << time_id;
      AddSpaceTimePartition(space_id, time_id, partition);
    }
  }
}

}
}
