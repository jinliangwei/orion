#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_partition.hpp>
#include <orion/bosen/julia_evaluator.hpp>

namespace orion {
namespace bosen {

DistArray::DistArray(
    const Config &config,
    type::PrimitiveType value_type,
    int32_t executor_id,
    size_t num_dims,
    task::DistArrayParentType parent_type,
    task::DistArrayInitType init_type,
    const DistArrayMeta *parent_dist_array_meta,
    bool is_dense):
    kConfig(config),
    kValueType(value_type),
    kValueSize(type::SizeOf(value_type)),
    kExecutorId(executor_id),
    meta_(num_dims, parent_type, init_type,
          parent_dist_array_meta,
          is_dense) {
  if (num_dims > 0)
    dims_.resize(num_dims);
}

DistArray::~DistArray() {
  for (auto &partition_pair : partitions_) {
    delete partition_pair.second;
  }
}

/*DistArray::DistArray(DistArray &&other):
    kConfig(other.kConfig),
    kValueType(other.kValueType),
    kValueSize(other.kValueSize),
    kExecutorId(other.kExecutorId),
    partitions_(other.partitions_),
    meta_(other.meta_) {
  other.partitions_.clear();
  } */

AbstractDistArrayPartition*
DistArray::CreatePartition() {
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
                new DistArrayPartition<int8_t>(this, kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kUInt8:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint8_t>(this, kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kInt16:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int16_t>(this, kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kUInt16:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint16_t>(this, kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kInt32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int32_t>(this, kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kUInt32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint32_t>(this, kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kInt64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int64_t>(this, kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kUInt64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int64_t>(this, kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kFloat32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<float>(this, kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kFloat64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<double>(this, kConfig, kValueType));
        break;
      }
    case type::PrimitiveType::kString:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<const char*>(this, kConfig, kValueType));
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
    std::string file_path,
    task::DistArrayMapType map_type,
    bool flatten_results,
    size_t num_dims,
    JuliaModule mapper_func_module,
    std::string mapper_func_name,
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
        map_type,
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
  meta_.AssignDims(dims.data());
}

void
DistArray::SetDims(const int64_t* dims, size_t num_dims) {
  dims_.resize(num_dims);
  memcpy(dims_.data(), dims, num_dims * sizeof(int64_t));
  for (auto partition : partitions_) {
    partition.second->SetDims(dims_);
  }
  meta_.AssignDims(dims);
}

std::vector<int64_t> &
DistArray::GetDims() {
  return dims_;
}

DistArrayMeta &
DistArray::GetMeta() {
  return meta_;
}

type::PrimitiveType
DistArray::GetValueType() {
  return kValueType;
}

DistArray::PartitionMap&
DistArray::GetLocalPartitionMap() {
  return partitions_;
}

AbstractDistArrayPartition*
DistArray::GetLocalPartition(int32_t partition_id) {
  auto partition_iter = partitions_.find(partition_id);
  if (partition_iter == partitions_.end()) return nullptr;
  return partition_iter->second;
}

DistArray::SpaceTimePartitionMap&
DistArray::GetSpaceTimePartitionMap() {
  return space_time_partitions_;
}

void
DistArray::GetAndClearLocalPartitions(std::vector<AbstractDistArrayPartition*>
                                      *buff) {
  if (space_time_partitions_.empty()) {
    buff->resize(partitions_.size());
    size_t i = 0;
    for (auto &dist_array_partition_pair : partitions_) {
      auto *dist_array_partition = dist_array_partition_pair.second;
      (*buff)[i] = dist_array_partition;
      i++;
    }
    partitions_.clear();
  } else {
    buff->clear();
    for (auto &time_partition_map_pair : space_time_partitions_) {
      auto &time_partition_map = time_partition_map_pair.second;
      for (auto &dist_array_partition_pair : time_partition_map) {
        auto *dist_array_partition = dist_array_partition_pair.second;
        buff->push_back(dist_array_partition);
      }
    }
    space_time_partitions_.clear();
  }
}

void
DistArray::RepartitionSerializeAndClear(
        std::unordered_map<int32_t, Blob> *send_buff_ptr) {
  if (meta_.GetPartitionScheme() == DistArrayPartitionScheme::kSpaceTime) {
    RepartitionSerializeAndClearSpaceTime(send_buff_ptr);
  } else {
    RepartitionSerializeAndClear1D(send_buff_ptr);
  }
}

void
DistArray::RepartitionSerializeAndClearSpaceTime(
    std::unordered_map<int32_t, Blob> *send_buff_ptr) {
  LOG(INFO) << __func__ << " id = " << kExecutorId;
  std::unordered_map<int32_t, size_t> send_buff_sizes;
  for (auto &time_partition_map_pair : space_time_partitions_) {
    int32_t space_id = time_partition_map_pair.first;
    int32_t recv_id = space_id % kConfig.kNumExecutors;
    if (recv_id == kExecutorId) continue;
    send_buff_sizes[recv_id] += sizeof(int32_t) + sizeof(size_t); // space_partition_id + num_time_partitions
    //LOG(INFO) << "send_buff_sizes[" << recv_id << "] = " << send_buff_sizes[recv_id];
    auto &partitions = time_partition_map_pair.second;
    for (auto &partition_pair : partitions) {
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

  for (auto &time_partition_map_pair : space_time_partitions_) {
    int32_t space_id = time_partition_map_pair.first;
    int32_t recv_id = space_id % kConfig.kNumExecutors;
    if (recv_id == kExecutorId) continue;
    auto &partitions = time_partition_map_pair.second;
    auto &buff = send_buff[recv_id];
    uint8_t *mem = buff.data() + send_buff_offsets[recv_id];
    *reinterpret_cast<int32_t*>(mem) = space_id;
    mem += sizeof(int32_t);
    *reinterpret_cast<size_t*>(mem) = partitions.size();
    mem += sizeof(size_t);
    for (auto &partition_pair : partitions) {
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
DistArray::RepartitionSerializeAndClear1D(
    std::unordered_map<int32_t, Blob> *send_buff_ptr) {
  LOG(INFO) << __func__ << " id = " << kExecutorId;
  std::unordered_map<int32_t, size_t> send_buff_sizes;
  for (auto &partition_pair : partitions_) {
    int32_t partition_id = partition_pair.first;
    auto *partition = partition_pair.second;
    int32_t recv_id = partition_id % kConfig.kNumExecutors;
    if (recv_id == kExecutorId) continue;
    send_buff_sizes[recv_id] += sizeof(int32_t) + sizeof(size_t)
                                + partition->GetNumKeyValues()
                                * (sizeof(int64_t) + partition->GetValueSize());
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

  for (auto &partition_pair : partitions_) {
    int32_t partition_id = partition_pair.first;
    auto *partition = partition_pair.second;
    int32_t recv_id = partition_id % kConfig.kNumExecutors;
    if (recv_id == kExecutorId) continue;
    auto &buff = send_buff[recv_id];
    uint8_t *mem = buff.data() + send_buff_offsets[recv_id];
    *reinterpret_cast<int32_t*>(mem) = partition_id;
    mem += sizeof(int32_t);
    *reinterpret_cast<size_t*>(mem) = partition->GetNumKeyValues();
    mem += sizeof(size_t);
    auto &keys = partition->GetKeys();
    memcpy(mem, keys.data(), keys.size() * sizeof(int64_t));
    mem += keys.size() * sizeof(int64_t);
    partition->CopyValues(mem);
    mem += partition->GetNumKeyValues() * partition->GetValueSize();
    delete partition;
    send_buff_offsets[recv_id] = mem - buff.data();
  }

  for (size_t i = 0; i < kConfig.kNumExecutors; i++) {
    if (i == kExecutorId) continue;
    CHECK_EQ(send_buff_offsets[i], send_buff_sizes[i]);
  }
  auto iter = partitions_.begin();

  while (iter != partitions_.end()) {
    int32_t partition_id = iter->first;
    int32_t recv_id = partition_id % kConfig.kNumExecutors;
    if (recv_id == kExecutorId) {
      iter++;
      continue;
    }
    partitions_.erase(iter++);
  }
}

void
DistArray::RepartitionDeserialize(
    const uint8_t *mem, size_t mem_size) {
  if (meta_.GetPartitionScheme() == DistArrayPartitionScheme::kSpaceTime) {
    RepartitionDeserializeSpaceTime(mem, mem_size);
  } else {
    RepartitionDeserialize1D(mem, mem_size);
  }
}

void
DistArray::RepartitionDeserializeSpaceTime(
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
    }
  }
}

void
DistArray::RepartitionDeserialize1D(
    const uint8_t *mem, size_t mem_size) {
  const uint8_t *cursor = mem;
  while (cursor - mem < mem_size) {
    int32_t partition_id = *reinterpret_cast<const int32_t*>(cursor);
    cursor += sizeof(int32_t);
    size_t num_key_values = *reinterpret_cast<const size_t*>(cursor);
    cursor += sizeof(size_t);
    auto *partition = partitions_[partition_id];
      if (partition == nullptr) {
        partition = CreatePartition();
        partitions_[partition_id] = partition;
      }
      const int64_t* keys = reinterpret_cast<const int64_t*>(cursor);
      cursor += sizeof(int64_t) * num_key_values;
      const uint8_t *values = cursor;
      cursor += kValueSize * num_key_values;
      for (size_t j = 0; j < num_key_values; j++) {
        partition->AppendKeyValue(keys[j], values + kValueSize * j);
      }
  }
}

void
DistArray::RandomInit(
      JuliaEvaluator *julia_eval,
      task::DistArrayInitType init_type,
      task::DistArrayMapType map_type,
      JuliaModule mapper_func_module,
      std::string mapper_func_name,
      type::PrimitiveType random_init_type) {
  size_t num_params = 1;
  for (auto d : dims_) {
    num_params *= d;
  }
  size_t num_params_this_executor = num_params / kConfig.kNumExecutors
                                    + ((kExecutorId < (num_params % kConfig.kNumExecutors))
                                       ? 1 : 0);
  LOG(INFO) << __func__ << " num_params_this_executor = " << num_params_this_executor;
  if (num_params_this_executor == 0) return;
  int64_t key_begin = (kExecutorId < (num_params % kConfig.kNumExecutors))
                      ? (num_params / kConfig.kNumExecutors + 1) * kExecutorId
                      : (num_params / kConfig.kNumExecutors) * kExecutorId + (num_params % kConfig.kNumExecutors);
  auto *dist_array_partition = CreatePartition();
  dist_array_partition->RandomInit(
      julia_eval,
      dims_,
      key_begin,
      num_params_this_executor,
      init_type,
      map_type,
      mapper_func_module,
      mapper_func_name,
      random_init_type);
  partitions_.emplace(std::make_pair(kExecutorId, dist_array_partition));

}

void
DistArray::CheckAndBuildIndex() {
}

}
}
