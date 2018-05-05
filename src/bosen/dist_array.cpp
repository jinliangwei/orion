#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_partition.hpp>
#include <orion/bosen/julia_evaluator.hpp>

namespace orion {
namespace bosen {

DistArray::DistArray(
    int32_t id,
    const Config &config,
    bool is_server,
    type::PrimitiveType value_type,
    int32_t executor_id,
    int32_t server_id,
    size_t num_dims,
    DistArrayParentType parent_type,
    DistArrayInitType init_type,
    DistArrayMapType map_type,
    DistArrayPartitionScheme partition_scheme,
    JuliaModule map_func_module,
    const std::string &map_func_name,
    type::PrimitiveType random_init_type,
    bool flatten_results,
    bool is_dense,
    const std::string &symbol,
    JuliaThreadRequester *julia_requester):
    kId(id),
    kConfig(config),
    kIsServer(is_server),
    kValueType(value_type),
    kValueSize(type::SizeOf(value_type)),
    kExecutorId(executor_id),
    kServerId(server_id),
    meta_(num_dims,
          parent_type,
          init_type,
          map_type,
          partition_scheme,
          map_func_module,
          map_func_name,
          random_init_type,
          flatten_results,
          is_dense,
          symbol),
    julia_requester_(julia_requester) {
}

DistArray::~DistArray() {
  for (auto &partition_pair : partitions_) {
    delete partition_pair.second;
  }
}

AbstractDistArrayPartition*
DistArray::CreatePartition() {
  AbstractDistArrayPartition *partition_ptr = nullptr;
  switch(kValueType) {
    case type::PrimitiveType::kVoid:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<void>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    case type::PrimitiveType::kInt8:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int8_t>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    case type::PrimitiveType::kUInt8:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint8_t>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    case type::PrimitiveType::kInt16:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int16_t>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    case type::PrimitiveType::kUInt16:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint16_t>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    case type::PrimitiveType::kInt32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int32_t>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    case type::PrimitiveType::kUInt32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint32_t>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    case type::PrimitiveType::kInt64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int64_t>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    case type::PrimitiveType::kUInt64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int64_t>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    case type::PrimitiveType::kFloat32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<float>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    case type::PrimitiveType::kFloat64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<double>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    case type::PrimitiveType::kString:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<std::string>(this, kConfig, kValueType, julia_requester_));
        break;
      }
    default:
      LOG(FATAL) << "unknown type " << static_cast<int>(kValueType);
  }
  return partition_ptr;
}

void
DistArray::LoadPartitionsFromTextFile(std::string file_path) {
  CHECK(partitions_.empty());
  bool read = true;
  size_t partition_id = kExecutorId;
  while (read) {
    auto *dist_array_partition = CreatePartition();
    read = dist_array_partition->LoadTextFile(file_path, partition_id);
    if (read)
      partitions_.emplace(partition_id, dist_array_partition);
    partition_id += kConfig.kNumExecutors;
  }
}

void
DistArray::ParseBufferedText(Blob *max_ids,
                             const std::vector<size_t> &line_number_start) {

  size_t num_dims = meta_.GetNumDims();
  max_ids->resize(sizeof(int64_t) * num_dims);
  if (line_number_start.size() == 0) {
    for (auto &partition_pair : partitions_) {
      partition_pair.second->ParseText(max_ids, 0);
    }
  } else {
    size_t i = 0;
    for (auto &partition_pair : partitions_) {
      partition_pair.second->ParseText(max_ids,
                                       line_number_start[i]);
      i += 1;
    }
  }
}

void
DistArray::GetPartitionTextBufferNumLines(std::vector<int64_t> *partition_ids,
                                          std::vector<size_t> *num_lines) {
  size_t num_partitions = partitions_.size();
  partition_ids->resize(num_partitions);
  num_lines->resize(num_partitions);
  size_t i = 0;
  for (auto &partition_pair : partitions_) {
    int64_t partition_id = partition_pair.first;
    auto *partition = partition_pair.second;
    size_t nlines = partition->CountNumLines();
    (*partition_ids)[i] = partition_id;
    (*num_lines)[i] = nlines;
    i++;
  }
}

void
DistArray::Init() {
  size_t num_params = 1;
  const auto &dims = meta_.GetDims();
  for (auto d : dims) {
    num_params *= d;
  }
  size_t num_params_this_executor = num_params / kConfig.kNumExecutors
                                    + ((kExecutorId < (num_params % kConfig.kNumExecutors))
                                       ? 1 : 0);
  if (num_params_this_executor == 0) return;
  int64_t key_begin = (kExecutorId < (num_params % kConfig.kNumExecutors))
                      ? (num_params / kConfig.kNumExecutors + 1) * kExecutorId
                      : (num_params / kConfig.kNumExecutors) * kExecutorId
                      + (num_params % kConfig.kNumExecutors);
  auto *dist_array_partition = CreatePartition();
  dist_array_partition->Init(key_begin, num_params_this_executor);
  partitions_.emplace(std::make_pair(kExecutorId, dist_array_partition));
}

void
DistArray::Map(DistArray* child_dist_array) {
  for (auto& partition_pair : partitions_) {
    int64_t partition_id = partition_pair.first;
    auto* partition = partition_pair.second;
    auto child_partition_pair = child_dist_array->GetAndCreateLocalPartition(partition_id);
    auto *child_partition = child_partition_pair.first;
    partition->Map(child_partition);
  }

  for (auto &time_partitions : space_time_partitions_) {
    int64_t space_id = time_partitions.first;
    for (auto &partition_pair : time_partitions.second) {
      int64_t time_id = partition_pair.first;
      auto *partition = partition_pair.second;
      auto child_partition_pair = child_dist_array->GetAndCreateLocalPartition(space_id, time_id);
      auto *child_partition = child_partition_pair.first;
      partition->Map(child_partition);
    }
  }
}

void
DistArray::ComputeHashRepartition(size_t num_partitions) {
  std::vector<AbstractDistArrayPartition*> partition_buff;
  GetAndClearLocalPartitions(&partition_buff);

  for (auto dist_array_partition : partition_buff) {
    dist_array_partition->BuildKeyValueBuffersFromSparseIndex();
    dist_array_partition->ComputeHashRepartitionIdsAndRepartition(num_partitions);
    delete dist_array_partition;
  }
}

void
DistArray::ComputeRepartition(const std::string &repartition_func_name) {
  std::vector<AbstractDistArrayPartition*> partition_buff;
  GetAndClearLocalPartitions(&partition_buff);

  for (auto dist_array_partition : partition_buff) {
    dist_array_partition->BuildKeyValueBuffersFromSparseIndex();
    dist_array_partition->ComputeRepartitionIdsAndRepartition(repartition_func_name);
    delete dist_array_partition;
  }

  bool from_server = meta_.GetPartitionScheme() == DistArrayPartitionScheme::kHashServer;
  bool to_server = meta_.GetPartitionScheme() == DistArrayPartitionScheme::kHashServer;

  bool repartition_recv = true;
  if ((to_server && (!kIsServer || (from_server && kConfig.kNumServers == 1))) ||
      (!to_server && (kIsServer || (!from_server && kConfig.kNumExecutors == 1)))
      ) {
    repartition_recv = false;
  }

  if (!repartition_recv) {
    ComputeMaxPartitionIds();
    CheckAndBuildIndex();
  }
}

void
DistArray::SetDims(const std::vector<int64_t> &dims) {
  meta_.AssignDims(dims.data());
  const auto &my_dims = meta_.GetDims();
  for (auto partition : partitions_) {
    partition.second->ComputeKeysFromBuffer(my_dims);
  }
}

void
DistArray::SetDims(const int64_t* dims, size_t num_dims) {
  meta_.AssignDims(dims);
  const auto &my_dims = meta_.GetDims();
  for (auto partition : partitions_) {
    partition.second->ComputeKeysFromBuffer(my_dims);
  }
}

const std::vector<int64_t> &
DistArray::GetDims() const {
  return meta_.GetDims();
}

DistArrayMeta &
DistArray::GetMeta() {
  return meta_;
}

type::PrimitiveType
DistArray::GetValueType() {
  return kValueType;
}

AbstractDistArrayPartition*
DistArray::GetLocalPartition(int32_t partition_id) {
  auto partition_iter = partitions_.find(partition_id);
  if (partition_iter == partitions_.end()) return nullptr;
  return partition_iter->second;
}

AbstractDistArrayPartition*
DistArray::GetLocalPartition(int32_t space_id,
                             int32_t time_id) {
  auto time_partition_map_iter = space_time_partitions_.find(space_id);
  if (time_partition_map_iter == space_time_partitions_.end()) return nullptr;
  auto &time_partition_map = time_partition_map_iter->second;
  auto partition_iter = time_partition_map.find(time_id);
  if (partition_iter == time_partition_map.end()) return nullptr;
  return partition_iter->second;
}

std::pair<AbstractDistArrayPartition*, bool>
DistArray::GetAndCreateLocalPartition(int32_t partition_id) {
  bool new_created = false;
  auto partition_iter = partitions_.find(partition_id);
  if (partition_iter == partitions_.end()) {
    auto partition_pair = partitions_.emplace(partition_id, CreatePartition());
    partition_iter = partition_pair.first;
    new_created = true;
  }
  return std::make_pair(partition_iter->second, new_created);
}

std::pair<AbstractDistArrayPartition*, bool>
DistArray::GetAndCreateLocalPartition(int32_t space_id,
                                      int32_t time_id) {
  bool new_created = false;
  auto time_partition_map_iter = space_time_partitions_.find(space_id);
  if (time_partition_map_iter == space_time_partitions_.end()) {
    auto iter_pair = space_time_partitions_.emplace(space_id, TimePartitionMap());
    time_partition_map_iter = iter_pair.first;
    new_created = true;
  }
  auto &time_partition_map = time_partition_map_iter->second;
  auto partition_iter = time_partition_map.find(time_id);
  if (partition_iter == time_partition_map.end()) {
    auto iter_pair = time_partition_map.emplace(time_id, CreatePartition());
    partition_iter = iter_pair.first;
    new_created = true;
  }
  return std::make_pair(partition_iter->second, new_created);
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
    ExecutorSendBufferMap *send_buff_ptr) {
  if (meta_.GetPartitionScheme() == DistArrayPartitionScheme::kSpaceTime) {
    RepartitionSerializeAndClearSpaceTime(send_buff_ptr);
  } else {
    RepartitionSerializeAndClear1D(send_buff_ptr);
  }
}

void
DistArray::RepartitionSerializeAndClearSpaceTime(
    ExecutorSendBufferMap* send_buff_ptr) {

  std::unordered_map<
    int32_t,
    std::vector<
      std::pair<int32_t,
                SendDataBuffer>
      >
    > send_partition_buffs;
  std::unordered_map<int32_t, size_t> send_buff_sizes;

  auto iter = space_time_partitions_.begin();
  while (iter != space_time_partitions_.end()) {
    auto &time_partition_map_pair = *iter;
    int32_t space_id = time_partition_map_pair.first;
    auto& time_partition_map = time_partition_map_pair.second;
    int32_t recv_id = space_id % kConfig.kNumExecutors;
    if (!kIsServer && (recv_id == kExecutorId)) {
      iter++;
      continue;
    }
    // space_partition_id + num_partitions
    send_buff_sizes[recv_id] += sizeof(int32_t) + sizeof(size_t);
    for (auto &partition_pair : time_partition_map) {
      int32_t time_id = partition_pair.first;
      auto *partition = partition_pair.second;
      SendDataBuffer partition_buff = partition->Serialize();
      send_partition_buffs[space_id].emplace_back(time_id, partition_buff);
      send_buff_sizes[recv_id] += sizeof(int32_t) + partition_buff.second;
      delete partition;
    }
    time_partition_map.clear();
    space_time_partitions_.erase(iter++);
  }

  auto &send_buff = *send_buff_ptr;
  for (int32_t recv_id = 0; recv_id < kConfig.kNumExecutors; recv_id++) {
    if (!kIsServer && recv_id == kExecutorId) continue;
    auto iter = send_buff_sizes.find(recv_id);
    if (iter == send_buff_sizes.end()) continue;
    size_t buff_size = iter->second;
    send_buff[recv_id] = std::make_pair(new uint8_t[buff_size], buff_size);
  }

  std::unordered_map<int32_t, size_t> send_buff_offsets;
  for (size_t i = 0; i < kConfig.kNumExecutors; i++) {
    send_buff_offsets[i] = 0;
  }

  for (const auto &partition_buff_vec_pair : send_partition_buffs) {
    int32_t space_id = partition_buff_vec_pair.first;
    int32_t recv_id = space_id % kConfig.kNumExecutors;
    if (!kIsServer && (recv_id == kExecutorId)) continue;
    const auto &partition_buff_vec = partition_buff_vec_pair.second;

    auto *buff = send_buff[recv_id].first;
    uint8_t *mem = buff + send_buff_offsets[recv_id];
    *reinterpret_cast<int32_t*>(mem) = space_id;
    mem += sizeof(int32_t);
    *reinterpret_cast<size_t*>(mem) = partition_buff_vec.size();
    mem += sizeof(size_t);
    for (const auto &partition_buff_pair : partition_buff_vec) {
      int32_t time_id = partition_buff_pair.first;
      *reinterpret_cast<int32_t*>(mem) = time_id;
      mem += sizeof(int32_t);
      const auto& partition_buff = partition_buff_pair.second;
      memcpy(mem, partition_buff.first, partition_buff.second);
      mem += partition_buff.second;
      delete[] partition_buff.first;
    }
    send_buff_offsets[recv_id] = mem - buff;
  }
}

void
DistArray::RepartitionSerializeAndClear1D(
    ExecutorSendBufferMap* send_buff_ptr) {
  std::vector<
    std::pair<int32_t, SendDataBuffer>
    > send_partition_buffs;
  std::unordered_map<int32_t, size_t> send_buff_sizes;
  bool send_to_server = meta_.GetPartitionScheme() == DistArrayPartitionScheme::kHashServer;
  auto iter = partitions_.begin();
  while (iter != partitions_.end()) {
    auto &partition_pair = *iter;
    int32_t partition_id = partition_pair.first;
    auto *partition = partition_pair.second;
    int32_t recv_id = send_to_server ? partition_id % kConfig.kNumServers
                      : partition_id % kConfig.kNumExecutors;
    if ((send_to_server && kIsServer && recv_id == kServerId)
        || (!send_to_server && !kIsServer && recv_id == kExecutorId)) {
      iter++;
      continue;
    }
    auto partition_buff = partition->Serialize();
    send_partition_buffs.emplace_back(partition_id, partition_buff);
    send_buff_sizes[recv_id] += sizeof(int32_t) + partition_buff.second;
    delete partition;
    partitions_.erase(iter++);
  }

  auto &send_buff = *send_buff_ptr;
  size_t num_receivers = send_to_server ? kConfig.kNumServers
                         : kConfig.kNumExecutors;
  for (int32_t recv_id = 0; recv_id < num_receivers; recv_id++) {
    if ((send_to_server && kIsServer && recv_id == kServerId)
        || (!send_to_server && !kIsServer && recv_id == kExecutorId)) continue;
    auto iter = send_buff_sizes.find(recv_id);
    if (iter == send_buff_sizes.end()) continue;
    size_t buff_size = iter->second;
    send_buff[recv_id] = std::make_pair(new uint8_t[buff_size], buff_size);
  }

  std::unordered_map<int32_t, size_t> send_buff_offsets;
  for (size_t i = 0; i < num_receivers; i++) {
    send_buff_offsets[i] = 0;
  }

  for (const auto &partition_buff_pair : send_partition_buffs) {
    int32_t partition_id = partition_buff_pair.first;
    int32_t recv_id = partition_id % num_receivers;
    if ((send_to_server && kIsServer && recv_id == kServerId)
        || (!send_to_server && !kIsServer && recv_id == kExecutorId)) continue;

    auto *buff = send_buff[recv_id].first;
    uint8_t *mem = buff + send_buff_offsets[recv_id];
    *reinterpret_cast<int32_t*>(mem) = partition_id;
    mem += sizeof(int32_t);
    const auto& partition_buff = partition_buff_pair.second;
    memcpy(mem, partition_buff.first, partition_buff.second);

    mem += partition_buff.second;
    delete[] partition_buff.first;
    send_buff_offsets[recv_id] = mem - buff;
  }
}

void
DistArray::RepartitionDeserialize(
    PeerRecvRepartitionDistArrayDataBuffer *data_buff_ptr) {
  auto byte_buffs = data_buff_ptr->byte_buffs;
  for (auto &buff_pair : byte_buffs) {
    auto &buff = buff_pair.second;
    RepartitionDeserializeInternal(buff.GetBytes(), buff.GetSize());
  }
  delete data_buff_ptr;
  ComputeMaxPartitionIds();
  CheckAndBuildIndex();
}

void
DistArray::RepartitionDeserializeInternal(
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
      auto partition_pair = GetAndCreateLocalPartition(space_id, time_id);
      auto *partition = partition_pair.first;
      cursor = partition->DeserializeAndAppend(cursor);
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
    auto partition_pair = GetAndCreateLocalPartition(partition_id);
    auto *partition = partition_pair.first;
    cursor = partition->DeserializeAndAppend(cursor);
  }
}

void
DistArray::CheckAndBuildIndex() {
  auto index_type = meta_.GetIndexType();
  switch (index_type) {
    case DistArrayIndexType::kNone:
      {
        BuildPartitionKeyValueBuffersFromSparseIndex();
      }
      break;
    case DistArrayIndexType::kRange:
      {
        BuildPartitionIndices();
      }
      break;
    default:
      LOG(FATAL) << "unknown index type " << static_cast<int>(index_type);
  }
}

void
DistArray::BuildPartitionIndices() {
  if (meta_.GetPartitionScheme() == DistArrayPartitionScheme::kSpaceTime) {
    for (auto &time_partition_map : space_time_partitions_) {
      for (auto &partition_pair : time_partition_map.second) {
        auto partition = partition_pair.second;
        partition->BuildIndex();
      }
    }
  } else {
    for (auto &partition_pair : partitions_) {
      auto partition = partition_pair.second;
      partition->BuildIndex();
    }
  }
}

void
DistArray::BuildPartitionKeyValueBuffersFromSparseIndex() {
  if (meta_.GetPartitionScheme() == DistArrayPartitionScheme::kSpaceTime) {
    for (auto &time_partition_map : space_time_partitions_) {
      for (auto &partition_pair : time_partition_map.second) {
        auto partition = partition_pair.second;
        partition->BuildKeyValueBuffersFromSparseIndex();
      }
    }
  } else {
    for (auto &partition_pair : partitions_) {
      auto partition = partition_pair.second;
      partition->BuildKeyValueBuffersFromSparseIndex();
    }
  }
}

void
DistArray::GetAndSerializeValue(int64_t key, Blob *bytes_buff) {
  auto iter = partitions_.begin();
  CHECK(iter != partitions_.end());
  auto *partition = iter->second;
  Blob temp_buff;
  partition->GetAndSerializeValue(key, bytes_buff);
}

void
DistArray::GetAndSerializeValues(const int64_t *keys,
                                 size_t num_keys,
                                 Blob *bytes_buff) {
  auto iter = partitions_.begin();
  CHECK(iter != partitions_.end());
  auto *partition = iter->second;
  partition->GetAndSerializeValues(keys, num_keys, bytes_buff);
}

void
DistArray::GetMaxPartitionIds(
    std::vector<int32_t>* ids) {
  if (meta_.GetPartitionScheme() == DistArrayPartitionScheme::kSpaceTime) {
    GetMaxPartitionIdsSpaceTime(ids);
  } else {
    GetMaxPartitionIds1D(ids);
  }
}

void
DistArray::GetMaxPartitionIdsSpaceTime(
    std::vector<int32_t>* ids) {
  ids->resize(2);
  int32_t max_space_id = 0,
           max_time_id = 1;
  for (auto &time_partition_map : space_time_partitions_) {
    int32_t space_id = time_partition_map.first;
    max_space_id = std::max(max_space_id, space_id);
    for (auto &partition_pair : time_partition_map.second) {
      int32_t time_id = partition_pair.first;
      max_time_id = std::max(max_time_id, time_id);
    }
  }
  (*ids)[0] = max_space_id;
  (*ids)[1] = max_time_id;
}

void
DistArray::GetMaxPartitionIds1D(
    std::vector<int32_t>* ids) {
  ids->resize(1);
  int32_t max_id = 0;
  for (auto &partition_pair : partitions_) {
    int32_t partition_id = partition_pair.first;
    max_id = std::max(max_id, partition_id);
  }
  (*ids)[0] = max_id;
}

void
DistArray::CreateDistArrayBuffer(const std::string &serialized_value_type) {
  const auto &symbol = meta_.GetSymbol();
  const auto &dims = meta_.GetDims();
  bool is_dense = meta_.IsDense();
  const auto &init_value = meta_.GetInitValue();
  JuliaEvaluator::DefineDistArray(kId, symbol, serialized_value_type,
                                  dims, is_dense, true, init_value,
                                  this);
  buffer_partition_.reset(CreatePartition());
}

AbstractDistArrayPartition*
DistArray::GetBufferPartition() {
  return buffer_partition_.get();
}

void
DistArray::DeletePartition(int32_t partition_id) {
  auto partition_iter = partitions_.find(partition_id);
  CHECK(partition_iter != partitions_.end());
  delete partition_iter->second;
  partitions_.erase(partition_id);
}

void
DistArray::ComputeMaxPartitionIds() {
  LOG(INFO) << __func__ << " dist_array id = " << kId;
  if (meta_.GetPartitionScheme() == DistArrayPartitionScheme::kSpaceTime) {
    ComputeMaxPartitionIdsSpaceTime();
  } else {
    ComputeMaxPartitionIds1D();
  }
}

void
DistArray::ComputeMaxPartitionIdsSpaceTime() {
  int64_t max_space_id = 0, max_time_id = 0;
  for (auto &time_partition_map_pair : space_time_partitions_) {
    int64_t space_id = time_partition_map_pair.first;
    auto &time_partition_map = time_partition_map_pair.second;
    if (max_space_id < space_id) max_space_id = space_id;
    for (auto &partition_pair : time_partition_map) {
      int32_t time_id = partition_pair.first;
      if (max_time_id < time_id) max_time_id = time_id;
    }
  }
  meta_.SetMaxPartitionIds(max_space_id, max_time_id);
}

void
DistArray::ComputeMaxPartitionIds1D() {
  int32_t max_partition_id = 0;
  for (auto &partition_pair : partitions_) {
    int32_t partition_id = partition_pair.first;
    if (max_partition_id < partition_id) max_partition_id = partition_id;
  }
  meta_.SetMaxPartitionIds(max_partition_id);
}

}
}
