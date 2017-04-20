#pragma once

#include <unordered_map>
#include <orion/noncopyable.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_partition.hpp>
#include <orion/bosen/type.hpp>
#include <orion/bosen/julia_evaluator.hpp>

namespace orion {
namespace bosen {

class DistArray {
 private:
  const Config &kConfig;
  const type::PrimitiveType kValueType;
  const int32_t kExecutorId;
  std::unordered_map<int32_t, AbstractDistArrayPartition*> partitions_;
  std::vector<int64_t> dims_;
 public:
  DistArray(const Config& config,
            type::PrimitiveType value_type, int32_t executor_id);
  ~DistArray();
  DistArray(DistArray &&other);
  void LoadPartitionFromTextFile(
      JuliaEvaluator *julia_eval,
      const std::string &file_path,
      bool map,
      bool flatten_results,
      size_t num_dims,
      JuliaModule mapper_func_module,
      const std::string &mapper_func_name,
      Blob *result_buff);
  void CreatePartitionFromParent();
  void CreatePartitoin();
  void SetDims(const std::vector<int64_t> &dims);

 private:
  DISALLOW_COPY(DistArray);
  AbstractDistArrayPartition *CreatePartition();
};

DistArray::DistArray(
    const Config &config,
    type::PrimitiveType value_type,
    int32_t executor_id):
    kConfig(config),
    kValueType(value_type),
    kExecutorId(executor_id) { }

DistArray::~DistArray() {
  for (auto &partition_pair : partitions_) {
    delete partition_pair.second;
  }
}

DistArray::DistArray(DistArray &&other):
    kConfig(other.kConfig),
    kValueType(other.kValueType),
    kExecutorId(other.kExecutorId),
    partitions_(other.partitions_) {
  other.partitions_.clear();
}

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
DistArray::LoadPartitionFromTextFile(
    JuliaEvaluator *julia_eval,
    const std::string &file_path,
    bool map,
    bool flatten_results,
    size_t num_dims,
    JuliaModule mapper_func_module,
    const std::string &mapper_func_name,
    Blob *result_buff) {
  CHECK(partitions_.empty());
  auto *dist_array_partition = CreatePartition();
  dist_array_partition->LoadTextFile(
      julia_eval,
      file_path,
      kExecutorId,
      map,
      flatten_results,
      num_dims, mapper_func_module, mapper_func_name,
      result_buff);
  partitions_.emplace(0, dist_array_partition);
}

void
DistArray::SetDims(const std::vector<int64_t> &dims) {
  dims_ = dims;
  for (auto partition : partitions_) {
    partition.second->SetDims(dims_);
  }
}

}
}
