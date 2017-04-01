#pragma once

#include <unordered_map>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_partition.hpp>
#include <orion/bosen/type.hpp>
#include <orion/bosen/julia_evaluator.hpp>

namespace orion {
namespace bosen {

class DistArray {
 private:
  JuliaEvaluator *julia_eval_;
  const Config *kConfig;
  const type::PrimitiveType kValueType;
  const int32_t kExecutorId;
  std::unordered_map<int32_t, AbstractDistArrayPartition*> partitions_;

 public:
  DistArray(JuliaEvaluator *julia_eval, const Config& config,
            type::PrimitiveType value_type, int32_t executor_id);
  ~DistArray();
  void LoadPartitionFromTextFile(
      const std::string &file_path,
      type::PrimitiveType value_type,
      bool fatten_results,
      bool value_only,
      bool parse,
      size_t num_dims,
      const std::string &parse_func,
      const std::string &parse_func_name);
  void CreatePartitionFromParent();
  void CreatePartitoin();
 private:
  AbstractDistArrayPartition *CreatePartition();
};

DistArray::DistArray(
    JuliaEvaluator *julia_eval,
    const Config &config,
    type::PrimitiveType value_type,
    int32_t executor_id):
    julia_eval_(julia_eval),
    kConfig(&config),
    kValueType(value_type),
    kExecutorId(executor_id) { }

DistArray::~DistArray() { }

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
                new DistArrayPartition<int8_t>(julia_eval_, *kConfig));
        break;
      }
    case type::PrimitiveType::kUInt8:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint8_t>(julia_eval_, *kConfig));
        break;
      }
    case type::PrimitiveType::kInt16:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int16_t>(julia_eval_, *kConfig));
        break;
      }
    case type::PrimitiveType::kUInt16:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint16_t>(julia_eval_, *kConfig));
        break;
      }
    case type::PrimitiveType::kInt32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int32_t>(julia_eval_, *kConfig));
        break;
      }
    case type::PrimitiveType::kUInt32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<uint32_t>(julia_eval_, *kConfig));
        break;
      }
    case type::PrimitiveType::kInt64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int64_t>(julia_eval_, *kConfig));
        break;
      }
    case type::PrimitiveType::kUInt64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<int64_t>(julia_eval_, *kConfig));
        break;
      }
    case type::PrimitiveType::kFloat32:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<float>(julia_eval_, *kConfig));
        break;
      }
    case type::PrimitiveType::kFloat64:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<double>(julia_eval_, *kConfig));
        break;
      }
    case type::PrimitiveType::kString:
      {
        partition_ptr
            = static_cast<AbstractDistArrayPartition*>(
                new DistArrayPartition<const char*>(julia_eval_, *kConfig));
        break;
      }
    default:
      LOG(FATAL) << "unknown type";
  }
  return partition_ptr;
}

void
DistArray::LoadPartitionFromTextFile(
    const std::string &file_path,
    type::PrimitiveType value_type,
    bool fatten_results,
    bool value_only,
    bool parse,
    size_t num_dims,
    const std::string &parse_func,
    const std::string &parse_func_name) {
  CHECK(partitions_.empty());
}

}
}
