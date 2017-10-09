#include <orion/bosen/constants.h>
#include <orion/bosen/task.pb.h>
#include <orion/bosen/type.hpp>
#include <orion/bosen/julia_module.hpp>
#include <orion/bosen/dist_array_meta.hpp>

extern "C" {
  const int32_t ORION_TYPE_VOID = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kVoid);
  const int32_t ORION_TYPE_INT8 = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kInt8);
  const int32_t ORION_TYPE_UINT8 = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kUInt8);
  const int32_t ORION_TYPE_INT16 = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kInt16);
  const int32_t ORION_TYPE_UINT16 = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kUInt16);
  const int32_t ORION_TYPE_INT32 = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kInt32);
  const int32_t ORION_TYPE_UINT32 = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kUInt32);
  const int32_t ORION_TYPE_INT64 = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kInt64);
  const int32_t ORION_TYPE_UINT64 = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kUInt64);
  const int32_t ORION_TYPE_FLOAT32 = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kFloat32);
  const int32_t ORION_TYPE_FLOAT64 = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kFloat64);
  const int32_t ORION_TYPE_STRING = static_cast<int32_t>(orion::bosen::type::PrimitiveType::kString);

  const int32_t ORION_TASK_TABLE_DEP_TYPE_PIPELINED = static_cast<int32_t>(orion::bosen::task::PIPELINED);
  const int32_t ORION_TASK_TABLE_DEP_TYPE_RANDOM_ACCESS = static_cast<int32_t>(orion::bosen::task::RANDOM_ACCESS);

  const int32_t ORION_TASK_READWRITE_READ_ONLY = static_cast<int32_t>(orion::bosen::task::READ_ONLY);
  const int32_t ORION_TASK_READWRITE_WRITE_ONLY = static_cast<int32_t>(orion::bosen::task::WRITE_ONLY);
  const int32_t ORION_TASK_READWRITE_READ_WRITE = static_cast<int32_t>(orion::bosen::task::READ_WRITE);

  const int32_t ORION_TASK_REPETITION_ONE_PARTITION = static_cast<int32_t>(orion::bosen::task::ONE_PARTITION);
  const int32_t ORION_TASK_REPETITION_ALL_LOCAL_PARTITIONS = static_cast<int32_t>(orion::bosen::task::ALL_LOCAL_PARTITIONS);
  const int32_t ORION_TASK_REPETITION_ALL_PARTITIONS = static_cast<int32_t>(orion::bosen::task::ALL_PARTITIONS);

  const int32_t ORION_TASK_PARTITION_SCHEME_STATIC = static_cast<int32_t>(orion::bosen::task::STATIC);
  const int32_t ORION_TASK_PARTITION_SCHEME_DYNAMIC = static_cast<int32_t>(orion::bosen::task::DYNAMIC);
  const int32_t ORION_TASK_PARTITION_SCHEME_RANDOM = static_cast<int32_t>(orion::bosen::task::RANDOM);

  const int32_t ORION_TASK_BASETABLE_TYPE_VIRTUAL = static_cast<int32_t>(orion::bosen::task::VIRTUAL);
  const int32_t ORION_TASK_BASETABLE_TYPE_CONCRETE = static_cast<int32_t>(orion::bosen::task::CONCRETE);

  const int32_t ORION_TASK_DIST_ARRAY_PARENT_TYPE_TEXT_FILE = static_cast<int32_t>(orion::bosen::task::TEXT_FILE);
  const int32_t ORION_TASK_DIST_ARRAY_PARENT_TYPE_DIST_ARRAY = static_cast<int32_t>(orion::bosen::task::DIST_ARRAY);
  const int32_t ORION_TASK_DIST_ARRAY_PARENT_TYPE_INIT = static_cast<int32_t>(orion::bosen::task::INIT);

  const int32_t ORION_TASK_DIST_ARRAY_INIT_TYPE_EMPTY = static_cast<int32_t>(orion::bosen::task::EMPTY);
  const int32_t ORION_TASK_DIST_ARRAY_INIT_TYPE_UNIFORM_RANDOM = static_cast<int32_t>(orion::bosen::task::UNIFORM_RANDOM);
  const int32_t ORION_TASK_DIST_ARRAY_INIT_TYPE_NORMAL_RANDOM = static_cast<int32_t>(orion::bosen::task::NORMAL_RANDOM);

  const int32_t ORION_JULIA_MODULE_CORE = static_cast<int32_t>(orion::bosen::JuliaModule::kCore);
  const int32_t ORION_JULIA_MODULE_BASE = static_cast<int32_t>(orion::bosen::JuliaModule::kBase);
  const int32_t ORION_JULIA_MODULE_MAIN = static_cast<int32_t>(orion::bosen::JuliaModule::kMain);
  const int32_t ORION_JULIA_MODULE_TOP = static_cast<int32_t>(orion::bosen::JuliaModule::kTop);

  const int32_t ORION_TASK_DIST_ARRAY_MAP_TYPE_NO_MAP = static_cast<int32_t>(orion::bosen::task::NO_MAP);
  const int32_t ORION_TASK_DIST_ARRAY_MAP_TYPE_MAP = static_cast<int32_t>(orion::bosen::task::MAP);
  const int32_t ORION_TASK_DIST_ARRAY_MAP_TYPE_MAP_FIXED_KEYS = static_cast<int32_t>(orion::bosen::task::MAP_FIXED_KEYS);
  const int32_t ORION_TASK_DIST_ARRAY_MAP_TYPE_MAP_VALUES = static_cast<int32_t>(orion::bosen::task::MAP_VALUES);
  const int32_t ORION_TASK_DIST_ARRAY_MAP_TYPE_MAP_VALUES_NEW_KEYS = static_cast<int32_t>(orion::bosen::task::MAP_VALUES_NEW_KEYS);

  const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_NAIVE
  = static_cast<int32_t>(orion::bosen::DistArrayPartitionScheme::kNaive);
  const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_SPACE_TIME
  = static_cast<int32_t>(orion::bosen::DistArrayPartitionScheme::kSpaceTime);
  const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_1D
  = static_cast<int32_t>(orion::bosen::DistArrayPartitionScheme::k1D);
  const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_HASH
  = static_cast<int32_t>(orion::bosen::DistArrayPartitionScheme::kHash);
  const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_RANGE
  = static_cast<int32_t>(orion::bosen::DistArrayPartitionScheme::kRange);

  const int32_t ORION_DIST_ARRAY_INDEX_TYPE_NONE
  = static_cast<int32_t>(orion::bosen::DistArrayIndexType::kNone);
  const int32_t ORION_DIST_ARRAY_INDEX_TYPE_GLOBAL
  = static_cast<int32_t>(orion::bosen::DistArrayIndexType::kGlobal);
  const int32_t ORION_DIST_ARRAY_INDEX_TYPE_LOCAL
  = static_cast<int32_t>(orion::bosen::DistArrayIndexType::kLocal);
}
