#include <orion/bosen/constants.h>
#include <orion/bosen/task.pb.h>
#include <orion/bosen/type.hpp>
#include <orion/bosen/julia_module.hpp>
#include <orion/bosen/dist_array_meta.hpp>
#include <orion/bosen/dist_array_buffer_info.hpp>

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

  const int32_t ORION_DIST_ARRAY_PARENT_TYPE_TEXT_FILE
  = static_cast<int32_t>(orion::bosen::DistArrayParentType::kTextFile);
  const int32_t ORION_DIST_ARRAY_PARENT_TYPE_DIST_ARRAY
  = static_cast<int32_t>(orion::bosen::DistArrayParentType::kDistArray);
  const int32_t ORION_DIST_ARRAY_PARENT_TYPE_INIT
  = static_cast<int32_t>(orion::bosen::DistArrayParentType::kInit);

  const int32_t ORION_DIST_ARRAY_INIT_TYPE_EMPTY
  = static_cast<int32_t>(orion::bosen::DistArrayInitType::kEmpty);
  const int32_t ORION_DIST_ARRAY_INIT_TYPE_UNIFORM_RANDOM
  = static_cast<int32_t>(orion::bosen::DistArrayInitType::kUniformRandom);
  const int32_t ORION_DIST_ARRAY_INIT_TYPE_NORMAL_RANDOM
  = static_cast<int32_t>(orion::bosen::DistArrayInitType::kNormalRandom);
  const int32_t ORION_DIST_ARRAY_INIT_TYPE_FILL
  = static_cast<int32_t>(orion::bosen::DistArrayInitType::kFill);

  const int32_t ORION_JULIA_MODULE_CORE = static_cast<int32_t>(orion::bosen::JuliaModule::kCore);
  const int32_t ORION_JULIA_MODULE_BASE = static_cast<int32_t>(orion::bosen::JuliaModule::kBase);
  const int32_t ORION_JULIA_MODULE_MAIN = static_cast<int32_t>(orion::bosen::JuliaModule::kMain);
  const int32_t ORION_JULIA_MODULE_TOP = static_cast<int32_t>(orion::bosen::JuliaModule::kTop);

  const int32_t ORION_DIST_ARRAY_MAP_TYPE_NO_MAP
  = static_cast<int32_t>(orion::bosen::DistArrayMapType::kNoMap);
  const int32_t ORION_DIST_ARRAY_MAP_TYPE_MAP
  = static_cast<int32_t>(orion::bosen::DistArrayMapType::kMap);
  const int32_t ORION_DIST_ARRAY_MAP_TYPE_MAP_FIXED_KEYS
  = static_cast<int32_t>(orion::bosen::DistArrayMapType::kMapFixedKeys);
  const int32_t ORION_DIST_ARRAY_MAP_TYPE_MAP_VALUES
  = static_cast<int32_t>(orion::bosen::DistArrayMapType::kMapValues);
  const int32_t ORION_DIST_ARRAY_MAP_TYPE_MAP_VALUES_NEW_KEYS
  = static_cast<int32_t>(orion::bosen::DistArrayMapType::kMapValuesNewKeys);

  const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_NAIVE
  = static_cast<int32_t>(orion::bosen::DistArrayPartitionScheme::kNaive);
  const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_SPACE_TIME
  = static_cast<int32_t>(orion::bosen::DistArrayPartitionScheme::kSpaceTime);
  const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_1D
  = static_cast<int32_t>(orion::bosen::DistArrayPartitionScheme::k1D);
  const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_HASH_SERVER
  = static_cast<int32_t>(orion::bosen::DistArrayPartitionScheme::kHashServer);
  const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_HASH_EXECUTOR
  = static_cast<int32_t>(orion::bosen::DistArrayPartitionScheme::kHashExecutor);
  const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_RANGE
  = static_cast<int32_t>(orion::bosen::DistArrayPartitionScheme::kRange);

  const int32_t ORION_DIST_ARRAY_INDEX_TYPE_NONE
  = static_cast<int32_t>(orion::bosen::DistArrayIndexType::kNone);
  const int32_t ORION_DIST_ARRAY_INDEX_TYPE_RANGE
  = static_cast<int32_t>(orion::bosen::DistArrayIndexType::kRange);

  const int32_t ORION_FOR_LOOP_PARALLEL_SCHEME_1D
  = static_cast<int32_t>(orion::bosen::ForLoopParallelScheme::k1D);
  const int32_t ORION_FOR_LOOP_PARALLEL_SCHEME_SPACE_TIME
  = static_cast<int32_t>(orion::bosen::ForLoopParallelScheme::kSpaceTime);

  const int32_t ORION_DIST_ARRAY_BUFFER_DELAY_MODE_DEFAULT
  = static_cast<int32_t>(orion::bosen::DistArrayBufferDelayMode::kDefault);
  const int32_t ORION_DIST_ARRAY_BUFFER_DELAY_MODE_MAX_DELAY
  = static_cast<int32_t>(orion::bosen::DistArrayBufferDelayMode::kMaxDelay);
  const int32_t ORION_DIST_ARRAY_BUFFER_DELAY_MODE_AUTO
  = static_cast<int32_t>(orion::bosen::DistArrayBufferDelayMode::kAuto);

}
