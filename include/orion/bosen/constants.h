#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__
#include <stdlib.h>

extern "C" {
   extern const int32_t ORION_TYPE_VOID;
   extern const int32_t ORION_TYPE_INT8;
   extern const int32_t ORION_TYPE_UINT8;
   extern const int32_t ORION_TYPE_INT16;
   extern const int32_t ORION_TYPE_UINT16;
   extern const int32_t ORION_TYPE_INT32;
   extern const int32_t ORION_TYPE_UINT32;
   extern const int32_t ORION_TYPE_INT64;
   extern const int32_t ORION_TYPE_UINT64;
   extern const int32_t ORION_TYPE_FLOAT32;
   extern const int32_t ORION_TYPE_FLOAT64;
   extern const int32_t ORION_TYPE_STRING;

  extern const int32_t ORION_TASK_DIST_ARRAY_PARENT_TYPE_TEXT_FILE;
  extern const int32_t ORION_TASK_DIST_ARRAY_PARENT_TYPE_DIST_ARRAY;
  extern const int32_t ORION_TASK_DIST_ARRAY_PARENT_TYPE_INIT;

  extern const int32_t ORION_TASK_DIST_ARRAY_INIT_TYPE_EMPTY;
  extern const int32_t ORION_TASK_DIST_ARRAY_INIT_TYPE_UNIFORM_RANDOM;
  extern const int32_t ORION_TASK_DIST_ARRAY_INIT_TYPE_NORMAL_RANDOM;

  extern const int32_t ORION_TASK_DIST_ARRAY_MAP_TYPE_NO_MAP;
  extern const int32_t ORION_TASK_DIST_ARRAY_MAP_TYPE_MAP;
  extern const int32_t ORION_TASK_DIST_ARRAY_MAP_TYPE_MAP_FIXED_KEYS;
  extern const int32_t ORION_TASK_DIST_ARRAY_MAP_TYPE_MAP_VALUES;
  extern const int32_t ORION_TASK_DIST_ARRAY_MAP_TYPE_MAP_VALUES_NEW_KEYS;

  extern const int32_t ORION_JULIA_MODULE_CORE;
  extern const int32_t ORION_JULIA_MODULE_BASE;
  extern const int32_t ORION_JULIA_MODULE_MAIN;
  extern const int32_t ORION_JULIA_MODULE_TOP;

  extern const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_NAIVE;
  extern const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_SPACE_TIME;
  extern const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_1D;
  extern const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_HASH;
  extern const int32_t ORION_DIST_ARRAY_PARTITION_SCHEME_RANGE;

  extern const int32_t ORION_DIST_ARRAY_INDEX_TYPE_NONE;
  extern const int32_t ORION_DIST_ARRAY_INDEX_TYPE_GLOBAL;
  extern const int32_t ORION_DIST_ARRAY_INDEX_TYPE_LOCAL;

  extern const int32_t ORION_FOR_LOOP_PARALLEL_SCHEME_NAIVE;
  extern const int32_t ORION_FOR_LOOP_PARALLEL_SCHEME_1D;
  extern const int32_t ORION_FOR_LOOP_PARALLEL_SCHEME_SPACE_TIME;
}
#endif
