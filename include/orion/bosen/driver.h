#ifndef __DRIVER_H__
#define __DRIVER_H__

#include <stdint.h>
#include <stddef.h>

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

   extern const int32_t ORION_TASK_TABLE_DEP_TYPE_PIPELINED;
   extern const int32_t ORION_TASK_TABLE_DEP_TYPE_RANDOM_ACCESS;

   extern const int32_t ORION_TASK_READWRITE_READ_ONLY;
   extern const int32_t ORION_TASK_READWRITE_WRITE_ONLY;
   extern const int32_t ORION_TASK_READWRITE_READ_WRITE;

   extern const int32_t ORION_TASK_REPETITION_ONE_PARTITION;
   extern const int32_t ORION_TASK_REPETITION_ALL_LOCAL_PARTITIONS;
   extern const int32_t ORION_TASK_REPETITION_ALL_PARTITIONS;

   extern const int32_t ORION_TASK_PARTITION_SCHEME_STATIC;
   extern const int32_t ORION_TASK_PARTITION_SCHEME_DYNAMIC;
   extern const int32_t ORION_TASK_PARTITION_SCHEME_RANDOM;

   extern const int32_t ORION_TASK_BASETABLE_TYPE_VIRTUAL;
   extern const int32_t ORION_TASK_BASETABLE_TYPE_CONCRETE;

  typedef struct VirtualBaseTable {
    size_t size;
  } VirtualBaseTable;

  typedef struct ConcreteBaseTable {
    int32_t tbale_id;
    int partition_scheme;
    size_t partition_size;
  } ConcreteBaseTable;

  typedef struct TableDep {
    int32_t table_id;
    int dep_type;
    int read_write;
    const char* function_compute_dep;
  } TableDep;

  void orion_init(
      const char *master_ip,
      uint16_t master_port,
      size_t comm_buff_capacity);

  void orion_connect_to_master();

  void orion_execute_code_on_one(
    int32_t executor_id,
    const char* code,
    int result_type,
    void *result_buff);

  void orion_call_func_on_one(
    int32_t executor_id,
    const char *function_name,
    const TableDep *deps,
    size_t num_deps,
    int repetition,
    size_t num_iterations,
    int result_type,
    void *result_buff);

  void orion_stop();

  typedef struct GLogConfig OrionGLogConfig;
  GLogConfig *orion_glogconfig_create(const char* progname);
  bool orion_glogconfig_set(GLogConfig* glogconfig, const char* key,
                            const char* value);
  void orion_glogconfig_free(GLogConfig* glogconfig);
  void orion_glog_init(GLogConfig* glogconfig);
}

#endif
