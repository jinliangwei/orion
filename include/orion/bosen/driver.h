#ifndef __DRIVER_H__
#define __DRIVER_H__

#include <stdint.h>
#include <stddef.h>
#include <julia.h>
#include "constants.h"

extern "C" {
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
      size_t comm_buff_capacity,
      size_t num_executors);

  void orion_connect_to_master();

  const uint8_t* orion_call_func_on_one(
    int32_t executor_id,
    const char *function_name,
    const TableDep *deps,
    size_t num_deps,
    int repetition,
    size_t num_iterations,
    size_t *result_size);

  jl_value_t* orion_eval_expr_on_all(
      const uint8_t* expr,
      size_t expr_size,
      int32_t module);

  void orion_create_dist_array(
      int32_t id,
      int32_t parent_type,
      int32_t map_type,
      bool flatten_results,
      size_t num_dims,
      int32_t value_type,
      const char* file_path,
      int32_t parent_id,
      int32_t init_type,
      int32_t mapper_func_module,
      const char* mapper_func_name,
      int64_t* dims,
      int32_t random_init_type);

  void orion_define_var(
      const char *var_name,
      const uint8_t *var_value,
      size_t value_size);

  void orion_space_time_repartition_dist_array(
      int32_t id,
      const char *partition_func_name);

  void orion_stop();

  bool orion_glogconfig_set(const char* key, const char* value);
  void orion_glogconfig_set_progname(const char* progname);
  void orion_glog_init();
}

#endif
