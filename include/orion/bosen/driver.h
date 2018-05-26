#ifndef __DRIVER_H__
#define __DRIVER_H__

#include <stdint.h>
#include <stddef.h>
#include <julia.h>
#include "constants.h"

extern "C" {
  void orion_init(
      const char *master_ip,
      uint16_t master_port,
      size_t comm_buff_capacity);

  void orion_connect_to_master();

  jl_value_t* orion_eval_expr_on_all(
      const uint8_t* expr,
      size_t expr_size,
      int32_t module);

  void orion_create_dist_array(
      int32_t id,
      int32_t parent_type,
      int32_t map_type,
      int32_t partition_scheme,
      bool flatten_results,
      size_t num_dims,
      int32_t value_type,
      const char* file_path,
      int32_t parent_id,
      int32_t init_type,
      int32_t map_func_module,
      const char* map_func_name,
      int64_t* dims,
      int32_t random_init_type,
      bool is_dense,
      const char* symbol,
      const uint8_t* value_type_bytes,
      size_t value_type_size,
      const uint8_t* init_value_bytes,
      size_t init_value_size);

  void orion_create_dist_array_buffer(
      int32_t id,
      int64_t *dims,
      size_t num_dims,
      bool is_dense,
      int32_t value_type,
      jl_value_t *init_value,
      const char* symbol,
      const uint8_t* value_type_bytes,
      size_t value_type_size);

  void orion_delete_dist_array(
      int32_t id);

  void orion_repartition_dist_array(
      int32_t id,
      const char *partition_func_name,
      int32_t partition_scheme,
      int32_t index_type,
      bool contiguous_partitions);

  void orion_update_dist_array_index(
      int32_t id,
      int32_t index_type);

  void orion_set_dist_array_buffer_info(
      int32_t dist_array_buffer_id,
      int32_t dist_array_id,
      const char *apply_buffer_func_name,
      const int32_t *helper_buffer_ids,
      size_t num_helper_buffers,
      const int32_t *helper_dist_array_ids,
      size_t num_helper_dist_arrays,
      int32_t dist_array_buffer_delay_mode,
      size_t max_delay);

  void orion_delete_dist_array_buffer_info(
      int32_t dist_array_buffer_id);

  void orion_exec_for_loop(
      int32_t exec_for_loop_id,
      int32_t iteration_space_id,
      int32_t parallel_scheme,
      const int32_t *space_partitioned_dist_array_ids,
      size_t num_space_partitioned_dist_arrays,
      const int32_t *time_partitioned_dist_array_ids,
      size_t num_time_partitioned_dist_arrays,
      const int32_t *global_indexed_dist_array_ids,
      size_t num_global_indexed_dist_arrays,
      const int32_t *dist_array_buffer_ids,
      size_t num_dist_array_buffers,
      const int32_t *written_dist_array_ids,
      size_t num_written_dist_array_ids,
      const int32_t *accessed_dist_array_ids,
      size_t num_accessed_dist_arrays,
      jl_value_t *global_read_only_var_vals,
      const char **accumulator_var_syms,
      size_t num_accumulator_var_syms,
      const char *loop_batch_func_name,
      const char *prefetch_batch_func_name,
      bool is_ordered,
      bool is_repeated);

  jl_value_t* orion_get_accumulator_value(
      const char *symbol,
      const char *combiner);

  void orion_stop();

  bool orion_glogconfig_set(const char* key, const char* value);
  void orion_glogconfig_set_progname(const char* progname);
  void orion_glog_init();
}

#endif
