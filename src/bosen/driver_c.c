#include <glog/logging.h>
#include <iostream>

#include <orion/bosen/driver.h>
#include <orion/bosen/driver.hpp>
#include <orion/glog_config.hpp>

extern "C" {

  orion::bosen::Driver *driver = nullptr;
  orion::GLogConfig glog_config("julia_driver");

  void orion_helloworld() {
    std::cout << "helloworld" << std::endl;
  }

  void orion_init(
      const char *master_ip,
      uint16_t master_port,
      size_t comm_buff_capacity) {
    orion::bosen::DriverConfig driver_config(
        master_ip, master_port,
        comm_buff_capacity);
    driver = new orion::bosen::Driver(driver_config);
    driver->ConnectToMaster();
  }

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
      size_t init_value_size,
      const char* key_func_name) {
    driver->CreateDistArray(
        id,
        parent_type,
        map_type,
        partition_scheme,
        flatten_results,
        num_dims,
        value_type,
        file_path,
        parent_id,
        init_type,
        static_cast<orion::bosen::JuliaModule>(map_func_module),
        map_func_name,
        dims,
        random_init_type,
        is_dense,
        symbol,
        value_type_bytes,
        value_type_size,
        init_value_bytes,
        init_value_size,
        key_func_name);
  }

  void orion_create_dist_array_buffer(
      int32_t id,
      int64_t *dims,
      size_t num_dims,
      bool is_dense,
      int32_t value_type,
      jl_value_t *init_value,
      const char* symbol,
      const uint8_t* value_type_bytes,
      size_t value_type_size) {
    driver->CreateDistArrayBuffer(id,
                                  dims,
                                  num_dims,
                                  is_dense,
                                  value_type,
                                  init_value,
                                  symbol,
                                  value_type_bytes,
                                  value_type_size);
  }

  void orion_delete_dist_array(
      int32_t id) {
    driver->DeleteDistArray(id);
  }

  jl_value_t* orion_eval_expr_on_all(
      const uint8_t* expr,
      size_t expr_size,
      int32_t module) {
    return driver->EvalExprOnAll(
        expr,
        expr_size,
        static_cast<orion::bosen::JuliaModule>(module));
  }

  void orion_repartition_dist_array(
      int32_t id,
      const char *partition_func_name,
      int32_t partition_scheme,
      int32_t index_type,
      bool contiguous_partitions,
      size_t *partition_dims,
      size_t num_partition_dims) {
    driver->RepartitionDistArray(id, partition_func_name,
                                 partition_scheme,
                                 index_type,
                                 contiguous_partitions,
                                 partition_dims,
                                 num_partition_dims);
  }

  void orion_update_dist_array_index(
      int32_t id,
      int32_t index_type) {
    driver->UpdateDistArrayIndex(id, index_type);
  }

  void orion_set_dist_array_buffer_info(
      int32_t dist_array_buffer_id,
      int32_t dist_array_id,
      const char *apply_buffer_func_name,
      const int32_t *helper_buffer_ids,
      size_t num_helper_buffers,
      const int32_t *helper_dist_array_ids,
      size_t num_helper_dist_arrays,
      int32_t dist_array_buffer_delay_mode,
      size_t max_delay) {
    driver->SetDistArrayBufferInfo(
        dist_array_buffer_id,
        dist_array_id,
        apply_buffer_func_name,
        helper_buffer_ids,
        num_helper_buffers,
        helper_dist_array_ids,
        num_helper_dist_arrays,
        dist_array_buffer_delay_mode,
        max_delay);
  }

  void orion_delete_dist_array_buffer_info(
      int32_t dist_array_buffer_id) {
    driver->DeleteDistArrayBufferInfo(dist_array_buffer_id);
  }

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
      bool is_repeated) {
    driver->ExecForLoop(
        exec_for_loop_id,
        iteration_space_id,
        parallel_scheme,
        space_partitioned_dist_array_ids,
        num_space_partitioned_dist_arrays,
        time_partitioned_dist_array_ids,
        num_time_partitioned_dist_arrays,
        global_indexed_dist_array_ids,
        num_global_indexed_dist_arrays,
        dist_array_buffer_ids,
        num_dist_array_buffers,
        written_dist_array_ids,
        num_written_dist_array_ids,
        accessed_dist_array_ids,
        num_accessed_dist_arrays,
        global_read_only_var_vals,
        accumulator_var_syms,
        num_accumulator_var_syms,
        loop_batch_func_name,
        prefetch_batch_func_name,
        is_ordered,
        is_repeated);
  }

  jl_value_t* orion_get_accumulator_value(
      const char *symbol,
      const char *combiner) {
    return driver->GetAccumulatorValue(
        symbol,
        combiner);
  }

  void orion_random_remap_partial_keys(
      int32_t dist_array_id,
      size_t *dim_indices,
      size_t num_dim_indices) {
    driver->RandomRemapPartialKeys(dist_array_id,
                                   dim_indices,
                                   num_dim_indices);
  }

  jl_value_t* orion_compute_histogram(
      int32_t dist_array_id,
      size_t dim_index,
      size_t num_bins) {
    return driver->ComputeHistogram(dist_array_id, dim_index, num_bins);
  }

  void orion_save_as_text_file(
      int32_t dist_array_id,
      const char* to_string_func_name,
      const char* file_path) {
    driver->SaveAsTextFile(dist_array_id, to_string_func_name, file_path);
  }

  void orion_stop() {
    driver->Stop();
    delete driver;
  }

  bool
  orion_glogconfig_set(const char* key, const char* value) {
    return glog_config.set(key, value);
  }

  void
  orion_glogconfig_set_progname(const char* progname) {
    glog_config.set_progname(progname);
  }

  void
  orion_glog_init() {
    int argc = glog_config.get_argc();
    char** argv = glog_config.get_argv();
    google::ParseCommandLineFlags(&argc, &argv, false);
    std::cout << argv[0] << std::endl;
    google::InitGoogleLogging(argv[0]);
  }
}
