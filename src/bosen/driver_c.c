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
      size_t value_type_size) {
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
        value_type_size);
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
      int32_t index_type) {
    driver->RepartitionDistArray(id, partition_func_name,
                                 partition_scheme,
                                 index_type);
  }

  void orion_set_dist_array_buffer(
      int32_t dist_array_id,
      int32_t *buffer_ids,
      size_t num_buffers) {
    driver->SetDistArrayBuffer(
        dist_array_id,
        buffer_ids,
        num_buffers);
  }

  void orion_exec_for_loop(
      int32_t iteration_space_id,
      int32_t parallel_scheme,
      const int32_t *space_partitioned_dist_array_ids,
      size_t num_space_partitioned_dist_arrays,
      const int32_t *time_partitioned_dist_array_ids,
      size_t num_time_partitioned_dist_arrays,
      const int32_t *global_indexed_dist_array_ids,
      size_t num_global_indexed_dist_arrays,
      const int32_t *buffered_dist_array_ids,
      size_t num_buffered_dist_arrays,
      const int32_t *dist_array_buffer_ids,
      const size_t *num_buffers_each_dist_array,
      const char *loop_batch_func_name,
      const char *prefetch_batch_func_name,
      bool is_ordered) {
    driver->ExecForLoop(
        iteration_space_id,
        parallel_scheme,
        space_partitioned_dist_array_ids,
        num_space_partitioned_dist_arrays,
        time_partitioned_dist_array_ids,
        num_time_partitioned_dist_arrays,
        global_indexed_dist_array_ids,
        num_global_indexed_dist_arrays,
        buffered_dist_array_ids,
        num_buffered_dist_arrays,
        dist_array_buffer_ids,
        num_buffers_each_dist_array,
        loop_batch_func_name,
        prefetch_batch_func_name,
        is_ordered);
  }

  jl_value_t* orion_get_accumulator_value(
      const char *symbol,
      const char *combiner) {
    return driver->GetAccumulatorValue(
        symbol,
        combiner);
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
