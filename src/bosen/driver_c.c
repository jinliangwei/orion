#include <glog/logging.h>
#include <iostream>

#include <orion/bosen/driver.h>
#include <orion/bosen/driver.hpp>
#include <orion/glog_config.hpp>

extern "C" {
  const int32_t ORION_TYPE_VOID = 0;
  const int32_t ORION_TYPE_INT8 = 1;
  const int32_t ORION_TYPE_UINT8 = 2;
  const int32_t ORION_TYPE_INT16 = 3;
  const int32_t ORION_TYPE_UINT16 = 4;
  const int32_t ORION_TYPE_INT32 = 5;
  const int32_t ORION_TYPE_UINT32 = 6;
  const int32_t ORION_TYPE_INT64 = 7;
  const int32_t ORION_TYPE_UINT64 = 8;
  const int32_t ORION_TYPE_FLOAT32 = 9;
  const int32_t ORION_TYPE_FLOAT64 = 10;
  const int32_t ORION_TYPE_STRING = 11;

  const int32_t ORION_TASK_TABLE_DEP_TYPE_PIPELINED = 1;
  const int32_t ORION_TASK_TABLE_DEP_TYPE_RANDOM_ACCESS = 2;

  const int32_t ORION_TASK_READWRITE_READ_ONLY = 1;
  const int32_t ORION_TASK_READWRITE_WRITE_ONLY = 2;
  const int32_t ORION_TASK_READWRITE_READ_WRITE = 3;

  const int32_t ORION_TASK_REPETITION_ONE_PARTITION = 1;
  const int32_t ORION_TASK_REPETITION_ALL_LOCAL_PARTITIONS = 2;
  const int32_t ORION_TASK_REPETITION_ALL_PARTITIONS = 3;

  const int32_t ORION_TASK_PARTITION_SCHEME_ = 1;
  const int32_t ORION_TASK_PARTITION_SCHEME_DYNAMIC = 2;
  const int32_t ORION_TASK_PARTITION_SCHEME_RANDOM = 3;

  const int32_t ORION_TASK_BASETABLE_TYPE_VIRTUAL = 1;
  const int32_t ORION_TASK_BASETABLE_TYPE_CONCRETE = 2;

  const int32_t ORION_TASK_DIST_ARRAY_PARENT_TYPE_TEXT_FILE = 1;
  const int32_t ORION_TASK_DIST_ARRAY_PARENT_TYPE_DIST_ARRAY = 2;
  const int32_t ORION_TASK_DIST_ARRAY_PARENT_TYPE_INIT = 3;

  const int32_t ORION_TASK_DIST_ARRAY_INIT_TYPE_EMPTY = 1;
  const int32_t ORION_TASK_DIST_ARRAY_INIT_TYPE_UNIFORM_RANDOM = 2;

  orion::bosen::Driver *driver = nullptr;

  void orion_helloworld() {
    std::cout << "helloworld" << std::endl;
  }

  void orion_init(
      const char *master_ip,
      uint16_t master_port,
      size_t comm_buff_capacity) {
    orion::bosen::DriverConfig driver_config(
        std::string(master_ip), master_port,
        comm_buff_capacity);
    driver = new orion::bosen::Driver(driver_config);
    driver->ConnectToMaster();
  }

  void orion_execute_code_on_one(
    int32_t executor_id,
    const char* code,
    int result_type,
    void *result_buff) {
    auto cast_result_type = static_cast<orion::bosen::type::PrimitiveType>(result_type);
    driver->ExecuteCodeOnOne(executor_id, code,
                             cast_result_type, result_buff);
  }

  void orion_call_func_on_one(
      int32_t executor_id,
      const char *function_name,
      const TableDep *deps,
      size_t num_deps,
      int repetition,
      size_t num_iterations,
      int result_type,
      void *result_buff) { }

  void orion_create_dist_array(
      int32_t id,
      int32_t parent_type,
      bool flatten_results,
      bool value_only,
      bool parse,
      size_t num_dims,
      int value_type,
      const char* file_path,
      int32_t parent_id,
      int32_t init_type,
      const char* parser_func,
      const char* parser_func_name) {
    driver->CreateDistArray(
        id,
        static_cast<orion::bosen::task::DistArrayParentType>(parent_type),
        flatten_results,
        value_only,
        parse,
        num_dims,
        static_cast<orion::bosen::type::PrimitiveType>(value_type),
        (file_path != nullptr) ? std::string(file_path) : std::string(),
        parent_id,
        static_cast<orion::bosen::task::DistArrayInitType>(init_type),
        (parser_func != nullptr) ? std::string(parser_func) : std::string(),
        (parser_func_name != nullptr) ? std::string(parser_func_name) : std::string());
  }


  void orion_stop() {
    driver->Stop();
    delete driver;
  }

  OrionGLogConfig*
  orion_glogconfig_create(const char* progname) {
    auto glogconfig = new orion::GLogConfig(progname);
    return reinterpret_cast<GLogConfig*>(glogconfig);
  }

  bool
  orion_glogconfig_set(
      OrionGLogConfig* glogconfig, const char* key,
      const char* value) {
    return reinterpret_cast<orion::GLogConfig*>(glogconfig)->set(key, value);
  }

  void
  orion_glogconfig_free(OrionGLogConfig* glogconfig) {
    delete reinterpret_cast<orion::GLogConfig*>(glogconfig);
  }

  void
  orion_glog_init(OrionGLogConfig* glogconfig) {
    int argc = 0;
    char **argv = nullptr;
    if (glogconfig == NULL) {
      orion::GLogConfig config = orion::GLogConfig("julia_driver");
      argc = config.get_argc();
      argv = config.get_argv();
    } else {
      auto *cast_glogconfig = reinterpret_cast<orion::GLogConfig*>(glogconfig);
      argc = cast_glogconfig->get_argc();
      argv = cast_glogconfig->get_argv();
    }
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
  }
}
