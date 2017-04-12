#include <glog/logging.h>
#include <iostream>

#include <orion/bosen/driver.h>
#include <orion/bosen/driver.hpp>
#include <orion/glog_config.hpp>
#include <orion/bosen/task.pb.h>

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

  const int32_t ORION_JULIA_MODULE_CORE = static_cast<int32_t>(orion::bosen::JuliaModule::kCore);
  const int32_t ORION_JULIA_MODULE_BASE = static_cast<int32_t>(orion::bosen::JuliaModule::kBase);
  const int32_t ORION_JULIA_MODULE_MAIN = static_cast<int32_t>(orion::bosen::JuliaModule::kMain);
  const int32_t ORION_JULIA_MODULE_TOP = static_cast<int32_t>(orion::bosen::JuliaModule::kTop);

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
      bool map,
      bool flatten_results,
      size_t num_dims,
      int32_t value_type,
      const char* file_path,
      int32_t parent_id,
      int32_t init_type,
      int32_t mapper_func_module,
      const char* mapper_func_name) {
    driver->CreateDistArray(
        id,
        static_cast<orion::bosen::task::DistArrayParentType>(parent_type),
        map,
        flatten_results,
        num_dims,
        static_cast<orion::bosen::type::PrimitiveType>(value_type),
        file_path,
        parent_id,
        static_cast<orion::bosen::task::DistArrayInitType>(init_type),
        static_cast<orion::bosen::JuliaModule>(mapper_func_module),
        mapper_func_name);
  }

  void orion_eval_expr_on_all(
      const uint8_t* expr,
      size_t expr_size,
      int32_t result_type,
      void *result_buff) {
    driver->EvalExprOnAll(
        expr,
        expr_size,
        static_cast<orion::bosen::type::PrimitiveType>(result_type),
        result_buff);
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
