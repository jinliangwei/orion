#include <glog/logging.h>
#include <gflags/gflags.h>

#include <orion/bosen/driver.hpp>
#include <orion/bosen/byte_buffer.hpp>
#include <orion/bosen/type.hpp>

DEFINE_string(master_ip, "127.0.0.1",
              "IP address that the master thread listens to for "
              "incoming connections");
DEFINE_int32(master_port, 10000,
             "port that the master thread listens to for "
             "incoming connections");
DEFINE_uint64(comm_buff_capacity, 1024 * 4, "communication buffer capacity");

DEFINE_uint64(num_executors, 1, "number of executors");

DEFINE_string(lib_path, "", "orion library path");

int
main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  LOG(INFO) << "hello, driver is started!";

  orion::bosen::DriverConfig driver_config(
      FLAGS_master_ip.c_str(), FLAGS_master_port,
      FLAGS_comm_buff_capacity, FLAGS_num_executors);

  orion::bosen::Driver driver(driver_config);
  driver.ConnectToMaster();
  //auto result_type = orion::bosen::type::PrimitiveType::kFloat64;
  //driver.ExecuteCodeOnOne(0, "sqrt(2.0)", result_type, &ret);
  /*driver.CreateDistArray(
      0,
      orion::bosen::task::TEXT_FILE,
      true,
      false,
      2,
      orion::bosen::type::PrimitiveType::kFloat64,
      //      "file:///home/ubuntu/data/ml-1m/ratings.csv",
      "hdfs:///data/ml-1m/ratings.csv",
      -1,
      orion::bosen::task::EMPTY,
      orion::bosen::JuliaModule::kMain,
      "parse_line",
      nullptr);*/

  driver.Stop();
  //while(1);
}
