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

int
main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  LOG(INFO) << "hello, driver is started!";

  orion::bosen::DriverConfig driver_config(
      FLAGS_master_ip, FLAGS_master_port,
      FLAGS_comm_buff_capacity);

  orion::bosen::Driver driver(driver_config);
  driver.ConnectToMaster();
  orion::bosen::ByteBuffer result_buff;
  std::vector<orion::bosen::task::TableDep> read_dep, write_dep;
  orion::bosen::task::ExecuteGranularity granularity = orion::bosen::task::PER_EXECUTOR;
  size_t repetition = 1;
  auto result_type = orion::bosen::type::PrimitiveType::kInt32;
  driver.ExecuteOnOne(0, "function f()\n\tprintln(sqrt(2.0))\nend\nf()",
                      -1, read_dep, write_dep, granularity,
                      repetition, result_type, &result_buff);

  driver.Stop();
}
