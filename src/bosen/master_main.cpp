#include <glog/logging.h>
#include <gflags/gflags.h>
#include <orion/bosen/config.hpp>
#include <orion/bosen/master.hpp>

DEFINE_int32(num_executors, 1, "number of executors");

DEFINE_string(master_ip, "127.0.0.1",
              "IP address that the master thread listens to for "
              "incoming connections");

DEFINE_int32(master_port, 10000,
             "port that the master thread listens to for "
             "incoming connections");

DEFINE_uint64(comm_buff_capacity, 1024 * 4, "communication buffer capacity");

int
main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  LOG(INFO) << "hello!";
  orion::bosen::Config config(FLAGS_num_executors, 0, FLAGS_master_ip,
                              FLAGS_master_port, "", 0, FLAGS_comm_buff_capacity,
                              0, 0, 0, "", "");
  orion::bosen::Master master(config);
  master.Run();
  master.WaitUntilExit();

  return 0;
}
