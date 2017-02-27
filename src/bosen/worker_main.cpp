#include <glog/logging.h>
#include <gflags/gflags.h>
#include <orion/bosen/worker.hpp>

DEFINE_int32(num_executors_per_worker, 1, "number of executors");

DEFINE_string(master_ip, "127.0.0.1",
              "IP address that the master thread listens to for "
              "incoming connections");

DEFINE_int32(master_port, 10000,
             "port that the master thread listens to for "
             "incoming connections");

DEFINE_string(worker_ip, "127.0.0.1",
              "IP address that the master thread listens to for "
              "incoming connections");

DEFINE_int32(worker_port, 11000,
             "port that the master thread listens to for "
             "incoming connections");

DEFINE_uint64(comm_buff_capacity, 1024 * 4, "communication buffer capacity");

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  orion::bosen::Config config(0, FLAGS_num_executors_per_worker, FLAGS_master_ip,
                              FLAGS_master_port, FLAGS_worker_ip, FLAGS_worker_port,
                              FLAGS_comm_buff_capacity);
  orion::bosen::Worker worker(config);
  worker.Run();
  worker.WaitUntilExit();

  return 0;
}
