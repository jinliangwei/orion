#include <glog/logging.h>
#include <gflags/gflags.h>
#include <orion/bosen/executor.hpp>

DEFINE_int32(num_executors, 1, "number of total executors");

DEFINE_int32(num_servers, 1, "number of total servers");

DEFINE_int32(num_executors_per_worker, 1, "number of executors per worker");
DEFINE_int32(num_servers_per_worker, 1, "number of servers per worker");

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

DEFINE_int32(worker_id, 0, "worker id");

DEFINE_int32(local_executor_index, 0, "executor id");

DEFINE_uint64(executor_thread_pool_size, 4, "thread pool size");

DEFINE_uint64(partition_size_mb, 1, "file partition size");

DEFINE_string(orion_home, "", "Orion home directory");

DEFINE_string(hdfs_name_node, "hdfs://localhost:9000",
              "name node URL of HDFS");

DEFINE_bool(is_server, false, "the worker serves as a server");

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  LOG(INFO) << "worker " << FLAGS_worker_id << " started!";

  orion::bosen::Config config(FLAGS_num_executors,
                              FLAGS_num_servers,
                              FLAGS_num_executors_per_worker,
                              FLAGS_num_servers_per_worker,
                              FLAGS_master_ip,
                              FLAGS_master_port, FLAGS_worker_ip, FLAGS_worker_port,
                              FLAGS_comm_buff_capacity, FLAGS_worker_id,
                              FLAGS_executor_thread_pool_size,
                              FLAGS_partition_size_mb,
                              FLAGS_hdfs_name_node,
                              FLAGS_orion_home);
  orion::bosen::Executor executor(config,
                                  FLAGS_local_executor_index,
                                  FLAGS_is_server);
  executor.operator()();

  return 0;
}
