#pragma once

#include <string>
#include <glog/logging.h>

namespace orion {
namespace bosen {

struct Config {
 public:
  const size_t kNumExecutors;
  const size_t kNumExecutorsPerWorker;
  const std::string kMasterIp;
  const uint16_t kMasterPort;
  const std::string kWorkerIp;
  const uint16_t kWorkerPort;
  const size_t kCommBuffCapacity {1024 * 4};
  const int32_t kWorkerId;

  Config(size_t num_executors,
         size_t num_executors_per_worker,
         std::string master_ip,
         uint16_t master_port,
         std::string worker_ip,
         uint16_t worker_port,
         uint64_t comm_buff_capacity,
         int32_t worker_id):
      kNumExecutors(num_executors),
      kNumExecutorsPerWorker(num_executors_per_worker),
      kMasterIp(master_ip),
      kMasterPort(master_port),
      kWorkerIp(worker_ip),
      kWorkerPort(worker_port),
      kCommBuffCapacity(comm_buff_capacity),
      kWorkerId(worker_id) { }
};

}
}
