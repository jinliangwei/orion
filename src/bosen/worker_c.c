#include <orion/bosen/worker_c.h>
#include <orion/bosen/worker.hpp>

extern "C" {

  Worker* create_worker(size_t comm_buff_capacity,
                    uint64_t listen_ip,
                    int32_t listen_port,
                    size_t num_executors_per_worker,
                    int32_t worker_id,
                    size_t num_workers,
                    const HostInfo* hosts) {
    auto worker = new orion::bosen::Worker(
        comm_buff_capacity,
        listen_ip,
        listen_port,
        num_executors_per_worker,
        worker_id,
        num_workers,
        hosts);

    return reinterpret_cast<Worker*>(worker);
  }

  int connect_to_peers(Worker* worker) {
    return 0;
  }

  int shut_down(Worker *worker) {
    return 0;
  }

}
