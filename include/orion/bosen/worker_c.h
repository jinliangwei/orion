#ifndef __WORKER_C_H__
#define __WORKER_C_H__

#include <stdint.h>
#include <stdlib.h>
#include <orion/bosen/host_info.h>

extern "C" {
  typedef struct Worker Worker;

  Worker* create_worker(size_t comm_buff_capacity,
                    uint64_t listen_ip,
                    int32_t listen_port,
                    size_t num_executors_per_worker,
                    int32_t worker_id,
                    size_t num_workers,
                    const HostInfo* hosts);
  int connect_to_peers(Worker* worker);
  int shut_down(Worker *worker);
}

#endif
