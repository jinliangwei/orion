#ifndef __WORKER_C_H__
#define __WORKER_C_H__

#include <stdint.h>
#include <stdlib.h>
#include <orion/bosen/host_info.h>
#include <orion/glog_config.hpp>

extern "C" {
  typedef struct Worker Worker;
  typedef struct GLogConfig GLogConfig;
  Worker* worker_create(size_t comm_buff_capacity,
                        uint64_t listen_ip,
                        int32_t listen_port,
                        int32_t worker_id,
                        size_t num_workers,
                        const HostInfo* hosts);
  int connect_to_peers(Worker* worker);
  int shut_down(Worker *worker);

  GLogConfig *glogconfig_create(char* progname);
  bool glogconfig_set(GLogConfig* glogconfig, const char* key, const char* value);
  char** glogconfig_get_argv(GLogConfig* glogconfig);
  int glogconfig_get_argc(GLogConfig* glogconfig);
  void glogconfig_free(GLogConfig* glogconfig);
  void glog_init(int argc, char* argv[]);

  uint64_t get_ip(const char* ifname);
}

#endif
