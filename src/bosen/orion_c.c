#include <orion/orion_c.h>
#include <orion/bosen/worker.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <orion/utils.hpp>

extern "C" {

  Worker* worker_create(size_t comm_buff_capacity,
                        uint64_t listen_ip,
                        int32_t listen_port,
                        int32_t worker_id,
                        size_t num_workers,
                        const HostInfo* hosts) {
    auto worker = new orion::bosen::Worker(
        comm_buff_capacity,
        listen_ip,
        listen_port,
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

  GLogConfig*
  glogconfig_create(const char* progname) {
    auto glogconfig = new orion::GLogConfig(progname);
    return reinterpret_cast<GLogConfig*>(glogconfig);
  }

  bool
  glogconfig_set(GLogConfig* glogconfig, const char* key,
                      const char* value) {
    return reinterpret_cast<orion::GLogConfig*>(glogconfig)->set(key, value);
  }

  char**
  glogconfig_get_argv(GLogConfig* glogconfig) {
    return reinterpret_cast<orion::GLogConfig*>(glogconfig)->get_argv();
  }

  int
  glogconfig_get_argc(GLogConfig* glogconfig) {
    return reinterpret_cast<orion::GLogConfig*>(glogconfig)->get_argc();
  }

  void glogconfig_free(GLogConfig* glogconfig) {
    delete reinterpret_cast<orion::GLogConfig*>(glogconfig);
  }

  void glog_init(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
  }

  uint64_t get_ip(const char* ifname) {
    return orion::GetIP(ifname);
  }
}
