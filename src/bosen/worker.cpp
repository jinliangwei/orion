#include <gflags/gflags.h>

namespace orion {
namespace bosen {

DEFINE_string(worker_driver_ip, "127.0.0.1",
              "IP address that the driver thread listens to for "
              "incoming connections");

DEFINE_int32(worker_driver_port, 10000,
             "port that the driver thread listens to for "
             "incoming connections");

DEFINE_string(worker_ip, "127.0.0.1",
              "IP address that the executor thread listens to for "
              "incoming connections");

DEFINE_int32(worker_port, 10001,
             "port that the thread thread listens to for "
             "incoming connections");

DEFINE_int32(worker_num_executors_per_worker, 1,
             "number of executors");

DEFINE_int32(worker_id, 0, "worker id");

}
}
