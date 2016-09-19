#include <gflags/gflags.h>

namespace orion {
namespace bosen {

DEFINE_string(driver_ip, "127.0.0.1",
              "IP address that the driver thread listens to for "
              "incoming connections");

DEFINE_int32(driver_port, 10000,
             "port that the driver thread listens to for "
             "incoming connections");

DEFINE_int32(driver_num_executors, 1,
             "number of executor nodes");

DEFINE_uint64(comm_buff_capacity, 1024,
              "comm buff capacity");

}
}
