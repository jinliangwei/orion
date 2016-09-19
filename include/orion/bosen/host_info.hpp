#pragma once

#include <string>
#include <glog/logging.h>
#include <type_traits>

namespace orion {
namespace bosen {

struct HostInfo {
  char ip[16];
  uint16_t port;
};

static_assert(std::is_pod<HostInfo>::value, "HostInfo must be POD!");

}
}
