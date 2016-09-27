#ifndef __HOST_INFO_H__
#define __HOST_INFO_H__
#include <stdint.h>
extern "C" struct HostInfo {
  uint64_t ip;
  uint16_t port;
};

#endif
