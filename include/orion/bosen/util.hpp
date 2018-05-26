#pragma once
#include <stdint.h>

namespace orion {
namespace bosen {

// Get the IP address in binary form from interface name
int
GetIPFromIF(const char *ifname, uint32_t* ip);

// Get the IP address in binary form from the IPv4 numbers-and-dots notation
int
GetIPFromStr(const char *str, uint32_t *ip);
}

inline bool
is_aligned(const void* addr, size_t alignment) {
  return (reinterpret_cast<uintptr_t>(addr) % alignment == 0);
}

inline bool
is_aligned(void* addr, size_t alignment) {
  return (reinterpret_cast<uintptr_t>(addr) % alignment == 0);
}

inline const void*
get_aligned(const void* addr, size_t alignment) {
  uintptr_t addr_int = reinterpret_cast<uintptr_t>(addr);
  uintptr_t aligned = addr_int - 1 + (alignment - (addr_int - 1)  % alignment);
  return reinterpret_cast<const void*>(aligned);
}

inline void*
get_aligned(void* addr, size_t alignment) {
  uintptr_t addr_int = reinterpret_cast<uintptr_t>(addr);
  uintptr_t aligned = addr_int - 1 + (alignment - (addr_int - 1)  % alignment);
  return reinterpret_cast<void*>(aligned);
}

template<typename T>
inline T Gcd(T m, T n) {
  CHECK_GT(m, 0);
  CHECK_GT(n, 0);
  if (m < n) std::swap(m, n);
  while (n != 0) {
    T n_temp = n;
    n = m % n;
    m = n_temp;
  }
  return m;
}

}
