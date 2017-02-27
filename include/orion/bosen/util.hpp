#pragma once

namespace orion {
namespace bosen {

// Get the IP address in binary form from interface name
int
GetIPFromIF(const char *ifname, uint32_t* ip);

// Get the IP address in binary form from the IPv4 numbers-and-dots notation
int
GetIPFromStr(const char *str, uint32_t *ip);
}

}
