#include <stdio.h>
#include <unistd.h>
#include <string.h> /* for strncpy */

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <utility>

namespace orion {
namespace bosen {
// Get the IP address in binary form from interface name
int
GetIPFromIF(const char *ifname, uint32_t* ip) {
  int fd;
  struct ifreq ifr;

  fd = socket(AF_INET, SOCK_DGRAM, 0);
  ifr.ifr_addr.sa_family = AF_INET;
  strncpy(ifr.ifr_name, ifname, IFNAMSIZ-1);
  int ret = ioctl(fd, SIOCGIFADDR, &ifr);
  close(fd);
  if (ret != 0) return ret;
  *ip = ((struct sockaddr_in *) &ifr.ifr_addr)->sin_addr.s_addr;
  return 0;
}

// Get the IP address in binary form from the IPv4 numbers-and-dots notation
int
GetIPFromStr(const char *str, uint32_t *ip) {
  struct in_addr inp;
  int ret = inet_aton(str, &inp);
  if (ret == 0) return ret;
  *ip = inp.s_addr;
  return ret;
}

}
}
