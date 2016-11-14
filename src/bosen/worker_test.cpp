#include <iostream>
#include <orion/orion_c.h>
#include <glog/logging.h>
#include <netinet/in.h>
#include <stdint.h>
#include <sys/socket.h>
#include <arpa/inet.h>

int main(int argc, char *argv[]) {
  std::cout << "hello world!" << std::endl;
  GLogConfig *glogconfig = glogconfig_create(argv[0]);
  glogconfig_set(glogconfig, "minloglevel", "0");
  glogconfig_set(glogconfig, "logtostderr", "false");
  glogconfig_set(glogconfig, "alsologtostderr", "true");
  glogconfig_set(glogconfig, "log_dir", "/home/jinliang/orion.git/");
  glog_init(glogconfig_get_argc(glogconfig), glogconfig_get_argv(glogconfig));
  glogconfig_free(glogconfig);

  LOG(INFO) << "hello world from glog INFO";
  LOG(WARNING) << "hello world from glog WARNING";

  in_addr ip;
  ip.s_addr = get_ip("lo");
  LOG(WARNING) << "lo IP is " << inet_ntoa(ip);
  return 0;
}
