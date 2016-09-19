#include <orion/bosen/worker.hpp>
#include <orion/bosen/worker_runtime.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>

int main (int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "matrix factorization worker starts!";

  orion::bosen::Worker worker;
  worker();
  return 0;
}
