#include <orion/bosen/driver.hpp>
#include <orion/bosen/driver_runtime.hpp>
#include <orion/bosen/driver_thread.hpp>
#include <orion/bosen/driver_task.hpp>
#include <orion/bosen/table.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <chrono>

int
main (int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  std::chrono::time_point<std::chrono::steady_clock> start, end;
  start = std::chrono::steady_clock::now();

  LOG(INFO) << "matrix factorization driver starts!";

  orion::bosen::DriverRuntimeConfig runtime_config;
  orion::bosen::Driver driver_node(runtime_config);
  driver_node();

  auto* task = new orion::bosen::DriverTask(orion::bosen::DriverTask::Inst::kStall);
  driver_node.ScheduleTask(task);
  auto* completed_task = driver_node.GetCompletedTask();
  CHECK(task == completed_task);
  delete completed_task;
  end = std::chrono::steady_clock::now();

  driver_node.WaitUntilExit();

  std::chrono::duration<double> elapsed_seconds = end - start;
  //std::time_t end_time = std::chrono::steady_clock::to_time_t(end);

  LOG(INFO) << "number seconds spent = " << elapsed_seconds.count();

  return 0;
}
