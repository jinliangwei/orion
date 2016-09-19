#pragma once

#include <memory>

#include <orion/bosen/table.hpp>
#include <orion/noncopyable.hpp>
#include <orion/helper.hpp>
#include <orion/bosen/driver_task.hpp>
#include <stdint.h>
#include <unordered_map>
#include <map>
#include <utility>

namespace orion {
namespace bosen {
using DriverTaskMap = std::map<int32_t, DriverTask>;
using DriverTaskIterator = DriverTaskMap::iterator;

class DriverRuntimeConfig {
 private:
  DriverTaskMap tasks_;

  friend class DriverRuntime;
 public:
  DriverRuntimeConfig() { }

  void AddTask(int32_t task_id, DriverTask::Inst inst,
               uint64_t param1, uint64_t param2,
               DriverFunc func) {
    tasks_.emplace(std::make_pair(
        task_id, DriverTask(inst, param1, param2, func)));
  }

};

class DriverRuntime {
 private:
  DriverTaskMap tasks_;
  DriverTaskIterator task_iter_;
  const size_t num_executors_;
  int32_t max_x_ {0}, max_y_ {0};

 public:
  DriverRuntime(const DriverRuntimeConfig &runtime_config,
                size_t num_executors):
      tasks_(runtime_config.tasks_),
      task_iter_(tasks_.begin()),
      num_executors_(num_executors) { }

  ~DriverRuntime() { }
  DISALLOW_COPY(DriverRuntime);

  DriverTaskMap::value_type &next() {
    CHECK(task_iter_ != tasks_.end());
    auto &task_pair = *task_iter_;
    task_iter_++;
    return task_pair;
  }

  void Jump(int32_t task_id) {
    auto iter = tasks_.find(task_id);
    CHECK(iter != tasks_.end());
    task_iter_ = iter;
  }

  void CompareSetMaxX(int32_t x) {
    if (max_x_ < x) max_x_ = x;
  }

  void CompareSetMaxY(int32_t y) {
    if (max_y_ < y) max_y_ = y;
  }

  int32_t get_max_x() {
    return max_x_;
  }

  int32_t get_max_y() {
    return max_y_;
  }
};

}
}
