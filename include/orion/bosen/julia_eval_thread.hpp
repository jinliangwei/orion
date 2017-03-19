#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>
#include <atomic>
#include <glog/logging.h>
#include <orion/bosen/conn.hpp>
#include <orion/bosen/julia_evaluator.hpp>

namespace orion {
namespace bosen {

class JuliaEvalThread {
 private:
  std::vector<uint8_t> send_mem_;
  conn::SendBuffer send_buff_;
  conn::Pipe write_pipe_;
  conn::Pipe read_pipe_;
  std::thread runner_;

  std::queue<const JuliaTask*> task_queue_;
  bool stop_ {false};
  std::mutex mtx_;
  std::condition_variable cv_;
  JuliaEvaluator julia_eval_;

 public:
  JuliaEvalThread(size_t send_buff_capacity):
      send_mem_(send_buff_capacity),
      send_buff_(send_mem_.data(), send_buff_capacity) {
    int ret = conn::Pipe::CreateUniPipe(&read_pipe_, &write_pipe_);
    CHECK_EQ(ret, 0) << "create pipe failed";
  }

  ~JuliaEvalThread() {
    write_pipe_.Close();
    read_pipe_.Close();
  }

  conn::Pipe get_read_pipe() const {
    return read_pipe_;
  }

  void Start() {
    runner_ = std::thread(&JuliaEvalThread::operator(), this);
  }

  void Stop() {
    std::unique_lock<std::mutex> lock(mtx_);
    stop_ = true;
    cv_.notify_one();
    lock.unlock();
    runner_.join();

  }

  void SchedTask(const JuliaTask* task) {
    std::unique_lock<std::mutex> lock(mtx_);
    task_queue_.push(task);
    cv_.notify_one();
  }

  void operator() () {
    julia_eval_.Init();
    while (true) {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait(lock, [this]{ return this->stop_ || (this->task_queue_.size() > 0); });
      if (stop_) break;
      auto task = task_queue_.front();
      task_queue_.pop();
      lock.unlock();
      julia_eval_.ExecuteTask(task);
    }
    julia_eval_.AtExitHook();
  }
};

}
}
