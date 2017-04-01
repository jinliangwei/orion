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

  std::queue<JuliaTask*> task_queue_;
  bool stop_ {false};
  std::mutex mtx_;
  std::condition_variable cv_;
  JuliaEvaluator julia_eval_;
  // for waiting to write
  conn::Poll poll_;
  static constexpr size_t kNumEvents = 1;
  epoll_event es_[kNumEvents];

 private:
  void BlockNotifyExecutor() {
    message::ExecuteMsgHelper::CreateMsg<
      message::ExecuteMsgJuliaEvalAck>(&send_buff_);
    bool sent = write_pipe_.Send(&send_buff_);
    if (sent) return;
    int ret = poll_.Add(write_pipe_.get_write_fd(), &write_pipe_, EPOLLOUT);
    CHECK_EQ(ret, 0);
    while (!sent) {
      int num_events = poll_.Wait(es_, kNumEvents);
      CHECK(num_events > 0);
      CHECK(es_[0].events & EPOLLOUT);
      CHECK(conn::Poll::EventConn<conn::Pipe>(es_, 0) == &write_pipe_);
      sent = write_pipe_.Send(&send_buff_);
    }
    poll_.Remove(write_pipe_.get_write_fd());
    send_buff_.reset_sent_sizes();
    send_buff_.clear_to_send();
  }

 public:
  JuliaEvalThread(size_t send_buff_capacity):
      send_mem_(send_buff_capacity),
      send_buff_(send_mem_.data(), send_buff_capacity) {
    int ret = conn::Pipe::CreateUniPipe(&read_pipe_, &write_pipe_);
    CHECK_EQ(ret, 0) << "create pipe failed";
    ret = poll_.Init();
    CHECK_EQ(ret, 0) << "init poll failed";
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

  void SchedTask(JuliaTask* task) {
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
      BlockNotifyExecutor();
    }
    julia_eval_.AtExitHook();
  }
};

}
}