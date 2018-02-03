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
#include <orion/bosen/julia_task.hpp>
#include <orion/bosen/julia_thread_requester.hpp>

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
  // for waiting to write
  conn::Poll poll_;
  static constexpr size_t kNumEvents = 1;
  epoll_event es_[kNumEvents];

  const std::string kOrionHome;
  const size_t kNumServers;
  const size_t kNumExecutors;
  JuliaThreadRequester julia_requester_;

 private:
  void BlockNotifyExecutor(JuliaTask *task) {
    message::ExecuteMsgHelper::CreateMsg<
      message::ExecuteMsgJuliaEvalAck>(&send_buff_, task);
    bool sent = write_pipe_.Send(&send_buff_);
    if (sent) {
      send_buff_.reset_sent_sizes();
      send_buff_.clear_to_send();
      return;
    }
    int ret = poll_.Add(write_pipe_.get_write_fd(), &write_pipe_, EPOLLOUT);
    CHECK_EQ(ret, 0);
    while (!sent) {
      int num_events = poll_.Wait(es_, kNumEvents);
      CHECK(num_events > 0);
      CHECK(es_[0].events & EPOLLOUT);
      CHECK(conn::Poll::EventConn<conn::Pipe>(es_, 0) == &write_pipe_);
      sent = write_pipe_.Send(&send_buff_);
    }
    LOG(INFO) << "send done!";
    poll_.Remove(write_pipe_.get_write_fd());
    send_buff_.reset_sent_sizes();
    send_buff_.clear_to_send();
  }

 public:
  JuliaEvalThread(size_t send_buff_capacity,
                  const std::string &orion_home,
                  size_t num_servers,
                  size_t num_executors,
                  int32_t my_executor_or_server_id,
                  bool is_executor):
      send_mem_(send_buff_capacity),
      send_buff_(send_mem_.data(), send_buff_capacity),
      kOrionHome(orion_home),
      kNumServers(num_servers),
      kNumExecutors(num_executors),
      julia_requester_(write_pipe_,
                       send_mem_,
                       send_buff_,
                       poll_,
                       kNumEvents,
                       es_,
                       my_executor_or_server_id,
                       is_executor) {
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
    JuliaEvaluator::Init(kOrionHome, kNumServers,
                         kNumExecutors);
    while (true) {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait(lock, [this]{ return this->stop_ || (this->task_queue_.size() > 0); });
      if (stop_) break;
      auto task = task_queue_.front();
      task_queue_.pop();
      lock.unlock();
      JuliaEvaluator::ExecuteTask(task);
      BlockNotifyExecutor(task);
    }
    JuliaEvaluator::AtExitHook();
  }

  JuliaThreadRequester* GetJuliaThreadRequester() {
    return &julia_requester_;
  }
};

}
}
