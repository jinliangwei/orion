#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <memory>
#include <atomic>
#include <glog/logging.h>
#include <orion/bosen/conn.hpp>

namespace orion {
namespace bosen {

class ComputeThreadContext {
 private:
  std::vector<uint8_t> send_mem_;
  conn::SendBuffer send_buff_;
  conn::Pipe write_pipe_;
  conn::Pipe read_pipe_;
  const int32_t kId;
 public:
  ComputeThreadContext(size_t send_buff_capacity, int32_t id):
      send_mem_(send_buff_capacity),
      send_buff_(send_mem_.data(), send_buff_capacity),
      kId(id) {
    int ret = conn::Pipe::CreateUniPipe(&read_pipe_, &write_pipe_);
    CHECK_EQ(ret, 0) << "create pipe failed";
  }

  ~ComputeThreadContext() {
    write_pipe_.Close();
    read_pipe_.Close();
  }

  conn::Pipe get_read_pipe() const {
    return read_pipe_;
  }

  conn::Pipe get_write_pipe() const {
    return write_pipe_;
  }

  int32_t get_id() const {
    return kId;
  }

  conn::SendBuffer& get_send_buff() {
    return send_buff_;
  }
};

class ThreadPool {
 private:
  std::queue<std::function<void(ComputeThreadContext*)>> task_queue_;
  std::mutex mtx_;
  std::condition_variable cv_;
  const size_t kNumThreads;
  std::vector<std::unique_ptr<ComputeThreadContext>> compute_thread_ctx_;
  std::vector<std::thread> runner_;
  bool stop_ {false};
 public:
  ThreadPool(size_t num_threads, size_t send_buff_capacity):
      kNumThreads(num_threads),
      compute_thread_ctx_(num_threads),
      runner_(num_threads) {
    for (int i = 0; i < kNumThreads; i++) {
      compute_thread_ctx_[i] = std::make_unique<ComputeThreadContext>(
          send_buff_capacity, i);
    }
  }

  void Start() {
    for (int i = 0; i < kNumThreads; i++) {
      runner_[i] = std::thread(
          &ThreadPool::RunTask, this, compute_thread_ctx_[i].get());
    }
  }

  void SchedTask(std::function<void(ComputeThreadContext*)> task) {
    std::unique_lock<std::mutex> lock(mtx_);
    task_queue_.push(task);
    cv_.notify_one();
  }

  void StopAll() {
    std::unique_lock<std::mutex> lock(mtx_);
    stop_ = true;
    cv_.notify_all();
    lock.unlock();

    for (int i = 0; i < kNumThreads; i++) {
      runner_[i].join();
    }
  }

  conn::Pipe get_read_pipe(int id) const {
    return compute_thread_ctx_[id]->get_read_pipe();
  }

 private:
  void RunTask(ComputeThreadContext* compute_thread_ctx) {
    while (true) {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait(lock, [this]{ return this->stop_ || (this->task_queue_.size() > 0); });
      if (stop_) break;
      std::function<void(ComputeThreadContext*)> task = task_queue_.front();
      task_queue_.pop();
      lock.unlock();
      task(compute_thread_ctx);
    }
  }
};

}
}
