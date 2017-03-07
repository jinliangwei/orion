#include <string>
#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <string.h>

#include <orion/bosen/thread_pool.hpp>
#include <orion/orion_test.hpp>
#include <orion/bosen/event_handler.hpp>

namespace orion {
namespace test {

void compute(bosen::ComputeThreadContext* thread) {
  int ret = 100;
  auto& send_buff = thread->get_send_buff();
  int* msg_ret = reinterpret_cast<int*>(send_buff.get_payload_mem());
  *msg_ret = ret;
  send_buff.set_payload_size(sizeof(int));
  auto write_pipe = thread->get_write_pipe();
  bool sent = write_pipe.Send(&send_buff);
  while (!sent) {
    sent = write_pipe.Send(&send_buff);
  }
  send_buff.clear_to_send();
}

class ThreadPoolTest : public OrionTest {
};

TEST_F(ThreadPoolTest, BasicTest) {
  bosen::ThreadPool thread_pool(4, 1024);
  thread_pool.Start();
  thread_pool.StopAll();
}

TEST_F(ThreadPoolTest, TestCompute) {
  bosen::ThreadPool thread_pool(4, 1024);
  std::vector<bosen::conn::Pipe> read_pipe_(4);
  std::vector<uint8_t> recv_mem(1024);
  bosen::conn::RecvBuffer recv_buff(recv_mem.data(), 1024);
  bosen::conn::Poll poll;
  int ret = poll.Init();
  ASSERT_EQ(ret, 0);
  for (int i = 0; i < 4; i++) {
    read_pipe_[i] = thread_pool.get_read_pipe(i);
    ret = poll.Add(read_pipe_[i].get_read_fd(), &read_pipe_[i], EPOLLIN);
    ASSERT_EQ(ret, 0);
  }
  thread_pool.Start();
  thread_pool.SchedTask(std::function<void(
      bosen::ComputeThreadContext*)>(compute));

  epoll_event es[10];
  int num_events = poll.Wait(es, 10);
  int ans = 0;
  for (int i = 0; i < num_events; ++i) {
    bosen::conn::Pipe* read_pipe_ptr
        = bosen::conn::Poll::EventConn<bosen::conn::Pipe>(es, i);
    if (es[i].events & EPOLLIN) {
      bool received = read_pipe_ptr->Recv(&recv_buff);
      if (!received) {
        received = read_pipe_ptr->Recv(&recv_buff);
      }
      CHECK (!recv_buff.is_error()) << "error during receiving " << errno;
      CHECK (!recv_buff.EOFAtIncompleteMsg()) << "error : early EOF";
      ans = *reinterpret_cast<int*>(recv_buff.get_payload_mem());
      break;
    }
  }

  EXPECT_EQ(ans, 100);
  thread_pool.StopAll();
}

}
}
