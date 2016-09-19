#include <orion/bosen/mpsc_queue.hpp>
#include <thread>
#include <glog/logging.h>
#include <pthread.h>

using namespace orion;
bosen::MPSCQueue<int> q(2000);
pthread_barrier_t barrier;

void PushToQueue(int start) {
  pthread_barrier_wait(&barrier);
  for (int i = start; i < 200; i += 4) {
    while(!q.Push(i));
  }

  LOG(INFO) << "done = " << start << std::endl;
}

void PullFromQueue() {
  pthread_barrier_wait(&barrier);
  int sum = 0;
  for (int i = 0; i < 200; ++i) {
    int v = 0;
    while(!q.Pull(&v));
    sum += v;
  }
  LOG(INFO) << "sum = " << sum;
  LOG(INFO) << std::endl;

  LOG(INFO) << "done";
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  pthread_barrier_init(&barrier, 0, 5);
  std::thread producer0(PushToQueue, 0);
  std::thread producer1(PushToQueue, 1);
  std::thread producer2(PushToQueue, 2);
  std::thread producer3(PushToQueue, 3);
  std::thread consumer(PullFromQueue);
  producer0.join();
  producer1.join();
  producer2.join();
  producer3.join();

  consumer.join();
  pthread_barrier_destroy(&barrier);
  return 0;
}
