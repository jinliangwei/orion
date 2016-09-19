#include <orion/bosen/spsc_queue.hpp>
#include <thread>
#include <iostream>

using namespace orion;
bosen::SPSCQueue<int> q(100);

void PushToQueue() {
  for (int i = 0; i < 200; ++i) {
    while(!q.Push(i));
  }
}

void PullFromQueue() {
  for (int i = 0; i < 200; ++i) {
    int v = 0;
    while(!q.Pull(&v));
    std::cout << v << " ";
  }
}

int main(int argc, char *argv[]) {
  std::thread producer(PushToQueue);
  std::thread consumer(PullFromQueue);
  producer.join();
  consumer.join();

  return 0;
}
