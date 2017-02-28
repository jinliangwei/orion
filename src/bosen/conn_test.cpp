#include <orion/bosen/conn.hpp>
#include <orion/bosen/util.hpp>
#include <orion/orion_test.hpp>
#include <string>
#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <string.h>
namespace orion {
namespace test {

class ConnTest : public OrionTest {
};

TEST_F(ConnTest, SocketBindConnect) {
  uint32_t ip;
  int ret = bosen::GetIPFromIF("lo", &ip);
  ASSERT_EQ(ret, 0);
  uint16_t port = 10000;
  bosen::conn::Socket socket;
  ret = socket.Bind(ip, port);
  ASSERT_EQ(ret, 0);
  ret = socket.Listen(64);
  ASSERT_EQ(ret, 0);

  bosen::conn::Socket client;
  ret = client.Connect(ip, port);
  ASSERT_EQ(ret, 0);
  bool in, out;
  ret = client.CheckInOutAvailability(&in, &out);
  ASSERT_EQ(ret, 0);
  EXPECT_FALSE(in);
  EXPECT_TRUE(out);

  bosen::conn::Socket connected;
  ret = socket.Accept(&connected);
  ASSERT_EQ(ret, 0);
  ret = connected.CheckInOutAvailability(&in, &out);
  ASSERT_EQ(ret, 0);
  EXPECT_FALSE(in);
  EXPECT_TRUE(out);

  connected.Close();
  client.Close();
  socket.Close();
}


TEST_F(ConnTest, PollListenTest) {
  uint32_t ip;
  int ret = bosen::GetIPFromIF("lo", &ip);
  EXPECT_EQ(ret, 0);
  uint16_t port = 11000;
  bosen::conn::Socket socket;
  ret = socket.Bind(ip, port);
  ASSERT_EQ(ret, 0);
  ret = socket.Listen(64);
  ASSERT_EQ(ret, 0);

  bosen::conn::Poll poll;
  ret = poll.Init();
  ASSERT_EQ(ret, 0);

  ret = poll.Add(socket.get_fd(), &socket);
  ASSERT_EQ(ret, 0);

  bosen::conn::Socket client;
  ret = client.Connect(ip, port);
  ASSERT_EQ(ret, 0);
  bool in, out;
  ret = client.CheckInOutAvailability(&in, &out);
  ASSERT_EQ(ret, 0);
  EXPECT_FALSE(in);
  EXPECT_TRUE(out);

  epoll_event es[10];
  int num_events = poll.Wait(es, 10);
  ASSERT_GT(num_events, 0);
  ASSERT_EQ(es[0].events, EPOLLIN);

  bosen::conn::Socket* socket_ptr
      = bosen::conn::Poll::EventConn<bosen::conn::Socket>(es, 0);
  ASSERT_EQ(&socket, socket_ptr);

  bosen::conn::Socket connected;
  ret = socket.Accept(&connected);
  ASSERT_EQ(ret, 0);
  ret = connected.CheckInOutAvailability(&in, &out);
  ASSERT_EQ(ret, 0);
  EXPECT_FALSE(in);
  EXPECT_TRUE(out);

  connected.Close();
  client.Close();
  ret = poll.Remove(socket.get_fd());
  ASSERT_EQ(ret, 0);
  socket.Close();
}

TEST_F(ConnTest, SocketBindConnectPresure) {
  uint32_t ip;
  int ret = bosen::GetIPFromIF("lo", &ip);
  ASSERT_EQ(ret, 0);
  uint16_t port = 12000;
  bosen::conn::Socket socket;
  ret = socket.Bind(ip, port);
  ASSERT_EQ(ret, 0);
  ret = socket.Listen(10);
  ASSERT_EQ(ret, 0);

  constexpr int num_clients = 10;
  bosen::conn::Socket client[num_clients];
  bool in, out;
  for (int i = 0; i < num_clients; i++) {
    ret = client[i].Connect(ip, port);
    ASSERT_EQ(ret, 0);
    std::cout << i << std::endl;
    ret = client[i].CheckInOutAvailability(&in, &out);
    ASSERT_EQ(ret, 0);
    EXPECT_FALSE(in);
    EXPECT_TRUE(out);
  }

  bosen::conn::Socket connected[num_clients];
  for (int i = 0; i < num_clients; i++) {
    ret = socket.Accept(&connected[i]);
    ASSERT_EQ(ret, 0);
    std::cout << i << std::endl;
    ret = connected[i].CheckInOutAvailability(&in, &out);
    ASSERT_EQ(ret, 0);
    EXPECT_FALSE(in);
    EXPECT_TRUE(out);
  }

  for (int i = 0; i < num_clients; i++) {
    connected[i].Close();
    client[i].Close();
  }
  socket.Close();
}

TEST_F(ConnTest, SendRecvTest) {
  uint32_t ip;
  int ret = bosen::GetIPFromIF("lo", &ip);
  ASSERT_EQ(ret, 0);
  uint16_t port = 16000;
  bosen::conn::Socket socket;
  ret = socket.Bind(ip, port);
  ASSERT_EQ(ret, 0);
  ret = socket.Listen(64);
  ASSERT_EQ(ret, 0);

  bosen::conn::Socket client;
  ret = client.Connect(ip, port);
  ASSERT_EQ(ret, 0);
  bool in, out;
  ret = client.CheckInOutAvailability(&in, &out);
  ASSERT_EQ(ret, 0);
  EXPECT_FALSE(in);
  EXPECT_TRUE(out);

  bosen::conn::Socket connected;
  ret = socket.Accept(&connected);
  ASSERT_EQ(ret, 0);
  ret = connected.CheckInOutAvailability(&in, &out);
  ASSERT_EQ(ret, 0);
  EXPECT_FALSE(in);
  EXPECT_TRUE(out);

  const size_t kBuffCapcity = 100;
  const char *msg = "The great orion project is here!";
  const size_t msg_len = strlen(msg) + 1;
  std::vector<uint8_t> data_send(kBuffCapcity), data_recv(kBuffCapcity);

  bosen::conn::SendBuffer send_buff(data_send.data(), kBuffCapcity);
  bosen::conn::RecvBuffer recv_buff(data_recv.data(), kBuffCapcity);

  memcpy(send_buff.get_payload_mem(), msg, msg_len);
  send_buff.set_payload_size(msg_len);

  size_t sent_size = client.Send(&send_buff);
  EXPECT_TRUE(bosen::conn::CheckSendSize(send_buff, sent_size));

  bool recved = connected.Recv(&recv_buff);
  EXPECT_TRUE(recved);
  EXPECT_TRUE(recv_buff.ReceivedFullMsg());
  EXPECT_FALSE(recv_buff.is_error());
  EXPECT_FALSE(recv_buff.EOFAtIncompleteMsg());
  int same = memcmp(recv_buff.get_payload_mem(), msg, msg_len);
  EXPECT_EQ(same, 0);
  connected.Close();
  client.Close();
  socket.Close();
}

}
}
