#include <orion/bosen/conn.hpp>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <strings.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <glog/logging.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>

namespace orion {
namespace bosen {
namespace conn {
static size_t WriteAllUntilBlock(int fd, const void *buf, size_t count) {
  size_t bytes_written = 0;
  do {
    ssize_t ret = write(fd, ((const char *) buf) + bytes_written, count - bytes_written);
    if (ret == -1) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        return bytes_written;
      } else {
        LOG(FATAL) << "write failed, errno = " << errno;
      }
    }
    bytes_written += ret;
  } while (bytes_written < count);
  return bytes_written;
}

int Socket::Bind(uint32_t ip, uint16_t port) {
  socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_ < 0) return -1;
  int reuse_port = 1;
  setsockopt(socket_, SOL_SOCKET, SO_REUSEPORT, &reuse_port, sizeof(reuse_port));

  sockaddr_in addr;
  bzero(&addr, sizeof(sockaddr_in));

  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = ip;
  addr.sin_port = htons(port);
  return bind(socket_, (sockaddr *) &addr, sizeof(sockaddr_in));
}

int Socket::Connect(uint32_t ip, uint16_t port) {
  socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_ < 0) return -1;

  sockaddr_in addr;
  bzero(&addr, sizeof(sockaddr_in));

  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = ip;
  addr.sin_port = htons(port);
  int ret = connect(socket_, (sockaddr *) &addr, sizeof(sockaddr_in));
  if (ret < 0) return -1;

  int file_status_flags = fcntl(socket_, F_GETFL);
  file_status_flags |= O_NONBLOCK;
  ret = fcntl(socket_, F_SETFL, file_status_flags);
  if (ret < 0) return -2;
  return 0;
}

int Socket::CheckInOutAvailability(bool *in, bool *out) {
  *in = false;
  *out = false;
  pollfd fd_arr[1];
  fd_arr[0].fd = socket_;
  fd_arr[0].events = POLLIN | POLLOUT;
  int ret = poll(fd_arr, 1, 0);

  if (ret == 0) return 0;
  if (ret < 0) return ret;

  if (fd_arr[0].revents & POLLIN) *in = true;
  if (fd_arr[0].revents & POLLOUT) *out = true;
  return 0;
}

int Socket::Listen(int backlog) const {
  return listen(socket_, backlog);
}

int Socket::Accept(Socket *socket) const {
  int sock = accept(socket_, 0, 0);
  if(sock < 0) return -1;

  int file_status_flags = fcntl(sock, F_GETFL);
  file_status_flags |= O_NONBLOCK;
  int ret = fcntl(sock, F_SETFL, file_status_flags);
  if (ret < 0) return -1;

  socket->socket_ = sock;
  return 0;
}

void Socket::Close() const {
  if (socket_) close(socket_);
}

int Pipe::CreateUniPipe(Pipe *read, Pipe *write) {
  int fd[2];

  int success = pipe(fd);
  if (success != 0) return -1;
  read->read_ = fd[0];
  read->write_ = 0;

  write->read_ = 0;
  write->write_ = fd[1];

  for (int i = 0; i < 2; i++) {
    int file_status_flags = fcntl(fd[i], F_GETFL);
    file_status_flags |= O_NONBLOCK;
    int ret = fcntl(fd[i], F_SETFL, file_status_flags);
    if (ret < 0) return -1;
  }

  return 0;
}

int Pipe::CreateBiPipe(Pipe pipes[2]) {
  int fd[2], fd2[2];

  int success = pipe(fd);
  if (success != 0) return -1;

  success = pipe(fd2);
  if (success != 0) {
    close(fd[0]);
    close(fd[1]);
    return -1;
  }

  pipes[0].read_ = fd[0];
  pipes[0].write_ = fd2[1];

  pipes[1].read_ = fd2[0];
  pipes[1].write_ = fd[1];

  for (int i = 0; i < 2; i++) {
    int file_status_flags = fcntl(fd[i], F_GETFL);
    file_status_flags |= O_NONBLOCK;
    int ret = fcntl(fd[i], F_SETFL, file_status_flags);
    if (ret < 0) return -1;
  }

  for (int i = 0; i < 2; i++) {
    int file_status_flags = fcntl(fd2[i], F_GETFL);
    file_status_flags |= O_NONBLOCK;
    int ret = fcntl(fd2[i], F_SETFL, file_status_flags);
    if (ret < 0) return -1;
  }

  return 0;
}

void Pipe::Close() const {
  if (read_) close(read_);
  if (write_) close(write_);
}

bool
Pipe::Send(SendBuffer *buf) const {
  size_t bytes_written = 0;
  if (buf->get_remaining_to_send_size() > 0) {
    bytes_written = WriteAllUntilBlock(
        write_, buf->get_remaining_to_send_mem(),
        buf->get_remaining_to_send_size());
    buf->inc_sent_size(bytes_written);
  }

  if (buf->get_remaining_to_send_size() == 0) {
    if (buf->get_remaining_next_to_send_size() == 0) return true;
    bytes_written = WriteAllUntilBlock(
        write_, buf->get_remaining_next_to_send_mem(),
        buf->get_remaining_next_to_send_size());
    buf->inc_next_to_send_sent_size(bytes_written);
  }
  if (buf->get_remaining_next_to_send_size() == 0) return true;
  return false;
}

bool Pipe::Recv(RecvBuffer *buf) const {
  ssize_t ret = read(read_, buf->get_recv_mem(),
                     buf->get_capacity() - buf->get_size());

  if (ret < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      return false;
    buf->set_error();
    return true;
  } else if (ret == 0) {
    buf->set_eof();
    return true;
  } else {
    buf->inc_size(ret);
  }

  if (buf->is_initialized())
    CHECK(buf->get_capacity() >= buf->get_expected_size())
        << "received the first " << ret << " bytes"
        << " message is too large "
        << "capacity = " << buf->get_capacity()
        << " expected size = " << buf->get_expected_size()
        << " buff = " << (void*) buf;
  if (buf->ReceivedFullMsg()) return true;
  return false;
}

bool Pipe::Recv(RecvBuffer *buf, void *mem) const {

  ssize_t ret = read(read_, reinterpret_cast<uint8_t*>(mem),
                     buf->get_next_expected_size() \
                     - buf->get_next_recved_size());

  if (ret < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      return false;
    buf->set_error();
    return true;
  } else if (ret == 0) {
    buf->set_eof();
    return true;
  } else {
    buf->inc_next_recved_size(ret);
  }

  if (buf->ReceivedFullNextMsg()) return true;
  return false;
}

bool
Socket::Send(SendBuffer *buf) const {
  size_t bytes_written = 0;
  if (buf->get_remaining_to_send_size() > 0) {
    bytes_written = WriteAllUntilBlock(
        socket_, buf->get_remaining_to_send_mem(),
        buf->get_remaining_to_send_size());
    buf->inc_sent_size(bytes_written);
  }

  if (buf->get_remaining_to_send_size() == 0) {
    if (buf->get_remaining_next_to_send_size() == 0) return true;
    bytes_written = WriteAllUntilBlock(
        socket_, buf->get_remaining_next_to_send_mem(),
        buf->get_remaining_next_to_send_size());
    buf->inc_next_to_send_sent_size(bytes_written);
  }
  if (buf->get_remaining_next_to_send_size() == 0) return true;
  return false;
}

bool Socket::Recv(RecvBuffer *buf) const {
  ssize_t ret = read(socket_, buf->get_recv_mem(),
                     buf->get_capacity() - buf->get_size());
  if (ret < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      return false;
    buf->set_error();
    return true;
  } else if (ret == 0) {
    buf->set_eof();
    return true;
  } else {
    buf->inc_size(ret);
  }

  if (buf->is_initialized())
    CHECK_GE(buf->get_capacity(), buf->get_expected_size())
        << "message is too large, expected size = "
        << buf->get_expected_size()
        << " buf = " << (void*) buf;
  if (buf->ReceivedFullMsg()) return true;
  return false;
}

bool Socket::Recv(RecvBuffer *buf, void *mem) const {
  ssize_t ret = read(socket_, reinterpret_cast<uint8_t*>(mem),
                     buf->get_next_expected_size() \
                     - buf->get_next_recved_size());
  if (ret < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      return false;
    buf->set_error();
    return true;
  } else if (ret == 0) {
    buf->set_eof();
    return true;
  } else {
    buf->inc_next_recved_size(ret);
  }

  if (buf->ReceivedFullNextMsg()) return true;
  return false;
}

bool CheckSendSize(const SendBuffer &send_buff, size_t sent_size) {
  return (send_buff.get_size() == sent_size);
}

}
}
}
