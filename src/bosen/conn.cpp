#include <orion/bosen/conn.hpp>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <strings.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <glog/logging.h>

namespace orion {
namespace bosen {
namespace conn {
static size_t write_all(int fd, const void *buf, size_t count) {
  size_t bytes_written = 0;
  do {
    ssize_t ret = write(fd, ((const char *) buf) + bytes_written, count - bytes_written);
    if (ret == -1) return bytes_written;
    bytes_written += ret;
  } while (bytes_written < count);
  return bytes_written;
}

int Socket::Bind(uint64_t ip, uint16_t port) {
  socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_ < 0) return -1;

  sockaddr_in addr;
  bzero(&addr, sizeof(sockaddr_in));

  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = ip;
  addr.sin_port = htons(port);
  return bind(socket_, (sockaddr *) &addr, sizeof(sockaddr_in));
}

int Socket::Connect(uint64_t ip, uint16_t port) {
  socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_ < 0) return -1;

  sockaddr_in addr;
  bzero(&addr, sizeof(sockaddr_in));

  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = ip;
  addr.sin_port = htons(port);
  return connect(socket_, (sockaddr *) &addr, sizeof(sockaddr_in));
}

int Socket::Listen(int backlog) const {
  return listen(socket_, backlog);
}

int Socket::Accept(Socket *socket) const {
  int sock = accept(socket_, 0, 0);
  if(sock < 0) return -1;

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

  return 0;
}

void Pipe::Close() const {
  if (read_) close(read_);
  if (write_) close(write_);
}

size_t Pipe::Send(const SendBuffer *buf) const {
  size_t bytes_written = write_all(write_, buf->get_mem(), buf->get_size());
  return bytes_written;
}

size_t Pipe::Send(const void *mem, size_t to_send) const {
  size_t bytes_written = write_all(write_, mem, to_send);
  return bytes_written;
}

bool Pipe::Recv(RecvBuffer *buf) const {
  ssize_t ret = read(read_, buf->GetAvailableMem(),
                     buf->get_capacity() - buf->get_size());

  if (ret < 0) {
    buf->set_error();
    return true;
  } else if (ret == 0) {
    buf->set_eof();
    return true;
  } else {
    buf->IncSize(ret);
  }

  if (buf->is_initialized())
    CHECK(buf->get_capacity() >= buf->get_expected_size()) << "message is too large";
  if (buf->ReceivedFullMsg()) return true;
  return false;
}

bool Pipe::Recv(RecvBuffer *buf, void *mem) const {
  ssize_t ret = read(read_, reinterpret_cast<uint8_t*>(mem) \
                     + buf->get_next_recved_size(),
                     buf->get_next_expected_size() \
                     - buf->get_next_recved_size());

  if (ret < 0) {
    buf->set_error();
    return true;
  } else if (ret == 0) {
    buf->set_eof();
    return true;
  } else {
    buf->IncNextRecvedSize(ret);
  }

  if (buf->ReceivedFullNextMsg()) return true;
  return false;
}

size_t Socket::Send(const SendBuffer *buf) const {
  size_t bytes_written = write_all(socket_, buf->get_mem(), buf->get_size());
  return bytes_written;
}

size_t Socket::Send(const void *mem, size_t to_send) const {
  size_t bytes_written = write_all(socket_, mem, to_send);
  return bytes_written;
}

bool Socket::Recv(RecvBuffer *buf) const {
  ssize_t ret = read(socket_, buf->GetAvailableMem(),
                     buf->get_capacity() - buf->get_size());
  if (ret < 0) {
    buf->set_error();
    return true;
  } else if (ret == 0) {
    buf->set_eof();
    return true;
  } else {
    buf->IncSize(ret);
  }

  if (buf->is_initialized())
    CHECK(buf->get_capacity() >= buf->get_expected_size()) << "message is too large";
  if (buf->ReceivedFullMsg()) return true;
  return false;
}

bool Socket::Recv(RecvBuffer *buf, void *mem) const {
  ssize_t ret = read(socket_, reinterpret_cast<uint8_t*>(mem) \
                     + buf->get_next_recved_size(),
                     buf->get_next_expected_size() \
                     - buf->get_next_recved_size());
  if (ret < 0) {
    buf->set_error();
    return true;
  } else if (ret == 0) {
    buf->set_eof();
    return true;
  } else {
    buf->IncNextRecvedSize(ret);
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
