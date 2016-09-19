#pragma once

#include <stdint.h>
#include <sys/epoll.h>
#include <cstring>
#include <utility>
#include <orion/noncopyable.hpp>
#include <glog/logging.h>

namespace orion {
namespace bosen {
namespace conn {

typedef size_t beacon_t;

class SendBuffer;

bool CheckSendSize(const SendBuffer &send_buff, size_t sent_size);

class RecvBuffer {
 private:
  uint8_t *mem_;
  const size_t capacity_;
  /* set when receving data,
     which the number of bytes constituting the full message,
     consists of two parts, 1) beacon and 2) payload
  */
  beacon_t &expected_size_;
  size_t size_;
  uint8_t status_;
  // if nonzero, the next message is likely to exceed the buffer's memory capacity
  // and thus requires special attention
  size_t next_expected_size_ {0};
  size_t next_recved_size_ {0};

  static const uint8_t kEOFBit = 0x2;
  static const uint8_t kErrorBit = 0x4;

  DISALLOW_COPY(RecvBuffer);

 public:
  RecvBuffer(void *mem, size_t capacity):
      mem_((uint8_t*) mem),
      capacity_(capacity),
      expected_size_(*((beacon_t*) (mem))),
      size_(0),
      status_(0) {
    expected_size_ = 0;
  }

  RecvBuffer(RecvBuffer && recv_buff):
      mem_(recv_buff.mem_),
      capacity_(recv_buff.capacity_),
      expected_size_(*((beacon_t*) (recv_buff.mem_))),
      size_(recv_buff.size_),
      status_(recv_buff.status_) {
    recv_buff.size_ = 0;
    recv_buff.mem_ = 0;
  }

  void set_next_expected_size(size_t next_expected_size) {
    next_expected_size_ = next_expected_size;
  }

  void IncNextRecvedSize(size_t delta) {
    next_recved_size_ += delta;
  }

  size_t get_next_expected_size() const {
    return next_expected_size_;
  }

  size_t get_next_recved_size() const {
    return next_recved_size_;
  }

  void ResetNextRecv() {
    next_expected_size_ = 0;
    next_recved_size_ = 0;
  }

  bool ReceivedFullNextMsg() const {
    CHECK_LE(next_recved_size_, next_expected_size_);
    return (next_recved_size_ == next_expected_size_);
  }

  bool IsExepectingNextMsg() const {
    return (next_expected_size_ > 0);
  }

  uint8_t *get_mem() {
    return mem_;
  }

  const uint8_t *get_mem() const {
    return mem_;
  }

  uint8_t *GetAvailableMem() {
    return mem_ + size_;
  }

  uint8_t *get_payload_mem() {
    return mem_ + sizeof(beacon_t);
  }

  size_t get_payload_capacity() const {
    return capacity_ - sizeof(beacon_t);
  }

  bool is_initialized() const {
    return (size_ >= sizeof(beacon_t));
  }

  bool is_eof() const {
    return status_ & kEOFBit;
  }

  bool is_error() const {
    return status_ & kErrorBit;
  }

  void set_eof() {
    status_ |= kEOFBit;
  }

  void set_error() {
    status_ |= kErrorBit;
  }

  size_t get_expected_size() const {
    return (size_t) expected_size_;
  }

  size_t get_size() const {
    return size_;
  }

  size_t get_capacity() const {
    return capacity_;
  }

  bool EOFAtIncompleteMsg() const {
    return is_eof()
        && (
            (!is_initialized()
             && size_ > 0
             && size_ < sizeof(beacon_t))
            | (is_initialized()
               && size_ < expected_size_)
            );
  }

  void IncSize(size_t delta) {
    size_ += delta;
  }

  void ClearOneMsg() {
    size_t last_expected_size = expected_size_;
    if (size_ > last_expected_size) {
      memmove(mem_, mem_ + last_expected_size, size_ - last_expected_size);
    }
    size_ -= last_expected_size;
  }

  void ClearOneAndNextMsg() {
    if (size_ > expected_size_) {
      size_t size_to_clear = std::min(size_,
                                      expected_size_ + next_expected_size_);
      memmove(mem_, mem_ + size_to_clear, size_ - size_to_clear);
      size_ -= size_to_clear;
    } else {
      size_ -= expected_size_;
    }
    ResetNextRecv();
  }

  void Reset() {
    size_ = 0;
    status_ = 0;
  }

  bool ReceivedFullMsg() const {
    return is_initialized() && (size_ >= expected_size_);
  }
};

class SendBuffer {
 private:
  uint8_t *mem_;
  /* the size to be sent, has 2 parts: 1) beacon and 2) payload */
  beacon_t &size_;
  const size_t capacity_;

  DISALLOW_COPY(SendBuffer);

 public:
  SendBuffer(void *mem, size_t capacity):
      mem_((uint8_t*) mem),
      size_(*((beacon_t*) (mem))),
      capacity_(capacity) {
    size_ = 0;
  }

  ~SendBuffer() { }

  SendBuffer(SendBuffer &&send_buff):
      mem_(send_buff.mem_),
      size_(*((beacon_t*) (send_buff.mem_))),
      capacity_(send_buff.capacity_) { }

  uint8_t *get_mem() {
    return mem_;
  }

  const uint8_t *get_mem() const {
    return mem_;
  }

  uint8_t *get_payload_mem() {
    return mem_ + sizeof(beacon_t);
  }

  size_t get_size() const {
    return size_;
  }

  void set_payload_size(size_t payload_size) {
    CHECK(payload_size < get_payload_capacity())
        << "message is too large and won't fit in the send buffer";
    size_ = payload_size + sizeof(beacon_t);
  }

  void inc_payload_size(size_t delta) {
    CHECK(size_ + delta < get_payload_capacity())
        << "message is too large and won't fit in the send buffer";
    size_ += delta;
  }

  uint8_t* get_avai_payload_mem() {
    return mem_ + size_;
  }

  size_t get_remaining_payload_capacity() const {
    return capacity_ - size_;
  }

  void reset() {
    size_ = 0;
  }

  void Copy(const RecvBuffer &recv_buff) {
    memcpy(mem_, recv_buff.get_mem(), recv_buff.get_expected_size());
  }

  size_t get_payload_capacity() const {
    return capacity_ - sizeof(beacon_t);
  }
};

class Socket {
 private:
  int socket_ {0};

 public:
  Socket ():
      socket_(0) { }

  explicit Socket(int fd):
      socket_(fd) { }

  Socket(const Socket &sock):
      socket_(sock.socket_) { }

  Socket & operator = (const Socket &sock) {
    socket_ = sock.socket_;
    return *this;
  }

  bool operator == (const Socket &sock) const {
    return socket_ == sock.socket_;
  }

  /*!
   * Used by listener for accepting connections
   */
  int Bind(const char *ip, uint16_t port);
  int Listen(int backlog) const;
  int Accept(Socket *conn) const;

  /*!
   * Used by connector for establishing connections
   */
  int Connect(const char *ip, uint16_t port);

  size_t Send(const SendBuffer *buf) const;
  size_t Send(const void *mem, size_t to_send) const;
  bool Recv(RecvBuffer *buf) const;

  /*
   * Receive the number of bytes as indicated by the receive buffer's
   * information about the next message.
   * Return true if
   * 1) received an complete message (EOF may or may not be set);
   * 2) received an incomplete message but there's error
   * (EOF or error is encountered);
   */
  bool Recv(RecvBuffer *buf, void *mem) const;

  void Close() const;

  int get_fd() const {
    return socket_;
  }

  size_t GetSerializedSize() const {
    return sizeof(socket_);
  }

  void Serialized(void *buf) const {
    *(reinterpret_cast<int*>(buf)) = socket_;
  }

  void Deserialize(const void *buf) {
    socket_ = *(reinterpret_cast<const int*>(buf));
  }
};

class Pipe {
 private:
  int read_;
  int write_;

 public:
  Pipe():
      read_(0),
      write_(0) { }

  Pipe(const Pipe &pipe):
      read_(pipe.read_),
      write_(pipe.write_) { }

  Pipe & operator = (const Pipe &pipe) {
    read_ = pipe.read_;
    write_ = pipe.write_;
    return *this;
  }

  bool operator == (const Pipe &pipe) const {
    return (read_ == pipe.read_) && (write_ == pipe.write_);
  }

  int get_read_fd() const {
    return read_;
  }

  int get_write_fd() const {
    return write_;
  }

  void Close() const;

  size_t Send(const SendBuffer *buf) const;
  size_t Send(const void *mem, size_t to_send) const;
  /*
   * Return true if
   * 1) received an complete message (EOF may or may not be set);
   * 2) received an incomplete message but there's error
   * (EOF or error is encountered);
   */
  bool Recv(RecvBuffer *buf) const;

  /*
   * Receive the number of bytes as indicated by the receive buffer's
   * information about the next message.
   * Return true if
   * 1) received an complete message (EOF may or may not be set);
   * 2) received an incomplete message but there's error
   * (EOF or error is encountered);
   */
  bool Recv(RecvBuffer *buf, void *mem) const;

  static int CreateUniPipe(Pipe *read, Pipe *write);

  static int CreateBiPipe(Pipe pipes[2]);
};

class Poll {
 private:
  int epollfd_;

 public:
  Poll() { }

  int Init() {
    epollfd_ = epoll_create(10);
    if (epollfd_ < 0) return -1;
    return 0;
  }

  int Wait(epoll_event *es, size_t max_events) {
    return epoll_wait(epollfd_, es, max_events, -1);
  }

  template<typename PollConn>
  static PollConn *EventConn(epoll_event *es, int i) {
    return (PollConn*) es[i].data.ptr;
  }

  int Add(int fd, void *conn) {
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.ptr = conn;
    int r = epoll_ctl(epollfd_, EPOLL_CTL_ADD, fd, &ev);
    if (r != 0) return -1;
    return 0;
  }

  int Remove(int fd) {
    int r = epoll_ctl(epollfd_, EPOLL_CTL_DEL, fd, 0);
    if (r != 0) return -1;
    return 0;
  }
};

struct SocketConn {
  Socket sock;
  RecvBuffer recv_buff;

  SocketConn(Socket _sock,
             void *recv_mem, size_t recv_capacity):
      sock(_sock),
      recv_buff(recv_mem, recv_capacity) { }
  ~SocketConn() { }

  DISALLOW_COPY(SocketConn);
};

struct PipeConn {
  Pipe pipe;
  RecvBuffer recv_buff;

  PipeConn(Pipe _pipe,
           void *recv_mem,
           size_t recv_capacity):
      pipe(_pipe),
      recv_buff(recv_mem, recv_capacity) { }

  DISALLOW_COPY(PipeConn);
};

}
}
}
