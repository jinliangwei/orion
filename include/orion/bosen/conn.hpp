#pragma once

#include <stdint.h>
#include <sys/epoll.h>
#include <cstring>
#include <utility>
#include <glog/logging.h>
#include <functional>
#include <cstddef>
#include <orion/bosen/util.hpp>
#include <orion/noncopyable.hpp>

namespace orion {
namespace bosen {
namespace conn {

// The conn implementation makes some effort to differentiate closed connection
// from other errors, but this is not extensive.
// Lost connections at arbitrary times may cause the program to crash.

typedef size_t beacon_t;

class SendBuffer;

bool CheckSendSize(const SendBuffer &send_buff, size_t sent_size);

class RecvBuffer {
 private:
  uint8_t * const mem_;
  const uint8_t *const mem_end_;
  /* set when receving data,
     which the number of bytes constituting the full message,
     consists of two parts, 1) beacon and 2) payload
  */
  beacon_t &expected_size_;
  uint8_t * const payload_mem_;
  size_t size_;
  uint8_t status_;
  // if nonzero, the next message is likely to exceed the buffer's memory capacity
  // and thus requires special attention
  size_t next_expected_size_ {0};
  size_t next_recved_size_ {0};

  static const uint8_t kEOFBit = 0x2;
  static const uint8_t kErrorBit = 0x4;

  DISALLOW_COPY(RecvBuffer);
  friend class SendBuffer;

 public:
  RecvBuffer(void *mem, size_t capacity):
      mem_((uint8_t*) get_aligned(mem, alignof(std::max_align_t))),
      mem_end_((uint8_t*) mem + capacity),
      expected_size_(*((beacon_t*) (mem_))),
      payload_mem_(mem_ + sizeof(beacon_t)),
      size_(0),
      status_(0) {
    expected_size_ = 0;
  }

  RecvBuffer(RecvBuffer && recv_buff):
      mem_(recv_buff.mem_),
      mem_end_(recv_buff.mem_end_),
      expected_size_(*((beacon_t*) (recv_buff.mem_))),
      payload_mem_(recv_buff.payload_mem_),
      size_(recv_buff.size_),
      status_(recv_buff.status_) {
    memset(mem_, 0, mem_end_ - mem_);
    recv_buff.size_ = 0;
  }

  void CopyOneMsg(const RecvBuffer &other) {
    memcpy(mem_, other.mem_, other.expected_size_);
    size_ = other.size_;
    status_ = other.status_;
    next_expected_size_ = other.next_expected_size_;
    next_recved_size_ = other.next_recved_size_;
  }

  uint8_t *get_recv_mem() {
    return mem_ + size_;
  }

  uint8_t *get_payload_mem() {
    return payload_mem_;
  }

  const uint8_t *get_payload_mem() const {
    return payload_mem_;
  }

  uint8_t *get_curr_msg_end_mem() {
    return mem_ + expected_size_;
  }

  size_t get_payload_capacity() const {
    return mem_end_ - payload_mem_;
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
    return mem_end_ - mem_;
  }

  bool EOFAtIncompleteMsg() const {
    return is_eof()
        && (
            (size_ > 0
             && size_ < sizeof(beacon_t))
            | (is_initialized()
               && size_ < expected_size_)
            );
  }

  void inc_size(size_t delta) {
    size_ += delta;
  }

  void ClearOneMsg() {
    size_t last_expected_size = expected_size_;
    CHECK(size_ >= last_expected_size) << "size = " << size_
                                       << " last_expected_size = " << last_expected_size;
    if (size_ > last_expected_size) {
      memmove(mem_, mem_ + last_expected_size, size_ - last_expected_size);
    }
    size_ -= last_expected_size;
  }

  void set_next_expected_size(size_t next_expected_size) {
    next_expected_size_ = next_expected_size;
  }

  void inc_next_recved_size(size_t delta) {
    next_recved_size_ += delta;
  }

  size_t get_next_expected_size() const {
    return next_expected_size_;
  }

  size_t get_next_recved_size() const {
    return next_recved_size_;
  }

  void reset_next_recv() {
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

  void ClearOneAndNextMsg() {
    if (size_ > expected_size_) {
      size_t size_to_clear = std::min(size_,
                                      expected_size_ + next_expected_size_);
      memmove(mem_, mem_ + size_to_clear, size_ - size_to_clear);
      size_ -= size_to_clear;
    } else {
      size_ -= expected_size_;
    }
    reset_next_recv();
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
  uint8_t * const mem_;
  const uint8_t * const mem_end_;
  /* the size to be sent, has 2 parts: 1) beacon and 2) payload */
  beacon_t &size_;
  uint8_t * const payload_mem_;
  size_t sent_size_ {0};

  size_t next_to_send_size_ {0};
  size_t next_to_send_sent_size_ {0};
  uint8_t const * next_to_send_mem_ {nullptr};
  bool owns_next_to_send_mem_ {false};

  DISALLOW_COPY(SendBuffer);

 public:
  SendBuffer(void *mem, size_t capacity):
      mem_((uint8_t*) get_aligned(mem, alignof(std::max_align_t))),
      mem_end_((uint8_t*) mem + capacity),
      size_(*((beacon_t*) (mem_))),
      payload_mem_(mem_ + sizeof(beacon_t)) {
    size_ = payload_mem_ - mem_;
    memset(mem_, 0, mem_end_ - mem_);
  }

  ~SendBuffer() { }

  SendBuffer(SendBuffer &&send_buff):
      mem_(send_buff.mem_),
      mem_end_(send_buff.mem_end_),
      size_(*((beacon_t*) (send_buff.mem_))),
      payload_mem_(send_buff.payload_mem_) { }

  uint8_t *get_payload_mem() {
    return payload_mem_;
  }

  size_t get_size() const {
    return size_;
  }

  void set_payload_size(size_t payload_size) {
    CHECK(payload_mem_ + payload_size < mem_end_)
        << "message is too large and won't fit in the send buffer";
    size_ = payload_mem_ - mem_ + payload_size;
  }

  void inc_payload_size(size_t delta) {
    CHECK(mem_ + size_ + delta < mem_end_)
        << "message is too large and won't fit in the send buffer";
    size_ += delta;
  }

  uint8_t* get_avai_payload_mem() {
    return mem_ + size_;
  }

  size_t get_payload_capacity() const {
    return mem_end_ - payload_mem_;
  }

  void CopyAndMoveNextToSend(SendBuffer *send_buff_ptr) {
    SendBuffer &send_buff = *send_buff_ptr;
    memcpy(mem_, send_buff.mem_, send_buff.get_size());
    owns_next_to_send_mem_ = send_buff.owns_next_to_send_mem_;
    send_buff.owns_next_to_send_mem_ = false;
    sent_size_ = send_buff.sent_size_;
    next_to_send_mem_ = send_buff.next_to_send_mem_;
    send_buff.next_to_send_mem_ = nullptr;
    next_to_send_size_ = send_buff.next_to_send_size_;
    send_buff.next_to_send_size_ = 0;
    next_to_send_sent_size_ = send_buff.next_to_send_sent_size_;
    send_buff.next_to_send_sent_size_ = 0;
    //reset_sent_sizes();
  }

  void Copy(const RecvBuffer &recv_buff) {
    memcpy(mem_, recv_buff.mem_, recv_buff.get_expected_size());
    reset_sent_sizes();
  }

  void set_next_to_send(const void *mem, size_t to_send_size,
                        bool owns_mem = false) {
    next_to_send_mem_ = reinterpret_cast<const uint8_t*>(mem);
    next_to_send_size_ = to_send_size;
    next_to_send_sent_size_ = 0;
    owns_next_to_send_mem_ = owns_mem;
  }

  void inc_sent_size(size_t sent_size) {
    sent_size_ += sent_size;
  }

  uint8_t* get_remaining_to_send_mem() {
    return mem_ + sent_size_;
  }

  size_t get_remaining_to_send_size() const {
    return (size_ == payload_mem_ - mem_) ? 0 : size_ - sent_size_;
  }

  void inc_next_to_send_sent_size(size_t sent_size) {
    next_to_send_sent_size_ += sent_size;
  }

  const uint8_t* get_remaining_next_to_send_mem() {
    return next_to_send_mem_ + next_to_send_sent_size_;
  }

  size_t get_remaining_next_to_send_size() {
    return next_to_send_size_ - next_to_send_sent_size_;
  }

  void reset_sent_sizes() {
    sent_size_ = 0;
    next_to_send_sent_size_ = 0;
  }

  void clear_to_send() {
    size_ = payload_mem_ - mem_;
    sent_size_ = 0;
    next_to_send_size_ = 0;
    next_to_send_sent_size_ = 0;
    if (owns_next_to_send_mem_) delete[] next_to_send_mem_;
    owns_next_to_send_mem_ = false;
    next_to_send_mem_ = nullptr;
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
  int Bind(uint32_t ip, uint16_t port);
  int Listen(int backlog) const;
  int Accept(Socket *conn) const;

  /*!
   * Used by connector for establishing connections
   * Connect to an address. The address must be "listened" on by an socket for
   * connect to succeed. This function doesn't blocking waiting for the other
   * end to accept.
   */
  int Connect(uint32_t ip, uint16_t port);

  int CheckInOutAvailability(bool *in, bool *out);

  bool Send(SendBuffer *buf) const;
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

  bool Send(SendBuffer *buf) const;
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

  int Add(int fd, void *conn, uint32_t event) {
    struct epoll_event ev;
    ev.events = event;
    ev.data.ptr = conn;
    int ret = epoll_ctl(epollfd_, EPOLL_CTL_ADD, fd, &ev);
    if (ret != 0) return ret;
    return 0;
  }

  int Set(int fd, void *conn, uint32_t event) {
    struct epoll_event ev;
    ev.events = event;
    ev.data.ptr = conn;
    int ret = epoll_ctl(epollfd_, EPOLL_CTL_MOD, fd, &ev);
    if (ret != 0) return ret;
    return 0;
  }

  int Remove(int fd) {
    int ret = epoll_ctl(epollfd_, EPOLL_CTL_DEL, fd, 0);
    if (ret != 0) return ret;
    return 0;
  }
};

struct SocketConn {
  Socket sock;
  RecvBuffer recv_buff;
  SendBuffer send_buff;
  SocketConn(Socket _sock,
             void *recv_mem,
             void *send_mem, size_t buff_capacity):
      sock(_sock),
      recv_buff(recv_mem, buff_capacity),
      send_buff(send_mem, buff_capacity) { }
  ~SocketConn() { }

  DISALLOW_COPY(SocketConn);
};

struct PipeConn {
  Pipe pipe;
  RecvBuffer recv_buff;
  SendBuffer send_buff;
  PipeConn(Pipe _pipe,
           void *recv_mem,
           void *send_mem, size_t buff_capacity):
      pipe(_pipe),
      recv_buff(recv_mem, buff_capacity),
      send_buff(send_mem, buff_capacity) { }

  DISALLOW_COPY(PipeConn);
};

}
}
}
