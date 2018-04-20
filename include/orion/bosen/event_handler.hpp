#pragma once
#include <orion/bosen/conn.hpp>
#include <functional>
#include <glog/logging.h>
#include <set>

namespace orion {
namespace bosen {

template<typename PollConn>
class EventHandler {
 private:
  static constexpr size_t kNumEvents = 100;
  conn::Poll poll_;
  epoll_event es_[kNumEvents];
  std::set<PollConn*> read_set_;
  std::function<void(PollConn*)> connect_event_handler_;
  std::function<int(PollConn*)> read_event_handler_;
  std::function<int(PollConn*)> closed_connection_handler_;
  std::function<void(PollConn*)> write_event_handler_;
  bool ReadAndRunReadEventHandler(PollConn* poll_conn);
  bool RunReadEventHandler(PollConn* poll_conn);

 public:
  EventHandler();
  void SetConnectEventHandler(const std::function<void(PollConn*)>&
                             connect_event_handler);
  void SetReadEventHandler(const std::function<int(PollConn*)>&
                           read_event_handler);
  void SetWriteEventHandler(const std::function<void(PollConn*)>&
                           write_event_handler);
  void SetDefaultWriteEventHandler();
  void SetClosedConnectionHandler(const std::function<int(PollConn*)>&
                                  closed_connection_handler);
  int SetToReadWrite(PollConn* poll_conn_ptr);
  int SetToReadOnly(PollConn* poll_conn_ptr);
  int SetToWriteOnly(PollConn* poll_conn_ptr);
  int Remove(PollConn* poll_conn_ptr);
  void WaitAndHandleEvent();
  void HandleWriteEvent(PollConn* poll_conn_ptr);

  static constexpr int kNoAction = 0x0;
  static constexpr int kClearOneMsg = 0x1;
  static constexpr int kClearOneAndNextMsg = 0x2;
  static constexpr int kExit = 0x4;
};

template<typename PollConn>
EventHandler<PollConn>::EventHandler() {
  int ret = poll_.Init();
  CHECK_EQ(ret, 0) << "poll init failed";
}

template<typename PollConn>
void
EventHandler<PollConn>::SetConnectEventHandler(
    const std::function<void(PollConn*)>& connect_event_handler) {
  connect_event_handler_ = connect_event_handler;

}

template<typename PollConn>
void
EventHandler<PollConn>::SetReadEventHandler(
    const std::function<int(PollConn*)>& read_event_handler) {
  read_event_handler_ = read_event_handler;
}

template<typename PollConn>
void
EventHandler<PollConn>::SetClosedConnectionHandler(
    const std::function<int(PollConn*)>& closed_connection_handler) {
  closed_connection_handler_ = closed_connection_handler;
}

template<typename PollConn>
void
EventHandler<PollConn>::SetWriteEventHandler(
    const std::function<void(PollConn*)>& write_event_handler) {
  write_event_handler_ = write_event_handler;
}

template<typename PollConn>
void
EventHandler<PollConn>::SetDefaultWriteEventHandler() {
  write_event_handler_ = std::bind(&EventHandler<PollConn>::HandleWriteEvent,
                                   this, std::placeholders::_1);
}

template<typename PollConn>
int
EventHandler<PollConn>::SetToReadWrite(PollConn* poll_conn_ptr) {
  read_set_.emplace(poll_conn_ptr);
  int read_fd = poll_conn_ptr->get_read_fd();
  int write_fd = poll_conn_ptr->get_write_fd();
  if (read_fd == 0 || write_fd == 0) return -1;
  int ret = 0;
  if (read_fd == write_fd) {
    ret = poll_.Add(read_fd, poll_conn_ptr, EPOLLIN | EPOLLOUT);
    if (ret == 0) return ret;
    if (ret < 0 && errno == EEXIST)
      return poll_.Set(read_fd, poll_conn_ptr, EPOLLIN | EPOLLOUT);
    return ret;
  } else {
    ret = poll_.Add(read_fd, poll_conn_ptr, EPOLLIN);
    if (ret < 0 && errno == EEXIST)
      ret = poll_.Set(read_fd, poll_conn_ptr, EPOLLIN);
    if(ret < 0) return ret;

    ret = poll_.Add(write_fd, poll_conn_ptr, EPOLLOUT);
    if (ret < 0 && errno == EEXIST)
      ret = poll_.Set(write_fd, poll_conn_ptr, EPOLLOUT);
    return ret;
  }
}

template<typename PollConn>
int
EventHandler<PollConn>::SetToReadOnly(PollConn* poll_conn_ptr) {
  read_set_.emplace(poll_conn_ptr);
  int read_fd = poll_conn_ptr->get_read_fd();
  int write_fd = poll_conn_ptr->get_write_fd();
  if (read_fd == 0) return -1;
  int ret = 0;
  if (read_fd == write_fd) {
    ret = poll_.Add(read_fd, poll_conn_ptr, EPOLLIN);
    if (ret == 0) return ret;
    if (ret < 0 && errno == EEXIST)
      return poll_.Set(read_fd, poll_conn_ptr, EPOLLIN);
    return ret;
  } else {
    ret = poll_.Add(read_fd, poll_conn_ptr, EPOLLIN);
    if (ret < 0 && errno == EEXIST)
      ret = poll_.Set(read_fd, poll_conn_ptr, EPOLLIN);
    if(ret < 0) return ret;
    if (write_fd == 0) return 0;

    ret = poll_.Remove(write_fd);
    if (ret < 0 && errno == ENOENT)
      return 0;
    return ret;
  }
}

template<typename PollConn>
int
EventHandler<PollConn>::SetToWriteOnly(PollConn* poll_conn_ptr) {
  read_set_.erase(poll_conn_ptr);
  int read_fd = poll_conn_ptr->get_read_fd();
  int write_fd = poll_conn_ptr->get_write_fd();
  if (write_fd == 0) return -1;
  int ret = 0;
  if (read_fd == write_fd) {
    ret = poll_.Add(write_fd, poll_conn_ptr, EPOLLOUT);
    if (ret == 0) return ret;
    if (ret < 0 && errno == EEXIST)
      return poll_.Set(write_fd, poll_conn_ptr, EPOLLOUT);
    return ret;
  } else {
    ret = poll_.Add(write_fd, poll_conn_ptr, EPOLLOUT);
    if (ret < 0 && errno == EEXIST)
      ret = poll_.Set(write_fd, poll_conn_ptr, EPOLLOUT);
    if(ret < 0) return ret;

    if (read_fd == 0) return 0;
    ret = poll_.Remove(read_fd);
    if (ret < 0 && errno == ENOENT) return 0;

    return ret;
  }
}

template<typename PollConn>
int
EventHandler<PollConn>::Remove(PollConn* poll_conn_ptr) {
  read_set_.erase(poll_conn_ptr);
  int read_fd = poll_conn_ptr->get_read_fd();
  int write_fd = poll_conn_ptr->get_write_fd();
  int ret = 0;
  if (read_fd == write_fd) {
    ret = poll_.Remove(read_fd);
    return ret;
  } else {
    bool read_noent = false;
    ret = poll_.Remove(read_fd);
    if (ret < 0 && errno == ENOENT) read_noent = true;
    if (ret < 0 && errno != ENOENT) return ret;

    ret = poll_.Remove(write_fd);
    if (ret < 0 && errno != ENOENT) return ret;
    if (ret < 0 && errno == ENOENT && read_noent) return -1;
    return 0;
  }

}

template<typename PollConn>
bool
EventHandler<PollConn>::ReadAndRunReadEventHandler(PollConn* poll_conn_ptr) {
  auto &recv_buff = poll_conn_ptr->get_recv_buff();

  if (!recv_buff.IsExepectingNextMsg()) {
    bool recv = poll_conn_ptr->Receive();
    if (!recv) return false;
    if (recv_buff.is_eof()) return false;
    CHECK (!recv_buff.is_error()) << "error during receiving " << errno
                                  << " poll_conn_ptr = " << (void*) poll_conn_ptr;
    CHECK (!recv_buff.EOFAtIncompleteMsg()) << "error : early EOF";
  }
  return RunReadEventHandler(poll_conn_ptr);
}

template<typename PollConn>
bool
EventHandler<PollConn>::RunReadEventHandler(PollConn* poll_conn_ptr) {
  int clear_code = read_event_handler_(poll_conn_ptr);
  auto &recv_buff = poll_conn_ptr->get_recv_buff();
  if (clear_code & kClearOneMsg) {
    recv_buff.ClearOneMsg();
  } else if (clear_code & kClearOneAndNextMsg) {
    recv_buff.ClearOneAndNextMsg();
  }
  if (clear_code & kExit) return true;
  return false;
}

template<typename PollConn>
void
EventHandler<PollConn>::HandleWriteEvent(PollConn* poll_conn_ptr) {
  bool sent = poll_conn_ptr->Send();
  if (sent) {
    auto &send_buff = poll_conn_ptr->get_send_buff();
    send_buff.clear_to_send();
    SetToReadOnly(poll_conn_ptr);
  }
}

template<typename PollConn>
void
EventHandler<PollConn>::WaitAndHandleEvent() {
  bool exit = false;
  for (auto poll_conn_ptr : read_set_) {
    auto& recv_buff = poll_conn_ptr->get_recv_buff();
    while (recv_buff.ReceivedFullMsg()
           && (!recv_buff.IsExepectingNextMsg())) {
      exit = RunReadEventHandler(poll_conn_ptr);
      if (exit) break;
    }
    if (exit) break;
  }
  if (exit) return;

  int num_events = poll_.Wait(es_, kNumEvents);
  CHECK(num_events > 0);
  for (int i = 0; i < num_events; ++i) {
    PollConn *poll_conn_ptr = conn::Poll::EventConn<PollConn>(es_, i);
    if (es_[i].events & EPOLLHUP) {
      int clear_code = closed_connection_handler_(poll_conn_ptr);
      if (clear_code & kExit) break;
    } else if (es_[i].events & EPOLLIN) {
      if (poll_conn_ptr->is_connect_event()
          && connect_event_handler_ != nullptr) {
        connect_event_handler_(poll_conn_ptr);
      } else {
        exit = ReadAndRunReadEventHandler(poll_conn_ptr);
        if (exit) break;
        auto& recv_buff = poll_conn_ptr->get_recv_buff();
        //while (recv_buff.ReceivedFullMsg()
        //       && (!recv_buff.IsExepectingNextMsg())) {
        //  exit = RunReadEventHandler(poll_conn_ptr);
        //  if (exit) break;
        //}
        //if (exit) break;
        if (recv_buff.is_eof()) {
          int clear_code = closed_connection_handler_(poll_conn_ptr);
          if (clear_code & kExit) break;
        }
      }
    }

    if (es_[i].events & EPOLLOUT) {
      write_event_handler_(poll_conn_ptr);
    }
  }
}

}
}
