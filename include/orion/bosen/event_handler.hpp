#include <orion/bosen/conn.hpp>
#include <functional>

namespace orion {
namespace bosen {

template<typename PollConn>
class EventHandler {
 private:
  static constexpr size_t kNumEvents = 100;
  conn::Poll poll_;
  epoll_event es_[kNumEvents];
  std::function<void(PollConn*)> noread_event_handler_;
  std::function<int(PollConn*)> read_event_handler_;
  std::function<void(PollConn*)> closed_connection_handler_;
  void ReadAndRunReadEventHandler(PollConn* poll_conn);
  void RunReadEventHandler(PollConn* poll_conn);
 public:
  EventHandler();
  void SetNoreadEventHandler(const std::function<void(PollConn*)>&
                             noread_event_handler);
  void SetReadEventHandler(const std::function<int(PollConn*)>&
                           read_event_handler);
  void SetClosedConnectionHandler(const std::function<void(PollConn*)>&
                                  closed_connection_handler);
  int AddPollConn(int fd, PollConn* poll_conn);
  int RemovePollConn(int fd);
  void WaitAndHandleEvent();
};

template<typename PollConn>
EventHandler<PollConn>::EventHandler() {
  int ret = poll_.Init();
  CHECK_EQ(ret, 0) << "poll init failed";
}

template<typename PollConn>
void
EventHandler<PollConn>::SetNoreadEventHandler(
    const std::function<void(PollConn*)>& noread_event_handler) {
  noread_event_handler_ = noread_event_handler;

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
    const std::function<void(PollConn*)>& closed_connection_handler) {
  closed_connection_handler_ = closed_connection_handler;
}

template<typename PollConn>
int
EventHandler<PollConn>::AddPollConn(int fd, PollConn* poll_conn_ptr) {
  return poll_.Add(fd, poll_conn_ptr);
}

template<typename PollConn>
int
EventHandler<PollConn>::RemovePollConn(int fd) {
  return poll_.Remove(fd);
}

template<typename PollConn>
void
EventHandler<PollConn>::ReadAndRunReadEventHandler(PollConn* poll_conn_ptr) {
  auto &recv_buff = poll_conn_ptr->get_recv_buff();

  if (!recv_buff.IsExepectingNextMsg()) {
    bool recv = poll_conn_ptr->Receive();
    if (!recv) return;

    CHECK (!recv_buff.is_error()) << "driver error during receiving " << errno;
    CHECK (!recv_buff.EOFAtIncompleteMsg()) << "driver error : early EOF";
    // maybe EOF but not received anything
    if (!recv_buff.ReceivedFullMsg()) return;
  }
  RunReadEventHandler(poll_conn_ptr);
}

template<typename PollConn>
void
EventHandler<PollConn>::RunReadEventHandler(PollConn* poll_conn_ptr) {
  int clear_code = read_event_handler_(poll_conn_ptr);
  auto &recv_buff = poll_conn_ptr->get_recv_buff();
  if (clear_code == 1) {
    recv_buff.ClearOneMsg();
  } else if (clear_code == 2) {
    recv_buff.ClearOneAndNextMsg();
  }

}

template<typename PollConn>
void
EventHandler<PollConn>::WaitAndHandleEvent() {
  int num_events = poll_.Wait(es_, kNumEvents);
  CHECK(num_events > 0);
  for (int i = 0; i < num_events; ++i) {
    PollConn *poll_conn_ptr = conn::Poll::EventConn<PollConn>(es_, i);
    if (es_[i].events & EPOLLIN) {
      if (poll_conn_ptr->is_noread_event()
          && noread_event_handler_ != nullptr) {
        noread_event_handler_(poll_conn_ptr);
      } else {
        ReadAndRunReadEventHandler(poll_conn_ptr);
        auto& recv_buff = poll_conn_ptr->get_recv_buff();
        while (recv_buff.ReceivedFullMsg()
               && (!recv_buff.IsExepectingNextMsg())) {
          RunReadEventHandler(poll_conn_ptr);
        }
        if (recv_buff.is_eof()) {
          LOG(INFO) << "someone has closed";
          closed_connection_handler_(poll_conn_ptr);
        }
      }
    } else {
      LOG(WARNING) << "unknown event happened happend: " << es_[i].events;
    }
  }
}

}
}
