#pragma once
#include <orion/bosen/conn.hpp>
#include <orion/bosen/byte_buffer.hpp>

namespace orion {
namespace bosen {

template<typename Conn>
bool
ReceiveArbitraryBytes(Conn &conn, conn::RecvBuffer* recv_buff,
                      ByteBuffer* byte_buff, size_t expected_size) {
  CHECK(static_cast<int64_t>(expected_size) > 0)
      << "got negative expected size = " << (int64_t) expected_size;
  if (recv_buff->IsExepectingNextMsg()) {
    LOG(INFO) << "expecting next message";
    bool recv = conn.Recv(recv_buff, byte_buff->GetAvailMem());
    LOG(INFO) << "Recv returned with " << recv
              << " IncSize by "
              << recv_buff->get_next_recved_size() - byte_buff->GetSize();
    byte_buff->IncSize(
        recv_buff->get_next_recved_size() - byte_buff->GetSize());
    if (!recv) return false;
    CHECK (!recv_buff->is_error()) << "error during receiving "
                                  << errno;
    CHECK (!recv_buff->EOFAtIncompleteMsg()) << "error : early EOF";
  } else {
    LOG(INFO) << "wasn't expecting next message ";
    recv_buff->set_next_expected_size(expected_size);
    byte_buff->Reset(expected_size);
    if (recv_buff->get_size() > recv_buff->get_expected_size()) {
      size_t size_to_copy = recv_buff->get_size()
                            - recv_buff->get_expected_size();
      LOG(INFO) << "size_to_copy = "
                << recv_buff->get_size() - recv_buff->get_expected_size();
      memcpy(byte_buff->GetAvailMem(),
             recv_buff->get_curr_msg_end_mem(),
             size_to_copy);
      byte_buff->IncSize(size_to_copy);
      recv_buff->inc_next_recved_size(size_to_copy);
    }
  }
  if (recv_buff->ReceivedFullNextMsg()) return true;
  else return false;
}

template<typename Conn>
bool
ReceiveArbitraryBytes(Conn &conn, conn::RecvBuffer* recv_buff,
                      uint8_t* byte_buff, size_t *recved_size,
                      size_t expected_size) {
  if (recv_buff->IsExepectingNextMsg()) {
    bool recv = conn.Recv(recv_buff, byte_buff + *recved_size);
    *recved_size = recv_buff->get_next_recved_size();
    if (!recv) return false;
    CHECK (!recv_buff->is_error()) << "error during receiving "
                                  << errno;
    CHECK (!recv_buff->EOFAtIncompleteMsg()) << "error : early EOF";
  } else {
    recv_buff->set_next_expected_size(expected_size);
    if (recv_buff->get_size() > recv_buff->get_expected_size()) {
      size_t size_to_copy = recv_buff->get_size()
                            - recv_buff->get_expected_size();
      memcpy(byte_buff,
             recv_buff->get_curr_msg_end_mem(),
             size_to_copy);
      *recved_size = size_to_copy;
      recv_buff->inc_next_recved_size(size_to_copy);
    }
  }
  if (recv_buff->ReceivedFullNextMsg()) return true;
  else return false;
}

}
}
