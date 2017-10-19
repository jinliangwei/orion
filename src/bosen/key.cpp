#include <orion/bosen/key.hpp>
#include <glog/logging.h>

namespace orion {
namespace bosen {
namespace key {

int64_t array_to_int64(const std::vector<int64_t> &dims,
                       const int64_t* key) {
  int64_t key_int = 0;
  int i = dims.size() - 1;
  for (; i > 0; i--) {
    key_int += key[i] - 1;
    key_int *= dims[i - 1];
  }

  key_int += key[0] - 1;

  return key_int;
}

int64_t vec_to_int64(const std::vector<int64_t> &dims,
                     const std::vector<int64_t> &key) {
  return array_to_int64(dims, key.data());
}

void int64_to_vec(const std::vector<int64_t> &dims,
                  int64_t key,
                  int64_t *key_vec) {
  for (int i = 0; i < dims.size(); i++) {
    key_vec[i] = key % dims[i] + 1;
    key /= dims[i];
  }
}

void int64_to_vec(const std::vector<int64_t> &dims,
                  int64_t key,
                  std::vector<int64_t> *key_vec) {
  int64_to_vec(dims, key, key_vec->data());
}
}
}
}
