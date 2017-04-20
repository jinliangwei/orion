#pragma once

#include <vector>
#include <stdint.h>

namespace orion {
namespace bosen {
namespace key {

inline int64_t array_to_int64(const std::vector<int64_t> &dims,
                              const int64_t* key) {
  int64_t key_int = 0;
  int i = 0;
  for (; i + 1 < dims.size(); i++) {
    key_int += key[i];
    key_int *= dims[i + 1];
  }

  key_int += key[i];

  return key_int;
}

inline int64_t vec_to_int64(const std::vector<int64_t> &dims,
                            const std::vector<int64_t> &key) {
  return array_to_int64(dims, key.data());
}

inline void int64_to_vec(const std::vector<int64_t> &dims,
                         int64_t key,
                         int64_t *key_vec) {
  for (int i = dims.size() - 1; i >= 0; i++) {
    key_vec[i] = key % dims[i];
    key /= dims[i];
  }
}

inline void int64_to_vec(const std::vector<int64_t> &dims,
                         int64_t key,
                         std::vector<int64_t> *key_vec) {
  int64_to_vec(dims, key, key_vec->data());
}
}
}
}
