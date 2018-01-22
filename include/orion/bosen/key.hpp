#pragma once

#include <vector>
#include <stdint.h>

namespace orion {
namespace bosen {
namespace key {

int64_t array_to_int64(const std::vector<int64_t> &dims,
                       const int64_t* key);

int64_t vec_to_int64(const std::vector<int64_t> &dims,
                     const std::vector<int64_t> &key);

void int64_to_vec(const std::vector<int64_t> &dims,
                  int64_t key,
                  int64_t *key_vec);

void int64_to_vec(const std::vector<int64_t> &dims,
                  int64_t key,
                  std::vector<int64_t> *key_vec);
}
}
}
