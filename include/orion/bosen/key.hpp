#pragma once

#include <vector>
#include <stdint.h>
#include <stddef.h>

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

int64_t int64_to_dim_key(const std::vector<int64_t> &dims,
                         int64_t key,
                         size_t dim_index);

void get_partial_dims(const std::vector<int64_t> &dims,
                      const std::vector<size_t> &dim_indices,
                      std::vector<int64_t> *partial_dims);

void get_partial_key(const std::vector<int64_t> &key_vec,
                      const std::vector<size_t> &dim_indices,
                      std::vector<int64_t> *partial_key_vec);

void update_key_with_partial_key(const std::vector<int64_t> &partial_key_vec,
                                 const std::vector<size_t> &dim_indices,
                                 std::vector<int64_t> *key_vec);

size_t get_length(const std::vector<int64_t> &dims);
}
}
}
