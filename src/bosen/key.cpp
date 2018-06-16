#include <orion/bosen/key.hpp>
#include <glog/logging.h>

namespace orion {
namespace bosen {
namespace key {

int64_t array_to_int64(const std::vector<int64_t> &dims,
                       const int64_t* key) {
  int64_t key_int = 0;
  size_t i = dims.size() - 1;
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
  for (size_t i = 0; i < dims.size(); i++) {
    key_vec[i] = key % dims[i] + 1;
    key /= dims[i];
  }
}

void int64_to_vec(const std::vector<int64_t> &dims,
                  int64_t key,
                  std::vector<int64_t> *key_vec) {
  int64_to_vec(dims, key, key_vec->data());
}

int64_t int64_to_dim_key(const std::vector<int64_t> &dims,
                         int64_t key,
                         size_t dim_index) {
  for (size_t i = 0; i < dim_index; i++) {
    key /= dims[i];
  }
  return key % dims[dim_index] + 1;
}

void get_partial_dims(const std::vector<int64_t> &dims,
                      const std::vector<size_t> &dim_indices,
                      std::vector<int64_t> *partial_dims) {
  for (size_t i = 0; i < dim_indices.size(); i++) {
    size_t dim_index = dim_indices[i];
    (*partial_dims)[i] = dims[dim_index];
  }
}

void get_partial_key(const std::vector<int64_t> &key_vec,
                     const std::vector<size_t> &dim_indices,
                     std::vector<int64_t> *partial_key_vec) {
  for (size_t i = 0; i < dim_indices.size(); i++) {
    size_t dim_index = dim_indices[i];
    (*partial_key_vec)[i] = key_vec[dim_index];
  }
}

void update_key_with_partial_key(const std::vector<int64_t> &partial_key_vec,
                                 const std::vector<size_t> &dim_indices,
                                 std::vector<int64_t> *key_vec) {
  for (size_t i = 0; i < dim_indices.size(); i++) {
    size_t dim_index = dim_indices[i];
    (*key_vec)[dim_index] = partial_key_vec[i];
  }
}

size_t get_length(const std::vector<int64_t> &dims) {
  size_t length = dims[0];
  for (size_t i = 1; i < dims.size(); i++) {
    length *= dims[i];
  }

  return length;
}

}
}
}
