#include <orion/bosen/worker.h>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_partition.hpp>
#include <julia.h>
#include <iostream>
#include <vector>
#include <orion/bosen/type.hpp>

extern "C" {
  void orion_dist_array_read_dense(void *dist_array_access_ptr,
                                   int64_t key_begin,
                                   uint64_t num_elements,
                                   jl_value_t *array_buff) {
    CHECK(dist_array_access_ptr != nullptr);
    auto *dist_array_access = reinterpret_cast<
                              orion::bosen::DistArrayAccess*>(dist_array_access_ptr);

    dist_array_access->ReadRangeDense(key_begin,
                                      num_elements,
                                      array_buff);
  }

  void orion_dist_array_read_sparse(void *dist_array_access_ptr,
                                    int64_t key_begin,
                                    uint64_t num_elements,
                                    jl_value_t **key_array_buff,
                                    jl_value_t **value_array_buff) {
    CHECK(dist_array_access_ptr != nullptr);
    auto *dist_array_access = reinterpret_cast<
                              orion::bosen::DistArrayAccess*>(dist_array_access_ptr);
    dist_array_access->ReadRangeSparse(key_begin,
                                       num_elements,
                                       key_array_buff,
                                       value_array_buff);
  }

  void orion_dist_array_read_sparse_with_init_value(void *dist_array_access_ptr,
                                                    int64_t key_begin,
                                                    uint64_t num_elements,
                                                    jl_value_t *array_buff) {
    CHECK(dist_array_access_ptr != nullptr);
    auto *dist_array_access = reinterpret_cast<
                              orion::bosen::DistArrayAccess*>(dist_array_access_ptr);
    dist_array_access->ReadRangeSparseWithInitValue(
        key_begin,
        num_elements,
        array_buff);
  }

  void orion_dist_array_write(void *dist_array_access_ptr,
                              int64_t key_begin,
                              uint64_t num_elements,
                              jl_value_t *array_buff) {
    CHECK(dist_array_access_ptr != nullptr);
    auto *dist_array_access = reinterpret_cast<
                              orion::bosen::DistArrayAccess*>(dist_array_access_ptr);
    dist_array_access->WriteRange(key_begin,
                                  num_elements,
                                  array_buff);
  }
}
