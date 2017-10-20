#include <orion/bosen/worker.h>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_partition.hpp>
#include <julia.h>
#include <iostream>
#include <vector>
#include <orion/bosen/type.hpp>

extern "C" {
  void orion_dist_array_read(void *dist_array_access_ptr,
                             int64_t key_begin,
                             uint64_t num_elements,
                             void *array_mem) {
    CHECK(dist_array_access_ptr != nullptr);
    auto *dist_array_access = reinterpret_cast<
                              orion::bosen::DistArrayAccess*>(dist_array_access_ptr);
    dist_array_access->ReadRange(key_begin, num_elements, array_mem);
  }

  void orion_dist_array_write(void *dist_array_access_ptr,
                             int64_t key_begin,
                             uint64_t num_elements,
                             void *array_mem) {
    CHECK(dist_array_access_ptr != nullptr);
    auto *dist_array_access = reinterpret_cast<
                              orion::bosen::DistArrayAccess*>(dist_array_access_ptr);
    dist_array_access->WriteRange(key_begin, num_elements, array_mem);
  }
}
