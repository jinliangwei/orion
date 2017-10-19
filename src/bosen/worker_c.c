#include <orion/bosen/worker.h>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_partition.hpp>
#include <julia.h>
#include <iostream>
#include <vector>
#include <orion/bosen/type.hpp>

extern "C" {
  void orion_dist_array_read(void *dist_array_partition,
                             int64_t key_begin,
                             uint64_t num_elements,
                             void *array_mem) {
    CHECK(dist_array_partition != nullptr);
    auto *dist_array_partition_ptr = reinterpret_cast<
                                     orion::bosen::AbstractDistArrayPartition*>(dist_array_partition);
    dist_array_partition_ptr->ReadRange(key_begin, num_elements, array_mem);
  }

  void orion_dist_array_write(void *dist_array_partition,
                             int64_t key_begin,
                             uint64_t num_elements,
                             void *array_mem) {
    CHECK(dist_array_partition != nullptr);
    auto *dist_array_partition_ptr = reinterpret_cast<
                                     orion::bosen::AbstractDistArrayPartition*>(dist_array_partition);
    dist_array_partition_ptr->WriteRange(key_begin, num_elements, array_mem);
  }
}
