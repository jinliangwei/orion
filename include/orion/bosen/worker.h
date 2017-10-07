#ifndef __WORKER_H__
#define __WORKER_H__

#include <orion/bosen/constants.h>
#include <stdlib.h>
#include <julia.h>
extern "C" {

  jl_value_t* orion_dist_array_get_dims(void *dist_array_partition);

  void orion_dist_array_read(void *dist_array_partition,
                             int64_t key_begin,
                             uint64_t num_elements,
                             void *array_mem);

  void orion_dist_array_write(void *dist_array_partition,
                             int64_t key_begin,
                             uint64_t num_elements,
                             void *array_mem);
  int32_t orion_dist_array_get_value_type(void *dist_array_partition);

}

#endif
