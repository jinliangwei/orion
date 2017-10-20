#ifndef __WORKER_H__
#define __WORKER_H__

#include <orion/bosen/constants.h>
#include <stdlib.h>
#include <julia.h>
extern "C" {
  void orion_dist_array_read(void *dist_array_access_ptr,
                             int64_t key_begin,
                             uint64_t num_elements,
                             void *array_mem);

  void orion_dist_array_write(void *dist_array_access_ptr,
                             int64_t key_begin,
                             uint64_t num_elements,
                             void *array_mem);
}

#endif
