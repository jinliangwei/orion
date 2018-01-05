#ifndef __WORKER_H__
#define __WORKER_H__

#include <orion/bosen/constants.h>
#include <stdlib.h>
#include <julia.h>
extern "C" {
  void orion_dist_array_read_dense(void *dist_array_access_ptr,
                                   int64_t key_begin,
                                   uint64_t num_elements,
                                   jl_value_t *array_buff);

  void orion_dist_array_read_sparse(void *dist_array_access_ptr,
                                    int64_t key_begin,
                                    uint64_t num_elements,
                                    jl_value_t **key_array_buff,
                                    jl_value_t **value_array_buff);

  void orion_dist_array_read_sparse_with_init_value(void *dist_array_access_ptr,
                                                    int64_t key_begin,
                                                    uint64_t num_elements,
                                                    jl_value_t *array_buff);

  void orion_dist_array_write(void *dist_array_access_ptr,
                              int64_t key_begin,
                              uint64_t num_elements,
                              jl_value_t *array_buff);
}

#endif
