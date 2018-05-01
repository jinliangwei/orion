#ifndef __WORKER_H__
#define __WORKER_H__

#include <orion/bosen/constants.h>
#include <orion/bosen/julia_thread_requester.hpp>
#include <stdlib.h>
#include <julia.h>
extern "C" {
  extern orion::bosen::JuliaThreadRequester *requester;
  void set_julia_thread_requester(orion::bosen::JuliaThreadRequester *_requester);
  void orion_request_dist_array_data(
      int32_t dist_array_id,
      int64_t key,
      int32_t value_type,
      jl_value_t *value_vec);
}

#endif
