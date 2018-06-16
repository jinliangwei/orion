#include <orion/bosen/worker.h>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_partition.hpp>
#include <orion/bosen/type.hpp>
#include <julia.h>
#include <iostream>
#include <vector>
#include <orion/bosen/type.hpp>

extern "C" {
  orion::bosen::JuliaThreadRequester *requester;
  void set_julia_thread_requester(orion::bosen::JuliaThreadRequester *_requester) {
    requester = _requester;
  }

  jl_value_t* orion_request_dist_array_data(
      int32_t dist_array_id,
      int64_t key,
      int32_t value_type) {
    return requester->RequestDistArrayData(dist_array_id, key,
                                           static_cast<orion::bosen::type::PrimitiveType>(value_type));
  }
}
