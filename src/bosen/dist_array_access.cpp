#include <orion/bosen/dist_array_access.hpp>
#include <julia.h>
#include <iostream>
#include <vector>
#include <orion/bosen/type.hpp>

namespace orion {
namespace bosen {
DistArrayAccess::DistArrayAccess() { }

void
DistArrayAccess::SetAccessPartition(
    AbstractDistArrayPartition *access_partition) {
  access_partition_ = access_partition;
}

AbstractDistArrayPartition*
DistArrayAccess::GetAccessPartition() {
  return access_partition_;
}

jl_value_t*
DistArrayAccess::GetDims() {
  CHECK(access_partition_ != nullptr);
  jl_value_t *dim_array_type = jl_apply_array_type(jl_int64_type, 1);

  auto& dims_vec = access_partition_->GetDims();
  auto* result_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(dim_array_type,
                                                                  dims_vec.data(),
                                                                  dims_vec.size(), 0));
  return result_array;
}

void
DistArrayAccess::ReadRange(int32_t key_begin,
                           size_t num_elements,
                           void *mem) {
  CHECK(access_partition_ != nullptr);
  access_partition_->ReadRange(key_begin, num_elements, mem);
}

void
DistArrayAccess::WriteRange(int64_t key_begin,
                            size_t num_elements,
                       void *mem) {
  CHECK(access_partition_ != nullptr);
  access_partition_->WriteRange(key_begin, num_elements, mem);
}

}
}
