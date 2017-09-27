#include <orion/bosen/worker.h>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_partition.hpp>
#include <julia.h>
#include <iostream>
#include <vector>
#include <orion/bosen/type.hpp>

std::vector<int64_t> dims = {5, 10};

extern "C" {
  jl_value_t *orion_dist_array_get_dims(void *dist_array_partition) {
    if (dist_array_partition == nullptr) {
      std::cout << __func__ << std::endl;
      jl_value_t *dim_array_type = jl_apply_array_type(jl_int64_type, 1);
      auto &dims_vec = dims;
      jl_value_t *result_array = nullptr;
      JL_GC_PUSH1(&result_array);
      result_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(dim_array_type,
                                                                     dims_vec.data(),
                                                                     dims_vec.size(), 0));
      JL_GC_POP();
      return result_array;
    }

    auto *dist_array_partition_ptr = reinterpret_cast<
                                     orion::bosen::AbstractDistArrayPartition*>(dist_array_partition);
    jl_value_t *dim_array_type = jl_apply_array_type(jl_int64_type, 1);
    auto &dims_vec = dist_array_partition_ptr->GetDims();
    jl_value_t *result_array = nullptr;
    JL_GC_PUSH1(&result_array);
    result_array = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(dim_array_type,
                                                                    dims_vec.data(),
                                                                    dims_vec.size(), 0));
    JL_GC_POP();
    return result_array;
  }

  void orion_dist_array_read(void *dist_array_partition,
                             int64_t key_begin,
                             uint64_t num_elements,
                             void *array_mem) {
    if (dist_array_partition == nullptr) {
      std::cout << __func__ << " key_begin = " << key_begin << " num_elements = " << num_elements
                << std::endl;
      int64_t *array = reinterpret_cast<int64_t*>(array_mem);
      for (size_t i = 0; i < num_elements; i++) {
        array[i] = i;
      }
      return;
    }
    auto *dist_array_partition_ptr = reinterpret_cast<
                                     orion::bosen::AbstractDistArrayPartition*>(dist_array_partition);
    dist_array_partition_ptr->ReadRange(key_begin, num_elements, array_mem);
  }

  void orion_dist_array_write(void *dist_array_partition,
                             int64_t key_begin,
                             uint64_t num_elements,
                             void *array_mem) {
    if (dist_array_partition == nullptr) {
      std::cout << __func__ << " key_begin = " << key_begin << " num_elements = " << num_elements
                << std::endl;
      return;
    }
    auto *dist_array_partition_ptr = reinterpret_cast<
                                     orion::bosen::AbstractDistArrayPartition*>(dist_array_partition);
    dist_array_partition_ptr->WriteRange(key_begin, num_elements, array_mem);
  }

  int32_t orion_dist_array_get_value_type(void *dist_array_partition) {
    if (dist_array_partition == nullptr) return static_cast<int32_t>(orion::bosen::type::PrimitiveType::kInt64);
    auto *dist_array_partition_ptr = reinterpret_cast<
                                     orion::bosen::AbstractDistArrayPartition*>(dist_array_partition);
    int32_t value_type = static_cast<int32_t>(dist_array_partition_ptr->GetValueType());
    return value_type;
  }
}
