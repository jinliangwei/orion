#pragma once

#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <unordered_map>
#include <julia.h>

namespace orion {
namespace bosen {

class DistArrayAccess {
 private:
  AbstractDistArrayPartition* access_partition_ {nullptr};
 public:
  DistArrayAccess();
  ~DistArrayAccess() { }

  void SetAccessPartition(AbstractDistArrayPartition *access_partition);
  AbstractDistArrayPartition* GetAccessPartition();

  jl_value_t* GetDims();
  void ReadRangeDense(int32_t key_begin,
                      size_t num_elements,
                      jl_value_t *array_buff);

  void ReadRangeSparse(int32_t key_begin,
                       size_t num_elements,
                       jl_value_t **key_array_buff,
                       jl_value_t **value_array_buff);

  void ReadRangeSparseWithInitValue(int32_t key_begin,
                                    size_t num_elements,
                                    jl_value_t *array_buff);

  void WriteRange(int64_t key_begin,
                  size_t num_elements,
                  jl_value_t *array_buff);
  //int32_t GetValueType();
};

}
}
