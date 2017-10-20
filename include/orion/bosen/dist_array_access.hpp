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
  void ReadRange(int32_t key_begin,
                 size_t num_elements,
                 void *mem);
  void WriteRange(int64_t key_begin,
                  size_t num_elements,
                  void *mem);
  //int32_t GetValueType();
};

}
}
