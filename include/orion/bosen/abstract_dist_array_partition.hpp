#pragma once

#include <stdint.h>
#include <string>
#include <orion/bosen/blob.hpp>

namespace orion {
namespace bosen {

class AbstractDistArrayPartition {
 public:
  AbstractDistArrayPartition() { }
  virtual ~AbstractDistArrayPartition() { }

  virtual bool LoadTextFile(const std::string &path, int32_t partition_id,
                            const std::string &parser_func) = 0;

  virtual void Insert(uint64_t key, const Blob &buff) = 0;
  virtual void Get(uint64_t key, Blob *buff) = 0;
  virtual void GetRange(uint64_t start, uint64_t end, Blob *buff) = 0;
};

}
}
