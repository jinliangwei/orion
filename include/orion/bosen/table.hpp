#pragma once

#include <string>
#include <orion/noncopyable.hpp>

namespace orion {
namespace bosen {

class TableConfig {
 public:
  enum class Type {
    kDense = 0,
      kSparseSeq = 1,
      kSparseRandom = 2,
      kPieceWiseSeq = 3,
      kPieceWiseRandom = 4
  };

 private:
  const Type type_;
  const size_t value_size_;
  const std::string data_path_;

 public:
  TableConfig(Type type,
              size_t item_size,
              const std::string &data_path):
      type_(type),
      value_size_(item_size),
      data_path_(data_path) { }

  TableConfig(const uint8_t *mem):
      type_(*reinterpret_cast<const Type*>(mem)),
      value_size_(*reinterpret_cast<const size_t*>(mem + sizeof(type_))),
      data_path_(*reinterpret_cast<const size_t*>(
          mem + sizeof(type_) + sizeof(value_size_)) == 0 ?
                 "" :
                 reinterpret_cast<const char*>(
                     mem + sizeof(type_)
                     + sizeof(value_size_)
                     + sizeof(size_t))) {  }

  ~TableConfig() { }

  const std::string &get_data_path() const {
    return data_path_;
  }

  size_t GetSerializedSize() const {
    return sizeof(type_)
        + sizeof(value_size_)
        + sizeof(size_t)
        + data_path_.length() + 1;
  }

  void Serialize(uint8_t *mem) const {
    size_t offset = 0;
    *reinterpret_cast<Type*>(mem + offset) = type_;
    offset += sizeof(type_);
    *reinterpret_cast<size_t*>(mem + offset)
        = value_size_;
    offset += sizeof(value_size_);
    *reinterpret_cast<size_t*>(mem + offset)
        = data_path_.empty() ? 0 : data_path_.length() + 1;
    offset += sizeof(size_t);
    if (!data_path_.empty())
      memcpy(mem + offset, data_path_.c_str(), data_path_.length() + 1);
  }
};

}
}
