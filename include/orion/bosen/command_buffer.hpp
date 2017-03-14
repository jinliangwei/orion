#pragma once
#include <string>

namespace orion {
namespace bosen {

class CommandBuffer {
 private:
  std::string buff_;
 public:
  CommandBuffer();
  ~CommandBuffer();
  void Reset(const std::string &cmd);
  const std::string& Get();
};

}
}
