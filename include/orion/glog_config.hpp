#include <string>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <string.h>

namespace orion {
class GLogConfig {
 private:
  std::unordered_map<std::string, std::string> map_;
  std::vector<char> buffer_[7];
  char* argv_[7];
 public:
  GLogConfig(const char* progname) {
    buffer_[0].resize(strlen(progname) + 1);
    memcpy(buffer_[0].data(), progname, strlen(progname) + 1);
    map_["logtostderr"] = "false";
    map_["minloglevel"] = "0";
    map_["v"] = "-1";
    map_["stderrthreshold"] = "2";
    map_["alsologtostderr"] = "true";
    map_["log_dir"] = "/tmp";
  }

  bool
  set(const char *key, const char* value) {
    auto iter = map_.find(key);
    if (iter == map_.end()) return false;
    iter->second = value;
    return true;
  }

  void
  set_progname(const char* progname) {
    buffer_[0].resize(strlen(progname) + 1);
    memcpy(buffer_[0].data(), progname, strlen(progname) + 1);
  }

  char **
  get_argv() {
    argv_[0] = buffer_[0].data();
    int i = 1;
    for (auto& kv : map_) {
      buffer_[i].resize(kv.first.length() + kv.second.length() + 4);
      memcpy(buffer_[i].data(), "--", 2);
      memcpy(buffer_[i].data() + 2, kv.first.data(), kv.first.length());
      memcpy(buffer_[i].data() + 2 + kv.first.length(), "=", 1);
      memcpy(buffer_[i].data() + 3 + kv.first.length(),
             kv.second.c_str(), kv.second.length() + 1);
      argv_[i] = buffer_[i].data();
      i++;
    }
    return argv_;
  }

  int
  get_argc() const {
    return sizeof(argv_) / sizeof(char*);
  }
};
}
