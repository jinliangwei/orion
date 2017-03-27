#include <string>
#include <memory>
#include <unordered_map>
#include <iostream>

namespace orion {
class GLogConfig {
 private:
  std::unordered_map<std::string, std::string> map_;
  std::string buffer_[7];
  char* argv_[7];
 public:
  GLogConfig(const char* progname) {
    buffer_[0] = std::string(progname);
    map_["logtostderr"] = "false";
    map_["minloglevel"] = "0";
    map_["v"] = "-1";
    map_["stderrthreshold"] = "2";
    map_["alsologtostderr"] = "false";
    map_["log_dir"] = "";
  }

    bool
    set(const char *key, const char*value) {
      auto iter = map_.find(key);
      if (iter == map_.end()) return false;
      iter->second = value;
      return true;
    }

    char **
    get_argv() {
      argv_[0] = const_cast<char*>(buffer_[0].c_str());
      int i = 1;
      for (auto& kv : map_) {
        buffer_[i].clear();
        buffer_[i].append("--");
        buffer_[i].append(kv.first);
        buffer_[i].append("=");
        buffer_[i].append(kv.second);
        buffer_[i].append(1, '\0');
        argv_[i] = const_cast<char*>(buffer_[i].c_str());
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
