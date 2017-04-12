#pragma once

#include <julia.h>

namespace orion {
namespace bosen {

enum class JuliaModule {
  kNone = 0,
  kCore = 1,
    kBase = 2,
    kMain = 3,
    kTop = 4,
    };

jl_module_t* GetJlModule(JuliaModule module) {
  switch (module) {
    case JuliaModule::kNone:
      return nullptr;
    case JuliaModule::kCore:
      return jl_core_module;
    case JuliaModule::kBase:
      return jl_base_module;
    case JuliaModule::kMain:
      return jl_main_module;
    case JuliaModule::kTop:
      return jl_top_module;
    default:
      LOG(FATAL) << "unJuliaModule::known module type "
                 << static_cast<int>(module);
      return nullptr;
  }
}

}
}
