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
    kOrionWorker = 5
          };

void SetOrionWorkerModule(jl_module_t* module);
jl_module_t* GetOrionWorkerModule();

inline jl_module_t* GetJlModule(JuliaModule module) {
  switch (module) {
    case JuliaModule::kCore:
      return jl_core_module;
    case JuliaModule::kBase:
      return jl_base_module;
    case JuliaModule::kMain:
      return jl_main_module;
    case JuliaModule::kTop:
      return jl_top_module;
    case JuliaModule::kOrionWorker:
      return GetOrionWorkerModule();
    default:
      return nullptr;
  }
}
}
}
