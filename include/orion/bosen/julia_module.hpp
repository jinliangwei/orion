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
    kOrionGen = 5,
    kOrionWorker = 6
    };

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
    case JuliaModule::kOrionGen:
    case JuliaModule::kOrionWorker:
    default:
      return nullptr;
  }
}


void SetOrionGenModule(jl_module_t* module);
jl_module_t* GetOrionGenModule();
void SetOrionWorkerModule(jl_module_t* module);
jl_module_t* GetOrionWorkerModule();

}
}
