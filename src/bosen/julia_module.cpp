#include <unordered_map>

#include <orion/bosen/julia_module.hpp>

namespace orion {
namespace bosen {
namespace {
jl_module_t *orion_generated_module = nullptr;
}

jl_module_t* GetJlModule(JuliaModule module) {
  switch (module) {
    case JuliaModule::kCore:
      return jl_core_module;
    case JuliaModule::kBase:
      return jl_base_module;
    case JuliaModule::kMain:
      return jl_main_module;
    case JuliaModule::kTop:
      return jl_top_module;
    case JuliaModule::kOrionGenerated:
      return orion_generated_module;
    default:
      return nullptr;
  }
}

void SetOrionGeneratedModule(jl_module_t* module) {
  orion_generated_module = module;
}

}
}
