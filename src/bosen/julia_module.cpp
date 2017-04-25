#include <unordered_map>

#include <orion/bosen/julia_module.hpp>

namespace orion {
namespace bosen {
namespace {
jl_module_t *orion_gen_module = nullptr;
jl_module_t *orion_worker_module = nullptr;
}

void SetOrionGenModule(jl_module_t* module) {
  orion_gen_module = module;
}

jl_module_t* GetOrionGenModule() {
  return orion_gen_module;
}

void SetOrionWorkerModule(jl_module_t* module) {
  orion_worker_module = module;
}

jl_module_t* GetOrionWorkerModule() {
  return orion_worker_module;
}


}
}
