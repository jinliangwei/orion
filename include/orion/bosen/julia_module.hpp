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
    kOrionGenerated = 5
    };

jl_module_t* GetJlModule(JuliaModule module);

void SetOrionGeneratedModule(jl_module_t* module);

}
}
