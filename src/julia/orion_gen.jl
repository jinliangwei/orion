#include("/home/ubuntu/orion/src/julia/orion_worker.jl")

module OrionGen
using OrionWorker
function orion_define_setter(var::AbstractString)
    var_sym = Symbol(var)
    set_func_name = Symbol("orion_set_", var)
    set_func = :(function $set_func_name(val)
                   global $(var_sym) = val
                 end)
    eval(set_func)
end

end
