module Orion

include("scope_context.jl")
include("dist_array_accessor.jl")
include("dist_array.jl")
include("dist_array_buffer.jl")
include("dist_array_partition.jl")
include("driver.jl")

include("ast.jl")
include("parse_function.jl")
include("get_scope_context.jl")
include("variable_scope.jl")
include("debug.jl")
include("ast_walk.jl")
include("constants.jl")
include("flow_analysis.jl")
include("macros.jl")
include("dist_array_access.jl")
include("static_parallelization.jl")
include("codegen.jl")
end
