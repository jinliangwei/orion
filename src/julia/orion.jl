module Orion

#using MacroTools
#using Sugar

#const MacroParserError = "MacroParserError"

include("scope_context.jl")
include("dist_array.jl")
include("driver.jl")

include("ast.jl")
include("parse_function.jl")
include("static_parallelization.jl")
include("get_scope_context.jl")
include("symbol_table.jl")
include("variable_scope.jl")
include("codegen.jl")
include("debug.jl")
include("ast_walk.jl")
include("evaluate.jl")
include("constants.jl")
include("macros.jl")
include("codegen_transform.jl")
end
