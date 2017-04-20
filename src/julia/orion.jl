module Orion

module OrionGenerated
end
using MacroTools
using Sugar

export set_lib_path

function set_lib_path(path::AbstractString)
    global const lib_path = path
end
include("parse_ast.jl")
include("parse_function.jl")
include("dist_array.jl")
include("driver.jl")
include("dependence_analysis.jl")
include("translate_stmt.jl")
end
