module Orion

module OrionGen
end
using MacroTools
using Sugar

export set_lib_path

function set_lib_path(path::AbstractString)
    global const lib_path = path
end

const MacroParserError = "MacroParserError"

include("dist_array.jl")
include("parse_ast.jl")
include("parse_function.jl")
include("driver.jl")
include("transform.jl")
include("translate_stmt.jl")
include("codegen.jl")
end
