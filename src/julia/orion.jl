module Orion

export set_lib_path

function set_lib_path(path::AbstractString)
    global const lib_path = path
end
include("share.jl")
include("function_parser.jl")
include("driver.jl")
end
