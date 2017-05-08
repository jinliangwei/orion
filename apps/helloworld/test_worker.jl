include("/home/ubuntu/orion/src/julia/orion_gen.jl")
OrionGen.define(:c)
OrionGen.set_c(Dict{Int, Int}())
println(OrionGen.get_c())
