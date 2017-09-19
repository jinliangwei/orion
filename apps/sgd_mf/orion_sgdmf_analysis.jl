include("/home/ubuntu/orion/src/julia/orion.jl")

const K = 100
const num_iterations = 1
const step_size = 0.001
const factor = 1

# set path to the C++ runtime library
Orion.set_lib_path("/home/ubuntu/orion/lib/liborion_driver.so")
# test library path
Orion.helloworld()

ratings = Orion.DistArray{Float32}()
W = Orion.DistArray{Float32}()
H = Orion.DistArray{Float32}()

Orion.@share function parse_line(line::AbstractString)
    tokens = split(line, ',')
    @assert length(tokens) == 3
    key_tuple = (parse(Int64, String(tokens[1])),
                 parse(Int64, String(tokens[2])))
    value = parse(Float64, String(tokens[3]))
    return (key_tuple, value)
end


#ratings = Orion.text_file(data_path, parse_line)
