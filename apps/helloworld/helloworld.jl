push!(LOAD_PATH, "/home/ubuntu/orion/src/julia/")
import orion
import Orion

Orion.set_lib_path("/home/ubuntu/orion/lib/liborion.so")
Orion.helloworld()
Orion.glog_init()
Orion.init("127.0.0.1", 12000, 1024)

Orion.@share function orion_parse(line::AbstractString)
    tokens = split(line, ',')
    @assert length(tokens) == 3
    key_tuple = (parse(Int64, String(tokens[1])),
                 parse(Int64, String(tokens[2])))
    value = parse(Float64, String(tokens[3]))
    return (key_tuple, value)
end

Orion.@share function parse_line(line::AbstractString)
#    tokens = split(line, ',')
#    @assert length(tokens) == 3
#    key_tuple = (parse(Int64, String(tokens[1])),
#                 parse(Int64, String(tokens[2])))
#    value = parse(Float64, String(tokens[3]))
    #    return (key_tuple, value)
#    tokens = split(line, ',')
    return orion_parse(line)
end

ret = Orion.execute_code(0, "sqrt(2.0)", Float64)
println(ret)

#const data_path = "file:///home/ubuntu/data/ml-1m/ratings.csv"
#const data_path = "file:///home/ubuntu/data/netflix.list"
const data_path = "file:///home/ubuntu/data/ml-10M100K/ratings.csv"
const K = 100
const num_iterations = 1
const step_size = 0.001

rating = Orion.text_file(data_path, parse_line)
Orion.materialize(rating)

x_dim, y_dim = size(rating)

#W = Orion.rand(x_dim, K)
#H = Orion.rand(y_dim, K)

#Orion.materialize(W)
#Orion.materialize(H)

Orion.stop()
