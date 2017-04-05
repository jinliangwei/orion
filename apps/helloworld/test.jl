push!(LOAD_PATH, "/home/ubuntu/orion/src/julia/")
import orion
import Orion

function parse_line(line::AbstractString)
    tokens = split(line, ',')
    @assert length(tokens) == 3
    key_tuple = (parse(Int64, String(tokens[1])),
                 parse(Int64, String(tokens[2])))
    value = parse(Float64, String(tokens[3]))
    return [(key_tuple, value)]
end

#Orion.set_lib_path("/home/ubuntu/orion/lib/liborion.so")
# initialize logging of the runtime library
#Orion.glog_init(C_NULL)
#Orion.init(master_ip, master_port, comm_buff_capacity)

Orion.Ast.parse_map_function(parse_line, (AbstractString,), true)
Orion.text_file("test", parse_line, (AbstractString,), true)
