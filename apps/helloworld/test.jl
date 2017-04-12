push!(LOAD_PATH, "/home/ubuntu/orion/src/julia/")
import orion
import Orion

macro print_ast(func)
    println(func)
    println(func.head)
    call_head = func.args[1].head
    call_args = func.args[1].args
    println(call_args[2].head)
    esc(func)
end

@print_ast function parse_line(line::AbstractString)
    tokens = split(line, ',')
    @assert length(tokens) == 3
    key_tuple = (parse(Int64, String(tokens[1])),
                 parse(Int64, String(tokens[2])))
    value = parse(Float64, String(tokens[3]))
    return (key_tuple, value)
end

#Orion.set_lib_path("/home/ubuntu/orion/lib/liborion.so")
# initialize logging of the runtime library
#Orion.glog_init(C_NULL)
#Orion.init(master_ip, master_port, comm_buff_capacity)

Orion.Ast.parse_map_function(parse_line, (String,))
#Orion.text_file("test", parse_line, (AbstractString,), true)
#Orion.Ast.test_sugar(parse_line, (AbstractString,) )

#Orion.test()
