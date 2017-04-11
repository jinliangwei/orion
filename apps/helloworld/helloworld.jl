push!(LOAD_PATH, "/home/ubuntu/orion/src/julia/")
import orion
import Orion

Orion.set_lib_path("/home/ubuntu/orion/lib/liborion.so")
Orion.helloworld()
Orion.glog_init(C_NULL)
Orion.init("127.0.0.1", 12000, 1024)

Orion.@share function parse_line(line::AbstractString)
    tokens = split(line, ',')
    @assert length(tokens) == 3
    key_tuple = (parse(Int64, String(tokens[1])),
                 parse(Int64, String(tokens[2])))
    value = parse(Float64, String(tokens[3]))
    return (key_array, value_tuple)
end

ret = Orion.execute_code(0, "sqrt(2.0)", Float64)
println(ret)

Orion.stop()

ret = Orion.get_result_type_value(Float64)
println(ret)
