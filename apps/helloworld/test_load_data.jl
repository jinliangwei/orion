include("/home/ubuntu/orion/src/julia/orion.jl")

println("application started")

# set path to the C++ runtime library
Orion.set_lib_path("/home/ubuntu/orion/lib/liborion_driver.so")
# test library path
Orion.helloworld()

const master_ip = "127.0.0.1"
const master_port = 10000
const comm_buff_capacity = 1024
const num_executors = 2

# initialize logging of the runtime library
Orion.glog_init()
Orion.init(master_ip, master_port, comm_buff_capacity, num_executors)

const data_path = "file:///home/ubuntu/data/ml-1m/ratings.csv"
#const data_path = "file:///home/ubuntu/data/ml-10M100K/ratings.csv"
const K = 100
const num_iterations = 1
const step_size = 0.001

Orion.@share function parse_line(line::AbstractString)
    tokens = split(line, ',')
    @assert length(tokens) == 3
    key_tuple = (parse(Int64, String(tokens[1])),
                 parse(Int64, String(tokens[2])))
    value = parse(Float64, String(tokens[3]))
    return (key_tuple, value)
end


ratings = Orion.text_file(data_path, parse_line)
Orion.materialize(ratings)

Orion.define_var(:step_size)

partition_func_name = Orion.gen_unique_symbol()
partition_func = Orion.gen_space_time_partition_function(partition_func_name,
                                                         [(1,), (2,)], [(1,), (1,)],
                                                         100, 100)
Orion.eval_expr_on_all(partition_func, :OrionGen)
Orion.space_time_repartition(ratings, string(partition_func_name))

Orion.stop()
