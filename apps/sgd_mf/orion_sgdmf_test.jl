include("/users/jinlianw/orion.git/src/julia/orion.jl")

println("application started")

# set path to the C++ runtime library
Orion.set_lib_path("/users/jinlianw/orion.git/lib/liborion_driver.so")
# test library path
Orion.helloworld()

const master_ip = "127.0.0.1"
const master_port = 10000
const comm_buff_capacity = 1024
const num_executors = 4

# initialize logging of the runtime library
Orion.glog_init()
Orion.init(master_ip, master_port, comm_buff_capacity, num_executors)

#const data_path = "file:///home/ubuntu/data/ml-1m/ratings.csv"
#const data_path = "file:///home/ubuntu/data/ml-10M100K/ratings.csv"
const data_path = "file:///users/jinlianw/ratings.csv"
const K = 100
const num_iterations = 1
const step_size = 0.001

Orion.@share function parse_line(line::AbstractString)
    tokens = split(line, ',')
    @assert length(tokens) == 3
    key_tuple = (parse(Int64, String(tokens[1])) + 1,
                 parse(Int64, String(tokens[2])) + 1)
    value = parse(Float64, String(tokens[3]))
    return (key_tuple, value)
end

Orion.@share function map_init_param(value::Float32)::Float32
    return value / 10
end

ratings = Orion.text_file(data_path, parse_line)
Orion.materialize(ratings)
dim_x, dim_y = size(ratings)

println((dim_x, dim_y))

loop_partition_func_name = Orion.gen_unique_symbol()
loop_partition_func = Orion.gen_2d_partition_function(loop_partition_func_name,
                                                      1,
                                                      2,
                                                      100,
                                                      100)

Orion.eval_expr_on_all(loop_partition_func, :Main)
dist_array_partition_info = Orion.DistArrayPartitionInfo(Orion.DistArrayPartitionType_2d,
                                                         loop_partition_func_name,
                                                         (1, 2),
                                                         Orion.DistArrayIndexType_none)
Orion.check_and_repartition(ratings, dist_array_partition_info)

param_partition_func_name = Orion.gen_unique_symbol()
param_partition_func = Orion.gen_1d_partition_function(param_partition_func_name,
                                                       1,
                                                       100)
Orion.eval_expr_on_all(param_partition_func, :Main)


dist_array_partition_info = Orion.DistArrayPartitionInfo(Orion.DistArrayPartitionType_1d,
                                                         param_partition_func_name,
                                                         (1,),
                                                         Orion.DistArrayIndexType_local)


W = Orion.randn(dim_x, K)
#W = Orion.map_value(W, map_init_param)
Orion.materialize(W)


H = Orion.randn(dim_y, K)
#H = Orion.map_value(H, map_init_param)
Orion.materialize(H)

Orion.check_and_repartition(W, dist_array_partition_info)
Orion.check_and_repartition(H, dist_array_partition_info)

#H.save_as_text_file("/home/ubuntu/model/H")
#W.save_as_text_file("/home/ubuntu/model/W")

Orion.stop()
