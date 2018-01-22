include("/users/jinlianw/orion.git/src/julia/orion.jl")

println("application started")

# set path to the C++ runtime library
Orion.set_lib_path("/users/jinlianw/orion.git/lib/liborion_driver.so")
# test library path
Orion.helloworld()

const master_ip = "127.0.0.1"
#const master_ip = "10.117.1.28"
const master_port = 10000
const comm_buff_capacity = 1024
const num_executors = 1
const num_servers = 1

# initialize logging of the runtime library
Orion.glog_init()
Orion.init(master_ip, master_port, comm_buff_capacity,
           num_executors, num_servers)

#const data_path = "file:///home/ubuntu/data/ml-1m/ratings.csv"
#const data_path = "file:///home/ubuntu/data/ml-10M100K/ratings.csv"
#const data_path = "file:///users/jinlianw/ratings.csv"
const data_path = "file:///proj/BigLearning/jinlianw/data/netflix.csv"
#const data_path = "file:///proj/BigLearning/jinlianw/data/ml-10M100K/ratings.csv"
const K = 1000
const num_iterations = 1
const step_size = 1e-2

@Orion.accumulator line_cnt = 0
@Orion.accumulator cnt = 0
@Orion.accumulator err = 0

Orion.@share function parse_line(line::AbstractString)
    global line_cnt
    line_cnt += 1
    tokens = split(line, ',')
    @assert length(tokens) == 3
    key_tuple = (parse(Int64, String(tokens[1])) + 1,
                 parse(Int64, String(tokens[2])) + 1)
    value = parse(Float32, String(tokens[3]))
    return (key_tuple, value)
end

Orion.@share function map_init_param(value::Float32)::Float32
    return value / 10
end

x_tile_size = 30000
y_tile_size = 3000

#x_tile_size = 1000
#y_tile_size = 500

@Orion.dist_array ratings = Orion.text_file(data_path, parse_line)
Orion.materialize(ratings)
dim_x, dim_y = size(ratings)
println((dim_x, dim_y))

loop_partition_func_name = Orion.gen_unique_symbol()
loop_partition_func = Orion.gen_2d_partition_function(loop_partition_func_name,
                                                      1,
                                                      2,
                                                      x_tile_size,
                                                      y_tile_size)
println("to define loop partition func")
Orion.eval_expr_on_all(loop_partition_func, :Main)
dist_array_partition_info = Orion.DistArrayPartitionInfo(Orion.DistArrayPartitionType_2d,
                                                         string(loop_partition_func_name),
                                                         (1, 2),
                                                         (x_tile_size, y_tile_size),
                                                         Orion.DistArrayIndexType_none)
Orion.check_and_repartition(ratings, dist_array_partition_info)

#Orion.stop()
#exit()

line_cnt = Orion.get_aggregated_value(:line_cnt, :+)
println("line_cnt = ", line_cnt)

W_param_partition_func_name = Orion.gen_unique_symbol()
W_param_partition_func = Orion.gen_1d_partition_function(W_param_partition_func_name,
                                                         2,
                                                         x_tile_size)
println("to define W param partition func")
Orion.eval_expr_on_all(W_param_partition_func, :Main)

H_param_partition_func_name = Orion.gen_unique_symbol()
H_param_partition_func = Orion.gen_1d_partition_function(H_param_partition_func_name,
                                                         2,
                                                         y_tile_size)
println("to define H param partition func")
Orion.eval_expr_on_all(H_param_partition_func, :Main)


W_dist_array_partition_info = Orion.DistArrayPartitionInfo(Orion.DistArrayPartitionType_1d,
                                                           string(W_param_partition_func_name),
                                                           (2,),
                                                           (x_tile_size,),
                                                           Orion.DistArrayIndexType_none)

H_dist_array_partition_info = Orion.DistArrayPartitionInfo(Orion.DistArrayPartitionType_1d,
                                                           string(H_param_partition_func_name),
                                                           (2,),
                                                           (y_tile_size,),
                                                           Orion.DistArrayIndexType_none)

@Orion.dist_array W = Orion.randn(K, dim_x)
@Orion.dist_array W = Orion.map(W, map_init_param, map_values = true)
Orion.materialize(W)


@Orion.dist_array H = Orion.randn(K, dim_y)
@Orion.dist_array H = Orion.map(H, map_init_param, map_values = true)
Orion.materialize(H)

Orion.check_and_repartition(W, W_dist_array_partition_info)
Orion.check_and_repartition(H, H_dist_array_partition_info)

#Orion.stop()
#exit()

println("to define function iteration_func")

@Orion.share function iteration_func(rating)
    global cnt
    x_idx = rating[1][1]
    y_idx = rating[1][2]
    rv = rating[2]

    W_row = W[:, x_idx]
    H_row = H[:, y_idx]
    pred = dot(W_row, H_row)
    diff = rv - pred
    W_grad = -2 * diff .* H_row
    H_grad = -2 * diff .* W_row
    W[:, x_idx] = W_row - step_size .* W_grad
    H[:, y_idx] = H_row - step_size .* H_grad
    cnt += 1
end

println("to define function loop_batch_func")

@Orion.share function loop_batch_func(keys::Vector{Int64},
                                      values::Vector{Float32},
                                      dims::Vector{Int64})
    for i in 1:length(keys)
        key = keys[i]
        value = values[i]
        dim_keys = OrionWorker.from_int64_to_keys(key, dims)
        key_value = (dim_keys, value)
        iteration_func(key_value)
    end
end

@Orion.share function eval_func(rating)
    global err
    x_idx = rating[1][1]
    y_idx = rating[1][2]
    rv = rating[2]

    W_row = W[:, x_idx]
    H_row = H[:, y_idx]
    pred = dot(W_row, H_row)
    diff = rv - pred
    err += diff ^ 2
end

@Orion.share function eval_batch_func(keys::Vector{Int64},
                                      values::Vector{Float32},
                                      dims::Vector{Int64})
    for i in 1:length(keys)
        key = keys[i]
        value = values[i]
        dim_keys = OrionWorker.from_int64_to_keys(key, dims)
        key_value = (dim_keys, value)
        eval_func(key_value)
    end
end

println("to define global variable")

Orion.define_vars(Set([:step_size]))

error_vec = Vector{Float64}()

@time for iteration = 1:num_iterations
    println("iteration ", iteration)
    @time Orion.exec_for_loop(ratings.id,
                              Orion.ForLoopParallelScheme_2d,
                              [W.id], [H.id],
                              Vector{Int32}(),
                              Vector{Int32}(),
                              Vector{Int32}(),
                              Vector{UInt64}(),
                              "loop_batch_func", "", false)

    if iteration % 4 == 1 ||
        iteration == num_iterations
        Orion.exec_for_loop(ratings.id,
                            Orion.ForLoopParallelScheme_2d,
                            [W.id], [H.id],
                            Vector{Int32}(),
                            Vector{Int32}(),
                            Vector{Int32}(),
                            Vector{UInt64}(),
                            "eval_batch_func", "", false)
        err = Orion.get_aggregated_value(:err, :+)
        println("err = ", err)
        Orion.reset_accumulator(:err)
        push!(error_vec, err)
    end
    cnt = Orion.get_aggregated_value(:cnt, :+)
    println("cnt = ", cnt)
    Orion.reset_accumulator(:cnt)
end

println(error_vec)

#H.save_as_text_file("/home/ubuntu/model/H")
#W.save_as_text_file("/home/ubuntu/model/W")


Orion.stop()
