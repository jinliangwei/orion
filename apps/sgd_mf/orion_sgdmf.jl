include("/users/jinlianw/orion.git/src/julia/orion.jl")

println("application started")

# set path to the C++ runtime library
Orion.set_lib_path("/users/jinlianw/orion.git/lib/liborion_driver.so")
# test library path
Orion.helloworld()

const master_ip = "10.117.1.1"
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
const data_path = "file:///users/jinlianw/ratings.csv"
#const data_path = "file:///proj/BigLearning/jinlianw/data/netflix.csv"
#const data_path = "file:///proj/BigLearning/jinlianw/data/ml-20m/ratings_p.csv"
#const data_path = "file:///proj/BigLearning/jinlianw/data/ml-latest/ratings_p.csv"
const K = 10
const num_iterations = 4
const step_size = Float32(0.01)

Orion.@accumulator err = Float32(0.0)
Orion.@accumulator line_cnt = 0

Orion.@share function parse_line(line::AbstractString)
    global line_cnt
    line_cnt += 1
    tokens = split(line, ',')
    @assert length(tokens) == 3
    key_tuple = (parse(Int64, String(tokens[2])) + 1,
                 parse(Int64, String(tokens[1])) + 1)
    value = parse(Float32, String(tokens[3]))
    return (key_tuple, value)
end

Orion.@share function parse_line_ml(line::AbstractString)
    global line_cnt
    line_cnt += 1
    tokens = split(line, ',')
    key_tuple = (parse(Int64, String(tokens[2])),
                 parse(Int64, String(tokens[1])))
    value = parse(Float32, String(tokens[3]))
    return (key_tuple, value)
end

Orion.@share function map_init_param(value::Float32)::Float32
    return value / 10
end

Orion.@dist_array ratings = Orion.text_file(data_path, parse_line)
Orion.materialize(ratings)
dim_x, dim_y = size(ratings)

println((dim_x, dim_y))
line_cnt = Orion.get_aggregated_value(:line_cnt, :+)
println("line_cnt = ", line_cnt)

Orion.@dist_array W = Orion.randn(K, dim_x)
Orion.@dist_array W = Orion.map(W, map_init_param, map_values = true)
Orion.materialize(W)

Orion.@dist_array H = Orion.randn(K, dim_y)
Orion.@dist_array H = Orion.map(H, map_init_param, map_values = true)
Orion.materialize(H)

#Orion.dist_array_set_num_partitions_per_dim(ratings, num_executors * 4)

error_vec = Vector{Float32}()
time_vec = Vector{Float64}()
start_time = now()
last_time = start_time

W_grad = zeros(Float32, K)
H_grad = zeros(Float32, K)

@time for iteration = 1:num_iterations
    @time Orion.@parallel_for histogram_partitioned for (rating_key, rv) in ratings
        x_idx = rating_key[1]
        y_idx = rating_key[2]

        W_row = @view W[:, x_idx]
        H_row = @view H[:, y_idx]
        pred = dot(W_row, H_row)
        diff = rv - pred
        #W_grad .= -2 * diff .* H_row
        #H_grad .= -2 * diff .* W_row
        #W[:, x_idx] .= W_row .- step_size .* W_grad
        #H[:, y_idx] .= H_row .- step_size .* H_grad
        @. W_grad = -2 * diff * H_row
        @. H_grad = -2 * diff * W_row
        @. W[:, x_idx] = W_row - step_size * W_grad
        @. H[:, y_idx] = H_row - step_size * H_grad
    end
    if iteration % 1 == 0 ||
        iteration == num_iterations
        println("evaluate model")
        @time Orion.@parallel_for for (rating_key, rv) in ratings
            x_idx = rating_key[1]
            y_idx = rating_key[2]

            W_row = @view W[:, x_idx]
            H_row = @view H[:, y_idx]
            pred = dot(W_row, H_row)
            err += abs2(rv - pred)
        end
        err = Orion.get_aggregated_value(:err, :+)
        curr_time = now()
        elapsed = Int(Dates.value(curr_time - start_time)) / 1000
        diff_time = Int(Dates.value(curr_time - last_time)) / 1000
        last_time = curr_time
        println("iteration = ", iteration, " elapsed = ", elapsed, " iter_time = ", diff_time,
                " err = ", err)
        Orion.reset_accumulator(:err)
        push!(error_vec, err)
        push!(time_vec, elapsed)
    end
end
println(error_vec)
println(time_vec)
loss_fobj = open("results.order/" * split(PROGRAM_FILE, "/")[end] * "-" *
                 split(data_path, "/")[end] * "-" * string(num_executors) * "-" *
                 string(K) * "-" * string(num_iterations) * "-" * string(step_size) * "-" * string(now()) * ".loss", "w")
for idx in eachindex(time_vec)
    write(loss_fobj, string(idx) * "\t" * string(time_vec[idx]) * "\t" * string(error_vec[idx]) * "\n")
end
Orion.stop()
close(loss_fobj)
exit()
