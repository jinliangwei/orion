include("/users/jinlianw/orion.git/src/julia/orion.jl")

# set path to the C++ runtime library
Orion.set_lib_path("/users/jinlianw/orion.git/lib/liborion_driver.so")
# test library path
Orion.helloworld()

const master_ip = "127.0.0.1"
const master_port = 10000
const comm_buff_capacity = 1024
const num_executors = 16
const num_servers = 1

# initialize logging of the runtime library
Orion.glog_init()
Orion.init(master_ip, master_port, comm_buff_capacity, num_executors,
           num_servers)

const data_path = "file:///proj/BigLearning/jinlianw/data/kdda"
#const data_path = "file:///proj/BigLearning/jinlianw/data/a1a"
const num_iterations = 128
const step_size = Float32(0.00001)
const num_features = 20216830
#const num_features = 123

Orion.@accumulator err = Float32(0)
Orion.@accumulator loss = Float32(0)
Orion.@accumulator line_cnt = 0

Orion.@share function parse_line(index::Int64, line::AbstractString)::Tuple{Tuple{Int64},
                                                                            Tuple{Int64, Vector{Tuple{Int64, Float32}}
                                                                                  }
                                                                            }
    global line_cnt += 1
    tokens = split(strip(line), ' ')
    label = parse(Int64, tokens[1])
    if label == -1
        label = 0
    end
    i = 1
    feature_vec = Vector{Tuple{Int64, Float32}}(length(tokens) - 1)
    for token in tokens[2:end]
        feature = split(token, ":")
        feature_id = parse(Int64, feature[1])
        @assert feature_id >= 1
        feature_val = parse(Float32, feature[2])
        feature_vec[i] = (feature_id, feature_val)
        i += 1
    end
    return ((index,), (label, feature_vec))
end

Orion.@dist_array samples_mat = Orion.text_file(data_path,
                                                parse_line,
                                                is_dense = true,
                                                with_line_number = true,
                                                new_keys = true,
                                                num_dims = 1)
Orion.materialize(samples_mat)

line_cnt = Orion.get_aggregated_value(:line_cnt, :+)
println("line_cnt = ", line_cnt)

Orion.@dist_array weights = Orion.rand(num_features)
Orion.materialize(weights)

Orion.@share function sigmoid(z)
    return Float32(1.0) ./ (Float32(1.0) .+ exp(-z))
end

Orion.@share function safe_log(x)
    if abs(x) < Float32(1e-15)
        x = Float32(1e-15)
    end
    return log(x)
end

Orion.@dist_array weights_buf = Orion.create_sparse_dist_array_buffer((weights.dims...), Float32(0.0))
Orion.materialize(weights_buf)

Orion.@share function apply_buffered_update(key, weight, update)
    return weight + update
end

Orion.set_write_buffer(weights_buf, weights, apply_buffered_update)
#Orion.dist_array_set_num_partitions_per_dim(samples_mat, 128)

error_vec = Vector{Float32}()
loss_vec = Vector{Float32}()

for iteration = 1:num_iterations
    Orion.@parallel_for for sample in samples_mat
        sum = 0.0
        label = sample[2][1]
        features = sample[2][2]
        for feature in features
            fid = feature[1]
            fval = feature[2]
            sum += weights[fid] * fval
        end
        diff = sigmoid(sum) - label
        for feature in features
            fid = feature[1]
            fval = feature[2]
             weights_buf[fid] -= step_size * fval * diff
        end
    end
    if iteration % 1 == 0 ||
        iteration == num_iterations
        Orion.@parallel_for for sample in samples_mat
            sum = 0.0
            label = sample[2][1]
            features = sample[2][2]
            for feature in features
                fid = feature[1]
                fval = feature[2]
                sum += weights[fid] * fval
            end

            if label == 1
                loss += -safe_log(sigmoid(sum))
            else
                loss += -safe_log(1 - sigmoid(sum))
            end
            diff = sigmoid(sum) - label
            err += abs2(diff)
        end
        err = Orion.get_aggregated_value(:err, :+)
        loss = Orion.get_aggregated_value(:loss, :+)
        println("iteration = ", iteration, " err = ", err, " loss = ", loss)
        push!(error_vec, err)
        push!(loss_vec, loss)
        Orion.reset_accumulator(:err)
        Orion.reset_accumulator(:loss)
    end
end

println(error_vec)
println(loss_vec)
Orion.stop()
exit()
