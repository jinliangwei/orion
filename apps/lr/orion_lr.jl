include("/users/jinlianw/orion.git/src/julia/orion.jl")

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

const data_path = "file:///users/jinlianw/ratings.csv"
const num_iterations = 10
const step_size = 0.001
Orion.@share const num_features = 123

Orion.@share function parse_line(index::Int64, line::AbstractString)
    feature_vec = Vector{Tuple{Int64, Float32}}()
    tokens = split(line, ' ')
    label = parse(Int64, tokens[1])
    push!(feature_vec, (index * num_features, label))
    for token in tokens[2:end]
        feature = split(token, ":")
        feature_id = parse(Int64, feature[1]) + 1
        feature_val = parse(Float32, feature[2])
        push!(feature_vec, (feature_id, feature_val))
    end
    return feature_vec
end

Orion.@dist_array samples_mat = Orion.text_file_with_line_number(data_path, parse_line)
Orion.materialize(samples_mat)

weights = Orion.rand(num_features)
Orion.materialize(weights)

Orion.@share function sigmoid(z)
    return 1.0 ./ (1.0 .+ exp(-z))
end

Orion.set_iterate_dims(samples_mat, samples_mat.dims[2:end])

Orion.@dist_array_buffer weights_buf = Orion.dist_array_buffer(weights.dims, 0)
Orion.materialize(weights_buf)

Orion.@share function apply_buffered_update(weight, update)
    return weight + update
end

Orion.set_write_buffer(weights, apply_buffered_update, weights_buf)

for i = 1:num_iterations
    Orion.@parallel_for for sample in samples_mat
        sum = 0.0
        sample_keys = sample[1]
        sample_values = sample[2]
        label = sample_values[1]

        for key_index in eachindex(sample_keys[2:end])
            fid = sample_keys[key_index][1]
            fval = sample_value[key_index]
            sum += weights[fid] * fval
        end
        error = label - sigmoid(sum)
        for key_index in eachindex(sample_keys[2:end])
            fid = sample_keys[key_index][1]
            fval = sample_value[key_index]
            weights_buf[fid] -= step_size * fval * error
        end
    end
end
