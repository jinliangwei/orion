include("/home/ubuntu/orion/src/julia/orion.jl")

const data_path = "/home/ubuntu/data/libsvm/a1a"
const num_iterations = 10
const step_size = 0.001
Orion.@share const num_features = 123

# set path to the C++ runtime library
Orion.set_lib_path("/home/ubuntu/orion/lib/liborion_driver.so")
Orion.load_constants()
# test library path
Orion.helloworld()

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

data_str = Orion.text_file(data_path)
indexed_data_str = Orion.zip_with_index(data_str)
features_mat = Orion.map(indexed_data_str, parse_line)
Orion.materialize(features_mat)

weights = Orion.rand(num_features)
Orion.materialize(weights)

Orion.@share function sigmoid(z)
    return 1.0 ./ (1.0 .+ exp(-z))
end

Orion.set_access_granularity(features, (1, :))

weights_buf = Orion.zeros(num_features)
Orion.materialize(weights_buf)

for i = 1:num_iterations
    Orion.@parallel_for for feature in features
        sum = 0.0
        label = feature[1][2]
        for feature_pair in feature[2:end]
            fid = feature_pair[1]
            fval = feature_pair[2]
            sum += weights[fid] * fval
        end
        error = label - sigmoid(sum)
        for feature_pair in feature[2:end]
            fid = feature_pair[1]
            fval = feature_pair[2]
            weights_buf[fid] -= step_size * fval * error
        end
    end
    Orion.add_to(weights, weights_buf)
    Orion.zeros(weights_buf)
end
