const data_path = "/proj/BigLearning/jinlianw/data/a1a"
const num_iterations = 16
step_size = 0.0001
const num_features = 123
const step_size_decay = 0.9

err = 0
loss = 0
line_cnt = 0

function parse_line(index::Int64, line::AbstractString)::Tuple{Tuple{Int64},
                                                               Tuple{Int64, Vector{Tuple{Int64, Float32}}
                                                                     }
                                                               }
    global line_cnt += 1
    feature_vec = Vector{Tuple{Int64, Float32}}()
    tokens = split(strip(line), ' ')
    label = parse(Int64, tokens[1])
    for token in tokens[2:end]
        feature = split(token, ":")
        feature_id = parse(Int64, feature[1])
        @assert feature_id >= 0
        feature_val = parse(Float32, feature[2])
        push!(feature_vec, (feature_id, feature_val))
    end
    return ((index,), (label, feature_vec))
end

function load_data(path::AbstractString)
    num_lines::Int64 = 0
    samples = Vector{Tuple{Tuple{Int64}, Tuple{Int64, Vector{Tuple{Int64, Float32}}}}}()
    open(path, "r") do dataf
        for line::String in eachline(dataf)
            token_tuple = parse_line(num_lines, line)
            push!(samples, token_tuple)
        end
    end
    return samples
end

samples = load_data(data_path)

weights = rand(num_features)

function sigmoid(z)
    return 1.0 / (1.0 + exp(-z))
end

function safe_log(x)
    if abs(x) < 1e-15
        x = 1e-15
    end
    return log(x)
end

error_vec = Vector{Float64}()
loss_vec = Vector{Float64}()
for iteration = 1:num_iterations
    for sample in samples
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
            weights[fid] -= step_size * fval * diff
        end
    end
    step_size *= step_size_decay
    println(weights)
    if iteration % 1 == 0 ||
        iteration == num_iterations
        for sample in samples
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
            err += diff ^ 2
        end
        println("iteration = ", iteration, " err = ", err, " loss = ", loss)
        push!(error_vec, err)
        push!(loss_vec, loss)
        err = 0
        loss = 0
    end
end

println(error_vec, err)
println(loss_vec, loss)
