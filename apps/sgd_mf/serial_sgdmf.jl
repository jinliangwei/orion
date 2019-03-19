const data_path = "/proj/BigLearning/jinlianw/data/netflix.csv"
#const data_path = "/users/jinlianw/ratings.csv"
#const data_path = "/proj/BigLearning/jinlianw/data/ml-20m/ratings_p.csv"
const K = 1000
const num_iterations = 100
const step_size = 0.01

function parse_line(line::AbstractString)
    tokens = split(line, ',')
    @assert length(tokens) == 3
    token_tuple = (parse(Int64, String(tokens[1])),
                   parse(Int64, String(tokens[2])),
                   parse(Float64, String(tokens[3])))
    return token_tuple
end

function load_data(path::AbstractString)
    num_lines::Int64 = 0
    ratings = Array{Tuple{Int, Int, Float64}}(0)
    open(path, "r") do dataf
        for line::String in eachline(dataf)
            token_tuple = parse_line(line)
            push!(ratings, token_tuple)
        end
    end
    return ratings
end

function get_dimension(ratings::Array{Tuple{Int, Int, Float64}})
    max_x = 0
    max_y = 0
    for idx in eachindex(ratings)
        if ratings[idx][1] > max_x
            max_x = ratings[idx][1]
        end
        if ratings[idx][2] > max_y
            max_y = ratings[idx][2]
        end
    end
    return max_x + 1, max_y + 1
end

println("serial sgd mf starts here!")
ratings = load_data(data_path)
println("load data done!")

function train(ratings, step_size, num_iterations)
    dim_x, dim_y = get_dimension(ratings)
    println((dim_x, dim_y))
    W = randn(K, dim_x) ./ 10
    H = randn(K, dim_y) ./ 10
    W_grad = zeros(K)
    H_grad = zeros(K)

    error_vec = Vector{Float32}()
    time_vec = Vector{Float64}()
    start_time = now()
    last_time = start_time

    for iteration = 1:num_iterations
        @time for rating in ratings
            x_idx = rating[1] + 1
            y_idx = rating[2] + 1
            rv = rating[3]

            W_row = @view W[:, x_idx]
            H_row = @view H[:, y_idx]
            pred = dot(W_row, H_row)
            diff = rv - pred
            @. W_grad = -2 * diff * H_row
            @. H_grad = -2 * diff * W_row
            @. W[:, x_idx] = W_row - step_size * W_grad
            @. H[:, y_idx] = H_row - step_size * H_grad
        end
        if iteration % 1 == 0 ||
            iteration == num_iterations
            println("evaluate model")
            err = 0.0
            @time for rating in ratings
                x_idx = rating[1] + 1
                y_idx = rating[2] + 1
                rv = rating[3]

                W_row = @view W[:, x_idx]
                H_row = @view H[:, y_idx]
                pred = dot(W_row, H_row)
                err += abs2(rv - pred)
            end
            println("iteration = ", iteration, " err = ", err)
            curr_time = now()
            elapsed = Int(Dates.value(curr_time - start_time)) / 1000
            diff_time = Int(Dates.value(curr_time - last_time)) / 1000
            last_time = curr_time
            push!(error_vec, err)
            push!(time_vec, elapsed)
        end
    end

    loss_fobj = open("results/" * split(PROGRAM_FILE, "/")[end] * "-" *
                 split(data_path, "/")[end] * "-" * "serial" * "-" *
                 string(K) * "-" * string(num_iterations) * "-" * string(step_size) * "-" * string(now()) * ".loss", "w")
    for idx in eachindex(time_vec)
        write(loss_fobj, string(idx) * "\t" * string(time_vec[idx]) * "\t" * string(error_vec[idx]) * "\n")
    end
    close(loss_fobj)
end

train(ratings, step_size, num_iterations)
