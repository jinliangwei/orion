const data_path = "/users/jinlianw/ratings.csv"
const K = 100
const num_iterations = 2
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
            err = 0
            for rating in ratings
                x_idx = rating[1] + 1
                y_idx = rating[2] + 1
                rv = rating[3]

                W_row = @view W[:, x_idx]
                H_row = @view H[:, y_idx]
                pred = dot(W_row, H_row)
                err += (rv - pred) ^ 2
            end
            println("iteration = ", iteration,
                    " err = ", err)
        end
    end
end

train(ratings, step_size, num_iterations)
