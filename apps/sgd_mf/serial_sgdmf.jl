#const data_path = "/users/jinlianw/ratings.csv"
#const data_path = "/home/ubuntu/data/ml-1m/ratings.csv"
const data_path = "/proj/BigLearning/jinlianw/data/netflix.csv"
const K = 100
const num_iterations = 1
const step_size = 0.001

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
    ratings = Array{Tuple{Integer, Integer, Real}}(0)
    open(path, "r") do dataf
        for line::String in eachline(dataf)
            token_tuple = parse_line(line)
            push!(ratings, token_tuple)
        end
    end
    return ratings
end

function get_dimension(ratings::Array{Tuple{Integer, Integer, Real}})
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

dim_x, dim_y = get_dimension(ratings)
println((dim_x, dim_y))
W = rand(K, dim_x)
H = rand(K, dim_y)

for i = 1:num_iterations
@time for rating in ratings
	x_idx = rating[1] + 1
	y_idx = rating[2] + 1
	rv = rating[3]

        W_row = W[:, x_idx]
	H_row = H[:, y_idx]
	pred = dot(W_row, H_row)
	diff = rv - pred
	W_grad = -2 * diff .* H_row
	H_grad = -2 * diff .* W_row
	W[:, x_idx] = W_row - step_size .* W_grad
	H[:, y_idx] = H_row - step_size .*H_grad
    end
end
