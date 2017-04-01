import Orion

const data_path = "/home/ubuntu/data/ml-1m/ratings_shuffled.csv"
const K = 100
const num_iterations = 1
const step_size = 0.001

function parse_line(line:AbstractString)
    tokens = split(line, ',')
    @assert length(tokens) == 3
    token_tuple = ((parse(Int64, AbstractString(tokens[1])),
                   parse(Int64, AbstractString(tokens[2]))),
                   parse(Float64, AbstractString(tokens[3])))
    return token_tuple
end

Orion.init()

ratings = Orion.loadTextFile(data_path).mapWithKey(parse_line)
max_x, max_y = ratings.get_dimensions()

W = Orion.rand(max_x + 1, K)
H = Orion.rand(max_y + 1, K)

@orion iterative
for i = 1:num_iterations
    error = 0.0
    for rating in ratings
	x_idx = rating[1] + 1
	y_idx = rating[2] + 1
	rv = rating[3]

        W_row = W[x_idx, :]
	H_row = H[y_idx, :]
	pred = dot(vec(W_row), vec(H_row))
	diff = rv - pred
	W_grad = -2 * diff .* H_row
	H_grad = -2 * diff .* W_row
	W[x_idx, :] = W_row - step_size .* W_grad
	H[y_idx, :] = H_row - step_size .*H_grad
    end
    for rating in ratings
	x_idx = rating[1] + 1
	y_idx = rating[2] + 1
	rv = rating[3]

        W_row = W[x_idx, :]
	H_row = H[y_idx, :]
	pred = dot(vec(W_row), vec(H_row))
	error += (pred - rv) ^ 2
    end
    @printf "iteration = %d, error = %f\n" i sqrt((error / length(ratings)))
end
