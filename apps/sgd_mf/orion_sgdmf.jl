include("/home/ubuntu/orion/src/julia/orion.jl")

# set path to the C++ runtime library
Orion.set_lib_path("/home/ubuntu/orion/lib/liborion.so")
# test library path
Orion.helloworld()

const master_ip = "127.0.0.1"
const master_port = 10000
const comm_buff_capacity = 1024

# initialize logging of the runtime library
Orion.glog_init()
Orion.init(master_ip, master_port, comm_buff_capacity)

const data_path = "file:///home/ubuntu/data/ml-1m/ratings.csv"
const K = 100
const num_iterations = 1
const step_size = 0.001

Orion.@share function parse_line(line::AbstractString)
    tokens = split(line, ',')
    @assert length(tokens) == 3
    key_tuple = (parse(Int64, String(tokens[1])),
                 parse(Int64, String(tokens[2])))
    value = parse(Float64, String(tokens[3]))
    return (key_tuple, value)
end


ratings = Orion.text_file(data_path, parse_line)
Orion.materialize(ratings)
dim_x, dim_y = size(ratings)

println((dim_x, dim_y))

W = Orion.rand(dim_x, K)
H = Orion.rand(dim_y, K)
Orion.materialize(W)
Orion.materialize(H)

Orion.@transform for i = 1:num_iterations
    Orion.@accumulator error = 0.0
    Orion.@parallel_for for rating in ratings
	x_idx = rating[1] + 1
	y_idx = rating[2] + 1
	rv = rating[3]

        W_row = W[x_idx, :]
	H_row = H[y_idx, :]
	pred = dot(W_row, H_row)
	diff = rv - pred
	W_grad = -2 * diff .* H_row
	H_grad = -2 * diff .* W_row
	W[x_idx, :] = W_row - step_size .* W_grad
	H[y_idx, :] = H_row - step_size .*H_grad
        error += (pred - rv) ^ 2
    end
    println("iteration = ", i, " error = ", error)
end

Orion.stop()
