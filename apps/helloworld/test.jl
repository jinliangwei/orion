include("/home/ubuntu/orion/src/julia/orion.jl")

macro print_ast(func)
    println(func)
    println(func.head)
    call_head = func.args[1].head
    call_args = func.args[1].args
    println(call_args[2].head)
    esc(func)
end

#@print_ast function parse_line(line::AbstractString)
#    tokens = split(line, ' ')
#    @assert length(tokens) == 3
#    key_tuple = (parse(Int64, String(tokens[1])),
#                 parse(Int64, String(tokens[2])))
#    value = parse(Float64, String(tokens[3]))
#    return (key_tuple, value)
#end

function parse_line(line::AbstractString)
    tokens = split(line, ' ')
    @assert length(tokens) == 3
    key_tuple = (parse(Int64, String(tokens[1])),
                 parse(Int64, String(tokens[2])))
    value = parse(Float64, String(tokens[3]))
    return (key_tuple, value)
end

#Orion.set_lib_path("/home/ubuntu/orion/lib/liborion.so")
# initialize logging of the runtime library
#Orion.glog_init(C_NULL)
#Orion.init(master_ip, master_port, comm_buff_capacity)

#Orion.Ast.parse_map_function(parse_line, (String,))
#Orion.text_file("test", parse_line, (AbstractString,), true)
#Orion.Ast.test_sugar(parse_line, (AbstractString,) )

#Orion.test()

Orion.set_lib_path("/home/ubuntu/orion/lib/liborion.so")
#Orion.helloworld()
#Orion.glog_init()

const data_path = "file:///home/ubuntu/data/ml-1m/ratings.csv"
const K = 100
const num_iterations = 10
const step_size = 0.0001
ratings = Orion.text_file(data_path, parse_line)

ratings.dims = [100, 100]
x_dim, y_dim = size(ratings)
println((x_dim, y_dim))

W = Orion.rand(x_dim, K)
H = Orion.rand(y_dim, K)

println(isa(ratings, Orion.DistArray))

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
    #println("iteration = ", i, " error = ", error)
    @printf "iteration = %d, error = %f\n" i sqrt((error / length(ratings)))
end
