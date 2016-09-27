include("/home/jinliang/orion.git/src/utils.jl")

using Orion

const data_path = "/home/jinliang/data/ml-1m/ratings_shuffled.csv"
const K = 100
const num_iterations = 1
const step_size = 0.001

println("serial sgd mf starts here!")
ratings, max_x, max_y = Orion.load_data(data_path)

W = rand(max_x + 1, K)
H = rand(max_y + 1, K)
@printf "max_x = %d, max_y = %d \n" max_x max_y

for i = 1:num_iterations
    error = 0.0
    @time for rating in ratings
        x_idx = rating[1] + 1
        y_idx = rating[2] + 1
        rv = rating[3]
        #@printf "here, %d, %d, %f\n" x_idx y_idx rv

        W_row = W[x_idx, :]
        H_row = H[y_idx, :]
        pred = dot(vec(W_row), vec(H_row))
        diff = rv - pred
        W_grad = -2 * diff .* H_row
        H_grad = -2 * diff .* W_row
        W[x_idx, :] = W_row - step_size .* W_grad
        H[y_idx, :] = H_row - step_size .*H_grad
    end
    @time for rating in ratings
        x_idx = rating[1] + 1
        y_idx = rating[2] + 1
        rv = rating[3]
        #@printf "here, %d, %d, %f\n" x_idx y_idx rv

        W_row = W[x_idx, :]
        H_row = H[y_idx, :]
        pred = dot(vec(W_row), vec(H_row))
        error += (pred - rv) ^ 2
    end
    @printf "iteration = %d, error = %f\n" i sqrt((error / length(ratings)))
end

exit()

# below is just random testing code
#addprocs(1)
#println("added 1 processes")

#expr = :(Orion.a + 3)

#@everywhere include("/home/jinliang/orion.git/src/utils.jl")
#@everywhere using Orion

#rr = remotecall(2, Orion.remote_eval, expr)
#ret = fetch(rr)
#println(ret)
