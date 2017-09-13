include("/home/ubuntu/orion/src/julia/orion.jl")

const K = 100
const num_iterations = 1
const step_size = 0.001
const factor = 1

ratings = Orion.DistArray{Float32}()
W = Orion.DistArray{Float32}()
H = Orion.DistArray{Float32}()

Orion.@objective function compute_loss(ratings, W, H)
    Orion.@accumulator error = 0.0
    for rating in ratings
        x_idx = rating[1] + 1
	y_idx = rating[2] + 1
	rv = rating[3]

        W_row = W[x_idx, :]
	H_row = H[y_idx, :]
	pred = dot(W_row, H_row)
        error += (rv - pred) ^ 2
    end
    error /= length(ratings)
    return error
end

Orion.@evaluate compute_loss ratings W H 4

Orion.@transform for i = 1:num_iterations
    Orion.@parallel_for for rating in ratings
	x_idx = rating[1] + 1
	y_idx = rating[2] + 1
	rv = rating[3]

        W_row = W[x_idx, :]
	H_row = H[y_idx, :]
	pred = dot(W_row, H_row) * factor
	diff = rv - pred
	W_grad = -2 * diff .* H_row
	H_grad = -2 * diff .* W_row
	W[x_idx, :] = W_row - step_size .* W_grad
	H[y_idx, :] = H_row - step_size .*H_grad
    end
    step_size *= 0.95
end
