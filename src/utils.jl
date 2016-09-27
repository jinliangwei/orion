module Orion

export load_data, remote_eval
a = 10
function load_data(path::AbstractString)
    num_lines::Int64 = 0
    ratings = Array{Tuple{Integer, Integer, Real}}(1)
    open(path, "r") do dataf
        for line::AbstractString in eachline(dataf)
            tokens = split(line, ',')
            @assert length(tokens) == 3
            token_tuple = (parse(Int64, AbstractString(tokens[1])),
                           parse(Int64, AbstractString(tokens[2])),
                           parse(Float64, AbstractString(tokens[3])))
            num_lines += 1
            push!(ratings, token_tuple)
        end
    end
    @printf "processed totally %d lines!\n" num_lines
    ret_ratings = Array{Tuple{Integer, Integer, Real}}(num_lines)
    max_x = 0
    max_y = 0
    @time for idx in eachindex(ratings)
        if idx == 1
            continue
        end
        ret_ratings[idx - 1] = ratings[idx]
        if ratings[idx][1] > max_x
            max_x = ratings[idx][1]
        end
        if ratings[idx][2] > max_y
            max_y = ratings[idx][2]
        end
    end
    return ret_ratings, max_x, max_y
end

function remote_eval(expr::Expr)
    arr_ptr = ccall((:allocate_array, "/home/jinliang/orion.git/src/liborion.so"), Ptr{Int32}, ())
    arr = pointer_to_array(arr_ptr, 1, false)
    a = arr[1]
    ret = eval(expr)
    ccall((:free, "/home/jinliang/orion.git/src/liborion.so"), Void, (Ptr{Int32},), arr_ptr)
    return ret
end

end
