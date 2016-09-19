module Orion

export load_data

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
    return ratings
end

end
