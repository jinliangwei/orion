module OrionWorker

function set_lib_path(path::AbstractString)
    global const lib_path = path
end

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
end

function from_int64_to_keys(key::Int64, rev_dims::Vector{Int64})
    dim_keys = []
    for dim in rev_dims
        key_this_dim = key % dim
        key = fld(key, dim)
        push!(dim_keys, key_this_dim)
    end
    return reverse(dim_keys)
end

function from_keys_to_int64(key::Tuple, dims::Vector{Int64})
    key_int = 0
    dim_size = 1
    for i = 0:(length(dims)-1)
        key_int += key[end - i] * dim_size
        dim_size *= dims[end- i]
    end
    return key_int
end

end
