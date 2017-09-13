module OrionWorker

function set_lib_path(path::AbstractString)
    global const lib_path = path
end

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
end

function from_int64_to_keys(key::Int64, rev_dims::Array{Int64, 1})
    dim_keys = []
    for dim in rev_dims
        key_this_dim = key % dim
        key = fld(key, dim)
        push!(dim_keys, key_this_dim)
    end
    return reverse(dim_keys)
end

end
