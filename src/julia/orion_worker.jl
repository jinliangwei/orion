module OrionWorker

function set_lib_path(path::AbstractString)
    global const lib_path = path
end

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
end

function from_int64_to_keys(key::Int64, dims::Array{Int64, 1})
    accum_dims = 1
    dim_keys = []
    for dim in dims
        accum_dims *= dim
        key_this_dim = key % accum_dims
        key = fld(key, accum_dims)
        push!(dim_keys, key_this_dim)
    end
    return reverse(dim_keys)
end

end
