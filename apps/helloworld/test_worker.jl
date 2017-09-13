include("/home/ubuntu/orion/src/julia/orion_worker.jl")

function partition(keys::Array{Int64, 1}, dims::Array{Int64, 1})
    results = []
    rev_dims = reverse(dims)
    for key in keys
        dim_keys = OrionWorker.from_int64_to_keys(key, rev_dims)
        push!(results, dim_keys[0])
        push!(results, dim_keys[1])
    end
end

function gen_partition_function(func_name::Symbol,
                                dims::Array{Tuple{Int64}, 1},
                                coeffs::Array{Tuple{Int64}, 1},
                                tile_length::Int64,
                                tile_width::Int64)
    @assert length(dims) == 2
    @assert length(coeffs) == 2
    partition_key_1 = Expr(:call, :+, 0)
    for i = 1:length(dims[1])
        push!(partition_key_1.args,
              :(dim_keys[$(dims[1][i])] * $(coeffs[1][i])))
    end
    partition_key_1 = Expr(:call, :fld, partition_key_1, tile_length)

    partition_key_2 = Expr(:call, :+, 0)
    for i = 1:length(dims[2])
        push!(partition_key_2.args,
              :(dim_keys[$(dims[2][i])] * $(coeffs[2][i])))
    end
    partition_key_2 = Expr(:call, :fld, partition_key_2, tile_width)

    add_pk1_stmt = :(results[2 * i - 1] = $(partition_key_1))
    add_pk2_stmt = :(results[2 * i] = $(partition_key_2))

    partition_func = :(
        function $func_name(keys::Array{Int64, 1},
                            dims::Array{Int64, 1},
                            results::Array{Int64, 1})
            rev_dims = reverse(dims)
            i = 1
            println("keys.size() = ", length(keys))
            for key in keys
              dim_keys = OrionWorker.from_int64_to_keys(key, rev_dims)
              println("key = ", key, "dim_keys = ", dim_keys)
              $add_pk1_stmt
              $add_pk2_stmt
              i += 1
            end
            return results
          end)

    return partition_func
end

partition_func = gen_partition_function(:get_partition,
                                        [(1,), (2,)], [(1,), (1,)],
                                        1,
                                        1)
dump(partition_func.args[2])
eval(partition_func)

keys = [32, 55, 102]
dims = [10, 10]
res = Array{Int64, 1}(length(keys) * 2)
get_partition(keys, dims, res)
println(res)
