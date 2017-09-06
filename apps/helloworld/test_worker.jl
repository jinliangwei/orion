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
                                coeffs::Array{Tuple{Int64}, 1})
    @assert length(dims) == 2
    @assert length(coeffs) == 2
    partition_key_1 = Expr(:call, :+, 0)
    for i = 1:length(dims[1])
        push!(partition_key_1.args,
              :(dim_keys[$(dims[1][i])] * $(coeffs[1][i])))
    end

    partition_key_2 = Expr(:call, :+, 0)
    for i = 1:length(dims[2])
        push!(partition_key_2.args,
              :(dim_keys[$(dims[2][i])] * $(coeffs[2][i])))
    end

    push_pk1_stmt = :(push!(results, $(partition_key_1)))
    push_pk2_stmt = :(push!(results, $(partition_key_2)))

    partition_func = :(
        function $func_name(keys::Array{Int64, 1}, dims::Array{Int64, 1})
           results = []
            rev_dims = reverse(dims)
            for key in keys
              dim_keys = OrionWorker.from_int64_to_keys(key, rev_dims)
              $push_pk1_stmt
              $push_pk2_stmt
            end
            return results
          end)

    return partition_func
end

partition_func = gen_partition_function(:get_partition,
                                        [(1,), (2,)], [(1,), (1,)])
dump(partition_func.args[2])
eval(partition_func)

keys = [32, 55, 102]
dims = [10, 10]
res = get_partition(keys, dims)
println(res)
