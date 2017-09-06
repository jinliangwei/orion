symbol_counter = 0
function gen_unique_symbol()::Symbol
    global symbol_counter += 1
    return Symbol("osym_", string(symbol_counter))
end

function gen_stmt_broadcast_var(var_set::Set{Symbol})::Array{Expr}
    var_set_sym = gen_unique_symbol()
    expr_array = Array{Expr, 1}()
    create_set_expr = :($var_set_sym = Set{Symbol}())
    push!(expr_array, create_set_expr)
    for var in var_set
        add_var_to_set_expr = Expr(:call,
                                   :(push!),
                                   var_set_sym,
                                   QuoteNode(var)
                                   )
        push!(expr_array, add_var_to_set_expr)
    end
    call_broadcast_func_expr = :(broadcast($var_set_sym))
    push!(expr_array, call_broadcast_func_expr)
    dump(expr_array)
    return expr_array
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
