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

function gen_space_time_partition_function(func_name::Symbol,
                                           dims::Array{Tuple{Int64}, 1},
                                           coeffs::Array{Tuple{Int64}, 1},
                                           tile_length::Int64,
                                           tile_width::Int64)
    @assert length(dims) == 2
    @assert length(coeffs) == 2
    space_partition_id = Expr(:call, :+, 0)
    for i = 1:length(dims[1])
        push!(space_partition_id.args,
              :(dim_keys[$(dims[1][i])] * $(coeffs[1][i])))
    end
    space_partition_id = Expr(:call, :fld, space_partition_id, tile_length)

    time_partition_id = Expr(:call, :+, 0)
    for i = 1:length(dims[2])
        push!(time_partition_id.args,
              :(dim_keys[$(dims[2][i])] * $(coeffs[2][i])))
    end
    time_partition_id = Expr(:call, :fld, time_partition_id, tile_width)

    add_space_partition_id_stmt = :(results[2 * i - 1] = $(space_partition_id))
    add_time_partition_id_stmt = :(results[2 * i] = $(time_partition_id))

    partition_func = :(
        function $func_name(keys::Array{Int64, 1},
                            dims::Array{Int64, 1},
                            results::Array{Int32, 1})
          rev_dims = reverse(dims)
          println("keys.size() = ", length(keys),
                " typeof(OrionWorker) = ", typeof(OrionWorker))
          i = 1
          for key in keys
            #println(key)
            dim_keys = OrionWorker.from_int64_to_keys(key, rev_dims)
            $add_space_partition_id_stmt
            $add_time_partition_id_stmt
            i += 1
          end
        end)

    return partition_func
end
