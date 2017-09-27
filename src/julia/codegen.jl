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

function gen_2d_partition_function(func_name::Symbol,
                                   time_partition_dim::Int64,
                                   space_partition_dim::Int64,
                                   time_dim_tile_size::Int64,
                                   space_dim_tile_size::Int64)

    space_partition_id = :(dim_keys[$space_partition_dim] % space_dim_tile_size)
    time_partition_id = :(dim_keys[$time_partition_dim] % time_dim_tile_size)

    add_space_partition_id_stmt = :(results[2 * i - 1] = $(space_partition_id))
    add_time_partition_id_stmt = :(results[2 * i] = $(time_partition_id))

    partition_func = :(
        function $func_name(keys::Vector{Int64},
                            dims::Vector{Int64},
                            results::Vector{Int32})
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

function gen_utransform_2d_partition_function(func_name::Symbol,
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
        function $func_name(keys::Vector{Int64},
                            dims::Vector{Int64},
                            results::Vector{Int32})
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

#function gen_map_no_flatten_function(func_name::Symbol,
#                                     map_func_names::Vector{String},
#                                     map_func_modules::Vector{Module},
#                                     map_types::Vector{DistArrayMapType})
#end


function gen_map_function(func_name::Symbol,
                          map_func_names::Vector{String},
                          map_func_modules::Vector{Module},
                          map_types::Vector{DistArrayMapType},
                          map_flattens::Vector{Bool})

    loop_stmt = :(for i = 1:length(keys)
                  key = keys[i]
                  value = values[i]
                  dim_keys = OrionWorker.from_int64_to_keys(key, rev_dims)
                  dim_keys = tuple(dim_keys...)
                  end)
    stmt_array_to_append = loop_stmt.args[2].args
    key_sym_counter = 1
    value_sym_counter = 1
    kv_array_counter = 1
    const key_var_prefix = "key"
    const value_var_prefix = "value"
    const kv_array_prefix = "kv_array"
    current_key_var = "dim_keys"
    current_value_var = "value"

    for i = 1:length(map_func_names)
        mapper_name = map_func_names[i]
        mapper_module = map_func_modules[i]
        map_type = map_types[i]
        flatten_result = map_flattens[i]

        mapper_name_sym = Symbol(mapper_name)
        mapper_module_sym = Symbol(string(mapper_module))
        current_key_sym = Symbol(current_key_var)
        current_value_sym = Symbol(current_value_var)
        output_key_sym = Symbol(key_var_prefix * "_" * string(key_sym_counter))
        output_value_sym = Symbol(value_var_prefix * "_" * string(value_sym_counter))
        output_kv_array_sym = Symbol(kv_array_prefix * "_" * string(kv_array_counter))

        new_key = false
        if map_type == DistArrayMapType_map
            if flatten_result
                mapper_call_expr = :(
                    $output_kv_array_sym = $mapper_module_sym.$mapper_name_sym(
                        $current_key_sym, $current_value_sym)
                )
            else
                mapper_call_expr = quote
                    $output_key_sym, $output_value_sym = $mapper_module_sym.$mapper_name_sym(
                        $current_key_sym, $current_value_sym)
                end
            end
            new_key = true
            key_sym_counter += 1
        elseif map_type == DistArrayMapType_map_fixed_keys
            mapper_call_expr = :(
            $output_value_sym = $mapper_module_sym.$mapper_name_sym(
                $current_key_sym, $current_value_sym)
            )
        elseif map_type == DistArrayMapType_map_values
            mapper_call_expr = :(
            $output_value_sym = $mapper_module_sym.$mapper_name_sym(
                $current_value_sym)
            )
        elseif map_type == DistArrayMapType_map_values_new_keys
            if flatten_result
                mapper_call_expr = :(
                    $output_kv_array_sym = $mapper_module_sym.$mapper_name_sym(
                        $current_key_sym, $current_value_sym)
                )
            else
                mapper_call_expr = quote
                    $output_key_sym, $output_value_sym = $mapper_module_sym.$mapper_name_sym(
                        $current_value_sym)
                end
            end
            new_key = true
            key_sym_counter += 1
        else
            @assert false
        end
        push!(stmt_array_to_append, mapper_call_expr)

        if flatten_result
            current_key_var = key_var_prefix * "_" * string(key_sym_counter)
            current_value_var = value_var_prefix * "_" * string(value_sym_counter)
            @assert new_key
            current_key_sym = Symbol(current_key_var)
            current_value_sym = Symbol(current_value_var)
            new_loop_stmt = :(
                for ($current_key_sym, $current_value_sym) in $output_kv_array_sym
                end
            )
            push!(stmt_array_to_append, new_loop_stmt)
            stmt_array_to_append = new_loop_stmt.args[2].args
         else
            if new_key
                current_key_var = string(output_key_sym)
            end
            current_value_var = string(output_value_sym)
        end
    end
    current_key_sym = Symbol(current_key_var)
    current_value_sym = Symbol(current_value_var)

    append_key_value_stmts = quote
        key_int64 = OrionWorker.from_keys_to_int64($current_key_sym, dims)
        push!(output_keys, key_int64)
        push!(output_values, $current_value_sym)
    end
    push!(stmt_array_to_append, append_key_value_stmts)

    map_func = :(
        function $func_name(
            dims::Vector{Int64},
            keys::Vector{Int64},
            values::Vector,
            output_value_type::DataType)::Tuple{Vector{Int64}, Vector{output_value_type}}
        rev_dims = reverse(dims)
        output_keys = Vector{Int64}()
        output_values = Vector{output_value_type}()
        $loop_stmt
        return (output_keys, output_values)
      end
    )
end

function gen_map_values_function(func_name::Symbol,
                          map_func_names::Vector{String},
                          map_func_modules::Vector{Module},
                          map_types::Vector{DistArrayMapType},
                          map_flattens::Vector{Bool})

    loop_stmt = :(for i = 1:length(values)
                  value = values[i]
                  end)
    stmt_array_to_append = loop_stmt.args[2].args
    key_sym_counter = 1
    value_sym_counter = 1
    kv_array_counter = 1
    const key_var_prefix = "key"
    const value_var_prefix = "value"
    const kv_array_prefix = "kv_array"
    current_key_var = ""
    current_value_var = "value"
    new_key = false

    for i = 1:length(map_func_names)
        mapper_name = map_func_names[i]
        mapper_module = map_func_modules[i]
        map_type = map_types[i]
        flatten_result = map_flattens[i]

        mapper_name_sym = Symbol(mapper_name)
        mapper_module_sym = Symbol(string(mapper_module))
        current_key_sym = Symbol(current_key_var)
        current_value_sym = Symbol(current_value_var)
        output_key_sym = Symbol(key_var_prefix * "_" * string(key_sym_counter))
        output_value_sym = Symbol(value_var_prefix * "_" * string(value_sym_counter))
        output_kv_array_sym = Symbol(kv_array_prefix * "_" * string(kv_array_counter))

        new_key = false
        if map_type == DistArrayMapType_map
            if flatten_result
                mapper_call_expr = :(
                    $output_kv_array_sym = $mapper_module_sym.$mapper_name_sym(
                        $current_key_sym, $current_value_sym)
                )
            else
                mapper_call_expr = quote
                    $output_key_sym, $output_value_sym = $mapper_module_sym.$mapper_name_sym(
                        $current_key_sym, $current_value_sym)
                end
            end
            new_key = true
            key_sym_counter += 1
        elseif map_type == DistArrayMapType_map_fixed_keys
            mapper_call_expr = :(
            $output_value_sym = $mapper_module_sym.$mapper_name_sym(
                $current_key_sym, $current_value_sym)
            )
        elseif map_type == DistArrayMapType_map_values
            mapper_call_expr = :(
            $output_value_sym = $mapper_module_sym.$mapper_name_sym(
                $current_value_sym)
            )
        elseif map_type == DistArrayMapType_map_values_new_keys
            if flatten_result
                mapper_call_expr = :(
                    $output_kv_array_sym = $mapper_module_sym.$mapper_name_sym(
                        $current_key_sym, $current_value_sym)
                )
            else
                mapper_call_expr = quote
                    $output_key_sym, $output_value_sym = $mapper_module_sym.$mapper_name_sym(
                        $current_value_sym)
                end
            end
            new_key = true
            key_sym_counter += 1
        else
            @assert false
        end
        push!(stmt_array_to_append, mapper_call_expr)

        if flatten_result
            current_key_var = key_var_prefix * "_" * string(key_sym_counter)
            current_value_var = value_var_prefix * "_" * string(value_sym_counter)
            @assert new_key
            current_key_sym = Symbol(current_key_var)
            current_value_sym = Symbol(current_value_var)
            new_loop_stmt = :(
                for ($current_key_sym, $current_value_sym) in $output_kv_array_sym
                end
            )
            push!(stmt_array_to_append, new_loop_stmt)
            stmt_array_to_append = new_loop_stmt.args[2].args
         else
            if new_key
                current_key_var = string(output_key_sym)
            end
            current_value_var = string(output_value_sym)
        end
    end
    if new_key
        current_key_sym = Symbol(current_key_var)
    end
    current_value_sym = Symbol(current_value_var)

    if new_key
        append_key_stmts = quote
            key_int64 = OrionWorker.from_keys_to_int64($current_key_sym, dims)
            push!(output_keys, key_int64)
        end
        push!(stmt_array_to_append, append_key_stmts)
    end
    append_value_stmts = :(push!(output_values, $current_value_sym))
    push!(stmt_array_to_append, append_value_stmts)

    map_func = :(
        function $func_name(
            dims::Vector{Int64},
            values::Vector,
            output_value_type::DataType)::Tuple{Vector{Int64}, Vector{output_value_type}}
        rev_dims = reverse(dims)
        output_keys = Vector{Int64}()
        output_values = Vector{output_value_type}()
        $loop_stmt
        println("num_of_output_values = ", length(output_values))
        return (output_keys, output_values)
      end
    )
end

function gen_parser_function(func_name::Symbol,
                             map_func_names::Vector{String},
                             map_func_modules::Vector{Module},
                             map_types::Vector{DistArrayMapType},
                             map_flattens::Vector{Bool})
end

type DistArrayAccessRewriteContext
    iteration_var::Symbol
    dist_array_sym_to_index::Dict{Symbol, Int64}
    DistArrayAccessRewriteContext(iteration_var::Symbol,
                                  dist_array_sym_to_index::Dict{Symbol, Int64}) = new(
                                      iteration_var,
                                      dist_array_sym_to_index)
end

function rewrite_dist_array_access_visit(expr,
                                         context::DistArrayAccessRewriteContext,
                                         top_level::Integer,
                                         is_top_level::Bool,
                                         read::Bool)
    if isa(expr, Expr) &&
        expr.head == :ref
        expr_referenced = expr.args[1]
        if isa(expr_referenced, Expr) &&
            expr_referenced.head == :(.) &&
            expr_referenced.args[1] == context.iteration_var &&
            isa(expr_referenced.args[2], QuoteNode)
            if expr_referenced.args[2].value == :key
                return :(dim_keys[$(expr.args[2])])
            elseif expr_referenced.args[2].value == :value
                return :value
            end
        elseif isa(expr_referenced, Symbol)
            if !isdefined(current_module(), expr_referenced)
                return expr
            end
            ref_dist_array = eval(current_module(), expr_referenced)
            if !isa(ref_dist_array, DistArray)
                return expr
            end
            if write
                @assert false
            end
            da_index = context.dist_array_sym_to_index[expr_referenced]

            call_expr = :(dist_array_read(dist_array_partitions[$da_index]))
            index_tuple_expr = Expr(:tuple)
            index_tuple_expr.args = expr.args[2:end]
            push!(call_expr.args, index_tuple_expr)
        else
            return  expr
        end
    elseif isa(expr, Expr) &&
        expr.head == :(=)
        expr_assigned = expr.args[1]
        if isa(expr_assigned, Expr) &&
            expr_assigned.head == :ref
            expr_referenced = expr_assigned.args[1]
            if isa(expr_referenced, Symbol)
                if !isdefined(current_module(), expr_referenced)
                    return expr
                end
                ref_dist_array = eval(current_module(), expr_referenced)
                if !isa(ref_dist_array, DistArray)
                    return expr
                end
                da_index = context.dist_array_sym_to_index[expr_referenced]
                values_array = expr.args[2]
                call_expr = :(dist_array_write(dist_array_partitions[$da_index]))
                index_tuple_expr = Expr(:tuple)
                index_tuple_expr.args = expr_assigned.args[2:end]
                push!(call_expr.args, index_tuple_expr)
                push!(call_expr.args, values_array)
            else
                return expr
            end
        else
            return expr
        end
    elseif isa(expr, Expr) &&
        (expr.head == :(+=) ||
         expr.head == :(-=) ||
         expr.head == :(.*=) ||
         expr.head == :(./=))
        @assert false
    elseif isa(expr, Symbol) ||
        isa(expr, Number) ||
        isa(expr, String)
        return expr
    else
        return AstWalk.AST_WALK_RECURSIVE
    end
end

# this is a very incomplete version, just to make MF and LDA work
function gen_loop_body_function(func_name::Symbol,
                                loop_body,
                                par_for_context::ParForContext,
                                par_for_scope::ScopeContext)

    @assert isa(loop_body, Expr)
    @assert loop_body.head == :block
    func_body = deepcopy(loop_body)
    iteration_space = par_for_context.iteration_space
    iteration_space_dist_array = eval(current_module(), iteration_space)
    iter_space_value_type = iteration_space_dist_array.ValueType

    dist_array_access_dict = par_for_context.dist_array_access_dict
    accessed_dist_array_id_vec = Vector{Int32}()
    accessed_dist_array_sym_to_index = Dict{Symbol, Int32}()

    for dist_array_access_pair in dist_array_access_dict
        da_sym = dist_array_access_pair.first
        da_id = eval(current_module(), da_sym).id
        push!(accessed_dist_array_id_vec, da_id)
        accessed_dist_array_sym_to_index[da_sym] = length(accessed_dist_array_id_vec)
    end
    sort!(accessed_dist_arrays, lt = ((x, y) -> x[2] < y[2]))
    println(accessed_dist_arrays)

    dist_array_access_rewrite_context = DistArrayAccessRewriteContext(
        par_for_context.iteration_var,
        accessed_dist_array_sym_to_index)
    func_body =

    batch_loop_stmt = :(
        for i in 1:length(keys)
          key = keys[i]
          value = values[i]
          dim_keys = OrionWorker.from_int64_to_keys(key, rev_dims)
        end
    )
    for stmt in loop_body.args
        rewritten_stmt = AstWalk.ast_walk(
            deepcopy(stmt),
            rewrite_dist_array_access_visit,
            dist_array_access_rewrite_context)
        push!(batch_loop_stmt.args, rewritten_stmt)
    end
    loop_func = :(
    function $func_name(keys::Vector{Int64},
                        values::Vector{$iter_space_value_type},
                        dist_array_partitions::Vector{Ptr{Void}})
        $(batch_loop_stmt)
    end
    )
    return loop_func
end
