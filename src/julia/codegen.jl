symbol_counter = 0
function gen_unique_symbol()::Symbol
    global symbol_counter += 1
    return Symbol("oriongen_", string(symbol_counter))
end

sp_symbol_counter = 0
function gen_unique_sp_symbol()::Symbol
    global sp_symbol_counter += 1
    return Symbol("oriongen_sp_", string(sp_symbol_counter))
end

function gen_2d_partition_function(func_name::Symbol,
                                   space_partition_dim::Int64,
                                   time_partition_dim::Int64,
                                   space_dim_tile_size::Int64,
                                   time_dim_tile_size::Int64)

    space_partition_id = :(fld(dim_keys[$space_partition_dim] - 1, $space_dim_tile_size))
    time_partition_id = :(fld(dim_keys[$time_partition_dim] - 1, $time_dim_tile_size))

    add_space_partition_id_stmt = :(repartition_ids[2 * i - 1] = $(space_partition_id))
    add_time_partition_id_stmt = :(repartition_ids[2 * i] = $(time_partition_id))

    partition_func = :(
        function $func_name(keys::Vector{Int64},
                            dims::Vector{Int64})
          repartition_ids = Vector{Int32}(length(keys) * 2)
          i = 1
          dim_keys = Vector{Int64}(length(dims))
          for key in keys
            OrionWorker.from_int64_to_keys(key, dims, dim_keys)
            $add_space_partition_id_stmt
            $add_time_partition_id_stmt
            i += 1
          end
          return repartition_ids
        end)

    return partition_func
end

function gen_1d_partition_function(func_name::Symbol,
                                   partition_dim::Int64,
                                   tile_size::Int64)
    partition_id = :(fld(dim_keys[$partition_dim] - 1, $tile_size))

    add_partition_id_stmt = :(repartition_ids[i] = $(partition_id))

    partition_func = :(
        function $func_name(keys::Vector{Int64},
                            dims::Vector{Int64})
          repartition_ids = Vector{Int32}(length(keys))
          i = 1
          dim_keys = Vector{Int64}(length(dims))
          for key in keys
              OrionWorker.from_int64_to_keys(key, dims, dim_keys)
              $add_partition_id_stmt
              i += 1
          end
          return repartition_ids
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

    add_space_partition_id_stmt = :(repartition_ids[2 * i - 1] = $(space_partition_id))
    add_time_partition_id_stmt = :(repartition_ids[2 * i] = $(time_partition_id))

    partition_func = :(
        function $func_name(keys::Vector{Int64},
                            dims::Vector{Int64},
                            results::Vector{Int32})
          repartition_ids = Vector{Int32}(length(keys) * 2)
          println("keys.size() = ", length(keys),
                " typeof(OrionWorker) = ", typeof(OrionWorker))
          i = 1
          dim_keys = Vector{Int64}(length(dims))
          for key in keys
            #println(key)
            OrionWorker.from_int64_to_keys(key, dims, dim_keys)
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
                      OrionWorker.from_int64_to_keys(key, parent_dims, dim_keys_vec)
                      dim_keys = tuple(dim_keys_vec...)
                  end)
    loop_stmt_block = quote
        dim_keys_vec = Vector{Int64}(length(parent_dims))
        $(loop_stmt)
    end
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
        key_int64 = OrionWorker.from_keys_to_int64($current_key_sym, child_dims)
        push!(output_keys, key_int64)
        push!(output_values, $current_value_sym)
    end
    push!(stmt_array_to_append, append_key_value_stmts)

    map_func = :(
        function $func_name(
            parent_dims::Vector{Int64},
            child_dims::Vector{Int64},
            keys::Vector{Int64},
            values::Vector,
            output_value_type::DataType)::Tuple{Vector{Int64}, Vector{output_value_type}}
        output_keys = Vector{Int64}()
        output_values = Vector{output_value_type}()
        $loop_stmt_block
        return (output_keys, output_values)
      end
    )
    println(map_func)
    return map_func
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
        output_keys = Vector{Int64}()
        output_values = Vector{output_value_type}()
        $loop_stmt
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

function gen_dist_array_read_func_call(dist_array_id::Integer,
                                       subscripts::Tuple)
    return :(OrionWorker.dist_array_read($dist_array_id, $(subscripts)))
end

function gen_dist_array_write_func_call(dist_array_id::Integer,
                                        subscripts::Tuple,
                                        source)
    return :(OrionWorker.dist_array_write($dist_array_id, $(subscripts), $source))
end


function gen_loop_body_function(func_name::Symbol,
                                loop_body::Expr,
                                iteration_space::Symbol,
                                iteration_var::Symbol,
                                accessed_dist_arrays::Vector{Symbol},
                                accessed_dist_array_buffers::Vector{Symbol},
                                global_read_only_vars::Vector{Symbol},
                                accumulator_vars::Vector{Symbol},
                                ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}})
    @assert isa(loop_body, Expr)
    @assert loop_body.head == :block

    return_stmt = :(return $iteration_var)
    if length(accumulator_vars) >= 1
        return_args = Expr(:tuple)
        push!(return_args.args, iteration_var)
        for var_sym in accumulator_vars
            push!(return_args.args, var_sym)
        end
        return_stmt = :(return $return_args)
    end

    loop_body_func = :(
        function $func_name($iteration_var)
        $loop_body
        $return_stmt
        end
    )

    for da_sym in accessed_dist_arrays
        push!(loop_body_func.args[1].args, da_sym)
    end
    for buffer_sym in accessed_dist_array_buffers
        push!(loop_body_func.args[1].args, buffer_sym)
    end
    for var_sym in global_read_only_vars
        push!(loop_body_func.args[1].args, var_sym)
    end
    for var_sym in accumulator_vars
        push!(loop_body_func.args[1].args, var_sym)
    end
    return loop_body_func
end

function gen_loop_body_batch_function(batch_func_name::Symbol,
                                      func_name::Symbol,
                                      iter_space_value_type::Any,
                                      accessed_dist_arrays::Vector{Symbol},
                                      accessed_dist_array_buffers::Vector{Symbol},
                                      global_read_only_vars::Vector{Symbol},
                                      accumulator_vars::Vector{Symbol})
    loop_body_func_call = :(loop_body_func_call_ret = $(func_name)(key_value))

    for da_sym in accessed_dist_arrays
        push!(loop_body_func_call.args[2].args, da_sym)
    end

    for buffer_sym in accessed_dist_array_buffers
        push!(loop_body_func_call.args[2].args, buffer_sym)
    end

    for var_sym in global_read_only_vars
        push!(loop_body_func_call.args[2].args, var_sym)
    end

    update_accumulator_vars_stmts = quote end
    update_global_accumulator_vars_stmts = quote end

    renamed_accumulator_vars = Vector{Symbol}()
    if length(accumulator_vars) > 0
        update_iteration_var_value_stmt = :(values[i] = loop_body_func_call_ret[1][2])

        for var_sym in accumulator_vars
            push!(renamed_accumulator_vars, Symbol(:oriongen, var_sym))
        end

        for var_sym in renamed_accumulator_vars
            push!(loop_body_func_call.args[2].args, var_sym)
        end

        for i = 1:length(renamed_accumulator_vars)
            update_accumulator_vars_stmt = :($(renamed_accumulator_vars[i]) = loop_body_func_call_ret[$i + 1])
            push!(update_accumulator_vars_stmts.args, update_accumulator_vars_stmt)
        end

        for i = 1:length(accumulator_vars)
            update_global_accumulator_vars_stmt = :(global $(accumulator_vars[i]) = $(renamed_accumulator_vars[i]))
            push!(update_global_accumulator_vars_stmts.args, update_global_accumulator_vars_stmt)
        end
    else
        update_iteration_var_value_stmt = :(values[i] = loop_body_func_call_ret[2])
    end

    batch_loop_stmt = quote
        dim_keys = Vector{Int64}(length(dims))
        start = 1 + offset
        for i in start:(start + num_elements - 1)
            key = keys[i]
            value = values[i]
            OrionWorker.from_int64_to_keys(key, dims, dim_keys)

            key_value = (dim_keys, value)
            $(loop_body_func_call)
            $(update_iteration_var_value_stmt)
            $(update_accumulator_vars_stmts)
        end
    end

    batch_func = :(
        function $batch_func_name(keys::Vector{Int64},
                                  values::Vector{$iter_space_value_type},
                                  dims::Vector{Int64},
                                  offset::UInt64,
                                  num_elements::UInt64)
        $(batch_loop_stmt)
        $(update_global_accumulator_vars_stmts)
        end
    )

    for da_sym in accessed_dist_arrays
        push!(batch_func.args[1].args, da_sym)
    end

    for buffer_sym in accessed_dist_array_buffers
        push!(batch_func.args[1].args, buffer_sym)
    end

    for var_sym in global_read_only_vars
        push!(batch_func.args[1].args, var_sym)
    end

    for var_sym in renamed_accumulator_vars
        push!(batch_func.args[1].args, var_sym)
    end

    return batch_func
end

function gen_loop_body_batch_function_iter_dims(batch_func_name::Symbol,
                                                func_name::Symbol,
                                                iter_space_value_type::Any,
                                                iterate_dims_length::Int64,
                                                accessed_dist_arrays::Vector{Symbol},
                                                accessed_dist_array_buffers::Vector{Symbol},
                                                global_read_only_vars::Vector{Symbol},
                                                accumulator_vars::Vector{Symbol})

    loop_body_func_call = :(loop_body_func_call_ret = $(func_name)(:key_value))

    for da_sym in accessed_dist_arrays
        push!(loop_body_func_call.args[2].args, da_sym)
    end

    for buffer_sym in accessed_dist_array_buffers
        push!(loop_body_func_call.args[2].args, buffer_sym)
    end

    for var_sym in global_read_only_vars
        push!(loop_body_func_call.args[2].args, var_sym)
    end

    update_accumulator_vars_stmts = quote end
    update_global_accumulator_vars_stmts = quote end

    renamed_accumulator_vars = Vector{Symbol}()
    if length(accumulator_vars) > 0
        update_iteration_var_value_stmt = :(values[(i - length(value_vec)):(i - 1)] = loop_body_func_call_ret[1][2])
        for var_sym in accumulator_vars
            push!(renamed_accumulator_vars, Symbol(:oriongen, var_sym))
        end

        for var_sym in renamed_accumulator_vars
            push!(loop_body_func_call.args[2].args, var_sym)
        end

        for i = 1:length(renamed_accumulator_vars)
            update_accumulator_vars_stmt = :($(renamed_accumulator_vars[i]) = loop_body_func_call_ret[$i + 1])
            push!(update_accumulator_vars_stmts.args, update_accumulator_vars_stmt)
        end

        for i = 1:length(accumulator_vars)
            update_global_accumulator_vars_stmt = :(global $(accumulator_vars[i]) = $(renamed_accumulator_vars[i]))
            push!(update_global_accumulator_vars_stmts.args, update_global_accumulator_vars_stmt)
        end
    else
        update_iteration_var_value_stmt = :(values[(i - length(value_vec)):(i - 1)] = loop_body_func_call_ret[2])
    end

    batch_loop_stmt = quote
        if num_elements == 0
          return
        end
        start = 1 + offset
        first_key = keys[start]
        first_dim_keys = OrionWorker.from_int64_to_keys(first_key, dims)
        dim_keys = Vector{Int64}(length(dims))
        prefix = first_dim_keys[(end - $iterate_dims_length):end]
        dim_keys_vec = Vector{Vector{Int64}}()
        value_vec = Vector{$iter_space_value_type}()
        for i in start:(start + num_elements - 1)
            key = keys[i]
            value = values[i]
            OrionWorker.from_int64_to_keys(key, dims, dim_keys)
            curr_prefix = dim_keys[(end - $iterate_dims_length):end]

            if curr_prefix == prefix
                push!(dim_keys_vec, dim_keys)
                push!(value_vec, value)
            else
                key_value = (dim_keys_vec, value_vec)
                $(loop_body_func_call)
                $(update_iteration_var_value_stmt)
                $(update_accumulator_vars_stmts)
                dim_keys_vec = [dim_keys]
                value_vec = [value]
                prefix = curr_prefix
            end
        end
    end

    batch_func = :(
    function $batch_func_name(keys::Vector{Int64},
                              values::Vector{$iter_space_value_type},
                              dims::Vector{Int64},
                              offset::UInt64,
                              num_elements::UInt64)
        $(batch_loop_stmt)
        $(update_global_accumulator_vars_stmts)
        end
    )
    for da_sym in accessed_dist_arrays
        push!(batch_func.args[1].args, da_sym)
    end

    for buffer_sym in accessed_dist_array_buffers
        push!(batch_func.args[1].args, buffer_sym)
    end

    for var_sym in global_read_only_vars
        push!(batch_func.args[1].args, var_sym)
    end

    for var_sym in renamed_accumulator_vars
        push!(batch_func.args[1].args, var_sym)
    end
    return batch_func
end

function gen_prefetch_function(prefetch_func_name::Symbol,
                               iterate_var::Symbol,
                               prefetch_stmts::Expr,
                               global_read_only_vars::Vector{Symbol},
                               accumulator_vars::Vector{Symbol})
    prefetch_func = :(
        function $prefetch_func_name($iterate_var,
                                     oriongen_prefetch_point_dict::Dict{Int32, OrionWorker.DistArrayAccessSetRecorder})
            $prefetch_stmts
        end
    )

    for var_sym in global_read_only_vars
        push!(prefetch_func.args[1].args, var_sym)
    end
    for var_sym in accumulator_vars
        push!(prefetch_func.args[1].args, var_sym)
    end
    return prefetch_func
end

function gen_prefetch_batch_function(prefetch_batch_func_name::Symbol,
                                     prefetch_func_name::Symbol,
                                     iter_space_value_type::Any,
                                     global_read_only_vars::Vector{Symbol},
                                     accumulator_vars::Vector{Symbol})

    prefetch_func_call = :($(prefetch_func_name)(key_value, prefetch_point_dict))
    for var_sym in global_read_only_vars
        push!(prefetch_func_call.args, var_sym)
    end
    for var_sym in accumulator_vars
        push!(prefetch_func_call.args, var_sym)
    end

    prefetch_batch_func = :(
        function $prefetch_batch_func_name(keys::Vector{Int64},
                                           values::Vector{$iter_space_value_type},
                                           dims::Vector{Int64},
                                           global_indexed_dist_array_ids::Vector{Int32},
                                           global_indexed_dist_array_dims::Vector{Vector{Int64}},
                                           offset::UInt64,
                                           num_elements::UInt64)::Vector{Vector{Int64}}
            prefetch_point_dict = Dict{Int32, OrionWorker.DistArrayAccessSetRecorder}()
            for idx in eachindex(global_indexed_dist_array_ids)
                dist_array_id = global_indexed_dist_array_ids[idx]
                dist_array_dims = global_indexed_dist_array_dims[idx]
                prefetch_point_dict[dist_array_id] = OrionWorker.DistArrayAccessSetRecorder{length(dist_array_dims)}(dist_array_dims)
            end
            dim_keys = Vector{Int64}(length(dims))
            start = 1 + offset
            for i in start:(start + num_elements - 1)
                key = keys[i]
                value = values[i]
                OrionWorker.from_int64_to_keys(key, dims, dim_keys)

                key_value = (dim_keys, value)
                $(prefetch_func_call)
            end

            prefetch_point_array = Vector{Vector{Int64}}()
            for id in global_indexed_dist_array_ids
                point_set = prefetch_point_dict[id].keys_set
                push!(prefetch_point_array, collect(point_set))
            end
            return prefetch_point_array
        end
    )

    for var_sym in global_read_only_vars
        push!(prefetch_batch_func.args[1].args[1].args, var_sym)
    end
    for var_sym in accumulator_vars
        push!(prefetch_batch_func.args[1].args[1].args, var_sym)
    end

    return prefetch_batch_func
end


function gen_prefetch_batch_function_iter_dims(prefetch_batch_func_name::Symbol,
                                               prefetch_func_name::Symbol,
                                               iter_space_value_type::Any,
                                               iterate_dims_length::Int64,
                                               global_read_only_vars::Vector{Symbol},
                                               accumulator_vars::Vector{Symbol})

    prefetch_func_call = :($(prefetch_func_name)(key_value, prefetch_point_dict))
    for var_sym in global_read_only_vars
        push!(prefetch_func_call.args, var_sym)
    end
    for var_sym in accumulator_vars
        push!(prefetch_func_call.args, var_sym)
    end

    prefetch_batch_func = :(
            function $prefetch_batch_func_name(keys::Vector{Int64},
                                               values::Vector{$iter_space_value_type},
                                               dims::Vector{Int64},
                                               global_indexed_dist_array_ids::Vector{Int32},
                                               global_indexed_dist_array_dims::Vector{Vector{Int64}},
                                               offset::UInt64,
                                               num_elements::UInt64)::Vector{Vector{Int64}}
            prefetch_point_dict = Dict{Int32, OrionWorker.DistArrayAccessSetRecorder}()
            for idx in eachindex(global_indexed_dist_array_ids)
                id = global_indexed_dist_array_ids[idx]
                dims = global_indexed_dist_array_dims[idx]
                prefetch_point_dict[id] = OrionWorker.DistArrayAccessSetRecorder{length(dims)}(dims)
            end

            if num_elements == 0
                return Vector{Vector{Int64}}()
            end

            start = 1 + offset
            first_key = keys[start]
            first_dim_keys = OrionWorker.from_int64_to_keys(first_key, dims)
            prefix = first_dim_keys[(end - $iterate_dims_length):end]
            dim_keys_vec = Vector{Vector{Int64}}()
            value_vec = Vector{$iter_space_value_type}()
            for i in start:(start + num_elements - 1)
                key = keys[i]
                value = values[i]
                dim_keys = OrionWorker.from_int64_to_keys(key, dims)
                curr_prefix = dim_keys[(end - $iterate_dims_length):end]

                if curr_prefix == prefix
                    push!(dim_keys_vec, dim_keys)
                    push!(value_vec, value)
                else
                    key_value = (dim_keys_vec, value_vec)
                    $(prefetch_func_call)
                    dim_keys_vec = [dim_keys]
                    value_vec = [value]
                    prefix = curr_prefix
                end
            end

            prefetch_point_array = Vector{Vector{Int64}}()
            for id in global_indexed_dist_array_ids
                point_set = prefetch_point_dict[id].keys_set
                push!(prefetch_point_array, collect(point_set))
            end
            return prefetch_point_array
        end
    )

    for var_sym in global_read_only_vars
        push!(prefetch_batch_func.args[1].args[1].args, var_sym)
    end
    for var_sym in accumulator_vars
        push!(prefetch_batch_func.args[1].args[1].args, var_sym)
    end

    return prefetch_batch_func
end

function gen_access_count_function(access_count_func_name::Symbol,
                                   iterate_var::Symbol,
                                   access_stmts::Expr)
    access_count_func = :(
        function $access_count_func_name($iterate_var,
                                         oriongen_access_count_dict::Dict{Int64, OrionWorker.DistArrayAccessCountRecorder})
            $access_stmts
        end
    )
    return access_count_func
end

function gen_access_count_batch_function(access_count_batch_func_name::Symbol,
                                         access_count_func_name::Symbol,
                                         iter_space_value_type::DataType)
        access_count_batch_func = :(
            function $access_count_batch_func_name(keys::Vector{Int64},
                                                   values::Vector{$iter_space_value_type},
                                                   dims::Vector{Int64},
                                                   global_indexed_dist_array_ids::Vector{Int32},
                                                   global_indexed_dist_array_dims::Vector{Vector{Int64}}
                                                   )::Vector{Vector{Tuple{Int64, UInt64}}}
            access_count_dict = Dict{Int32, OrionWorker.DistArrayAccessCountRecorder}()
            for idx in eachindex(global_indexed_dist_array_ids)
                id = global_indexed_dist_array_ids[idx]
                dims = global_indexed_dist_array_dims[idx]
                access_count_dict[id] = OrionWorker.DistArrayAccessCountRecorder(dims)
            end

            for i in 1:length(keys)
                key = keys[i]
                value = values[i]
                dim_keys = OrionWorker.from_int64_to_keys(key, dims)

                key_value = (dim_keys, value)
                $(access_count_func_name)(key_value, access_count_dict)
            end

            access_count_array = Vector{Vector{Tuple{Int64, UInt64}}}()
            for id in global_indexed_dist_array_ids
                access_count = access_count_dict[id].keys_dict
                access_count_vec = Vector{Tuple{Int64, UInt64}}()
                for ac_key in keys(access_count)
                    push!(access_count_vec, (ac_key, access_count[ac_key]))
                end
                push!(access_count_array, access_count_vec)
            end
            return access_count_array
        end
    )

    return access_count_batch_func
end

function gen_access_count_batch_function_iter_dims(access_count_batch_func_name::Symbol,
                                                   access_count_func_name::Symbol,
                                                   iter_space_value_type::DataType,
                                                   iterate_dims_length::Int64)
    access_count_batch_func = :(
        function $access_count_batch_func_name(keys::Vector{Int64},
                                               values::Vector{$iter_space_value_type},
                                               dims::Vector{Int64},
                                               global_indexed_dist_array_ids::Vector{Int32},
                                               global_indexed_dist_array_dims::Vector{Vector{Int64}}
                                               )::Vector{Vector{Tuple{Int64, UInt64}}}
            access_count_dict = Dict{Int32, OrionWorker.DistArrayAccessCountRecorder}()
            for idx in eachindex(global_indexed_dist_array_ids)
                id = global_indexed_dist_array_ids[idx]
                dims = global_indexed_dist_array_dims[idx]
                access_count_dict[id] = OrionWorker.DistArrayAccessCountRecorder(dims)
            end

            if length(keys) > 0
                first_key = keys[1]
                first_dim_keys = OrionWorker.from_int64_to_keys(key, dims)
                prefix = first_dim_keys[(end - $iterate_dims_length):end]
                dim_keys_vec = Vector{Vector{Int64}}()
                value_vec = Vector{$iter_space_value_type}()
                for i in 1:length(keys)
                    key = keys[i]
                    value = values[i]
                    dim_keys = OrionWorker.from_int64_to_keys(key, dims)
                    curr_prefix = dim_keys[(end - $iterate_dims_length):end]

                    if curr_prefix == prefix
                        push!(dim_keys_vec, dim_keys)
                        push!(value_vec, value)
                    else
                        key_value = (dim_keys_vec, value_vec)
                        $(access_count_func_name)(key_value, access_count_dict)
                        dim_keys_vec = [dim_keys]
                        value_vec = [value]
                        prefix = curr_prefix
                    end
                end
            end

            access_count_array = Vector{Vector{Tuple{Int64, UInt64}}}()
            for id in global_indexed_dist_array_ids
                access_count = access_count_dict[id].keys_dict
                access_count_vec = Vector{Tuple{Int64, UInt64}}()
                for ac_key in keys(access_count)
                    push!(access_count_vec, (ac_key, access_count[ac_key]))
                end
                push!(access_count_array, access_count_vec)
            end
            return access_count_array
        end
    )

    return access_count_batch_func
end

function apply_buffered_updates_gen_get_value_stmts(value_var_sym::Symbol,
                                                    key_sym::Symbol,
                                                    start_index_sym::Symbol,
                                                    index_sym::Symbol,
                                                    key_vec_sym::Symbol,
                                                    value_vec_sym::Symbol)
    stmts = quote
        $index_sym = orionres_get_index($key_vec_sym, $key_sym, $start_index_sym)
        @assert $index_sym >= 1 #string($key_vec_sym) * " " * string($key_sym) * " " * string($start_index_sym)
        $value_var_sym = $value_vec_sym[$index_sym]
        $start_index_sym = $index_sym
    end
    return stmts.args
end

function gen_apply_batch_buffered_updates_func(batch_func_name::Symbol,
                                               apply_buffered_updates_func_name::Symbol,
                                               num_helper_dist_arrays::Int64,
                                               num_helper_dist_array_buffers::Int64)
    dist_array_val_sym = :dist_array_value
    apply_buffered_updates_func_call = :(ret = $apply_buffered_updates_func_name(update_key, $dist_array_val_sym, update_val))

    helper_dist_array_val_base_sym = gen_unique_symbol()
    helper_dist_array_val_base_sym_str = string(helper_dist_array_val_base_sym)
    helper_dist_array_val_sym_vec = Vector{Symbol}()

    for i = 1:num_helper_dist_arrays
        var_sym = Symbol(helper_dist_array_val_base_sym_str * "_" * string(i))
        push!(apply_buffered_updates_func_call.args[2].args, var_sym)
        push!(helper_dist_array_val_sym_vec, var_sym)
    end

    helper_dist_array_buffer_val_base_sym = gen_unique_symbol()
    helper_dist_array_buffer_val_base_sym_str = string(helper_dist_array_buffer_val_base_sym)
    helper_dist_array_buffer_val_sym_vec = Vector{Symbol}()

    helper_dist_array_buffer_index_base_sym = gen_unique_symbol()
    helper_dist_array_buffer_index_base_sym_str = string(helper_dist_array_buffer_index_base_sym)
    helper_dist_array_buffer_index_sym_vec = Vector{Symbol}()

    helper_dist_array_buffer_start_index_base_sym = gen_unique_symbol()
    helper_dist_array_buffer_start_index_base_sym_str = string(helper_dist_array_buffer_start_index_base_sym)
    helper_dist_array_buffer_start_index_sym_vec = Vector{Symbol}()

    for i = 1:num_helper_dist_array_buffers
        var_sym = Symbol(helper_dist_array_buffer_val_base_sym_str * "_" * string(i))
        index_sym = Symbol(helper_dist_array_buffer_index_base_sym_str * "_" * string(i))
        start_index_sym = Symbol(helper_dist_array_buffer_start_index_base_sym_str * "_" * string(i))
        push!(apply_buffered_updates_func_call.args[2].args, var_sym)
        push!(helper_dist_array_buffer_val_sym_vec, var_sym)
        push!(helper_dist_array_buffer_index_sym_vec, index_sym)
        push!(helper_dist_array_buffer_start_index_sym_vec, start_index_sym)
    end

    init_start_index_stmts = Vector{Expr}()
    for i = 1:num_helper_dist_array_buffers
        push!(init_start_index_stmts, :($(helper_dist_array_buffer_start_index_sym_vec[i]) = 1))
    end

    helper_dist_array_base_sym = gen_unique_symbol()
    helper_dist_array_base_sym_str = string(helper_dist_array_base_sym)
    helper_dist_array_base_val_sym_str = helper_dist_array_base_sym_str * "_vals_"
    helper_dist_array_vals_sym_vec = Vector{Symbol}()

    for i = 1:num_helper_dist_arrays
        value_sym = Symbol(helper_dist_array_base_val_sym_str * string(i))
        push!(helper_dist_array_vals_sym_vec, value_sym)
    end

    helper_dist_array_buffer_base_sym = gen_unique_symbol()
    helper_dist_array_buffer_base_sym_str = string(helper_dist_array_buffer_base_sym)
    helper_dist_array_buffer_base_key_sym_str = helper_dist_array_buffer_base_sym_str * "_keys_"
    helper_dist_array_buffer_base_val_sym_str = helper_dist_array_buffer_base_sym_str * "_vals_"
    helper_dist_array_buffer_keys_sym_vec = Vector{Symbol}()
    helper_dist_array_buffer_vals_sym_vec = Vector{Symbol}()

    for i = 1:num_helper_dist_array_buffers
        key_sym = Symbol(helper_dist_array_buffer_base_key_sym_str * string(i))
        push!(helper_dist_array_buffer_keys_sym_vec, key_sym)
        value_sym = Symbol(helper_dist_array_buffer_base_val_sym_str * string(i))
        push!(helper_dist_array_buffer_vals_sym_vec, value_sym)
    end

    apply_buffered_updates_loop = :(
        for update_index in eachindex(buffered_updates_keys)
            update_key = buffered_updates_keys[update_index]
            update_val = buffered_updates_values[update_index]
        end
    )

    get_value_stmt = :($dist_array_val_sym = dist_array_values[update_index])
    push!(apply_buffered_updates_loop.args[2].args, get_value_stmt)
    for i = 1:num_helper_dist_arrays
        var_sym = helper_dist_array_val_sym_vec[i]
        get_value_stmt = :($var_sym = $(helper_dist_array_vals_sym_vec[i])[update_index])
        push!(apply_buffered_updates_loop.args[2].args, get_value_stmt)
    end

    for i = 1:num_helper_dist_array_buffers
        var_sym = helper_dist_array_buffer_val_sym_vec[i]
        index_sym = helper_dist_array_buffer_index_sym_vec[i]
        start_index_sym = helper_dist_array_buffer_start_index_sym_vec[i]

        get_value_stmts = apply_buffered_updates_gen_get_value_stmts(var_sym,
                                                                     :update_key,
                                                                     start_index_sym,
                                                                     index_sym,
                                                                     helper_dist_array_buffer_keys_sym_vec[i],
                                                                     helper_dist_array_buffer_vals_sym_vec[i])
        append!(apply_buffered_updates_loop.args[2].args, get_value_stmts)
    end

    push!(apply_buffered_updates_loop.args[2].args, apply_buffered_updates_func_call)
    if num_helper_dist_arrays == 0
        push!(apply_buffered_updates_loop.args[2].args, :(dist_array_values[update_index] = ret))
     else
        push!(apply_buffered_updates_loop.args[2].args, :(dist_array_values[update_index] = ret[1]))
        for i = 1:num_helper_dist_arrays
            push!(apply_buffered_updates_loop.args[2].args, :($(helper_dist_array_vals_sym_vec[i])[update_index] = ret[$(i + 1)]))
        end
    end
    apply_batch_buffered_updates_func = :(
      function $batch_func_name(buffered_updates_keys::Vector{Int64},
                                buffered_updates_values::Vector,
                                dist_array_values::Vector)
        end
    )
    for i = 1:num_helper_dist_arrays
        push!(apply_batch_buffered_updates_func.args[1].args, helper_dist_array_vals_sym_vec[i])
    end
    for i = 1:num_helper_dist_array_buffers
        push!(apply_batch_buffered_updates_func.args[1].args, helper_dist_array_buffer_keys_sym_vec[i])
        push!(apply_batch_buffered_updates_func.args[1].args, helper_dist_array_buffer_vals_sym_vec[i])
    end
    append!(apply_batch_buffered_updates_func.args[2].args, init_start_index_stmts)
    push!(apply_batch_buffered_updates_func.args[2].args, apply_buffered_updates_loop)
    return apply_batch_buffered_updates_func
end
