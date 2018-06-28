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

function gen_2d_histogram_partition_function(func_name::Symbol,
                                             space_partition_dim::Int64,
                                             time_partition_dim::Int64,
                                             space_partition_bounds::Vector{Int64},
                                             time_partition_bounds::Vector{Int64})

    partition_func = :(
        function $func_name(keys::Vector{Int64},
                            dims::Vector{Int64})
          repartition_ids = Vector{Int32}(length(keys) * 2)
          dim_keys = Vector{Int64}(length(dims))
          space_partition_bounds = $space_partition_bounds
          time_partition_bounds = $time_partition_bounds

          for index in eachindex(keys)
              key = keys[index]
              OrionWorker.from_int64_to_keys(key, dims, dim_keys)
              space_dim_key = dim_keys[$space_partition_dim] - 1
              space_partition_pos = Base.Sort.searchsortedfirst(space_partition_bounds, space_dim_key)
              space_partition_index = space_partition_pos - 1
              repartition_ids[index * 2 - 1] = space_partition_index

              time_dim_key = dim_keys[$time_partition_dim] - 1
              time_partition_pos = Base.Sort.searchsortedfirst(time_partition_bounds, time_dim_key)
              time_partition_index = time_partition_pos - 1
              repartition_ids[index * 2] = time_partition_index
        end
          return repartition_ids
        end)

    return partition_func
end

function gen_1d_histogram_partition_function(func_name::Symbol,
                                             partition_dim::Int64,
                                             partition_bounds::Vector{Int64})
    partition_func = :(
        function $func_name(keys::Vector{Int64},
                            dims::Vector{Int64})
          repartition_ids = Vector{Int32}(length(keys))
          dim_keys = Vector{Int64}(length(dims))
          partition_bounds = $partition_bounds
          for index in eachindex(keys)
              key = keys[index]
              OrionWorker.from_int64_to_keys(key, dims, dim_keys)
              dim_key = dim_keys[$partition_dim] - 1
              partition_pos = Base.Sort.searchsortedfirst(partition_bounds, dim_key)
              partition_index = partition_pos - 1
              repartition_ids[index] = partition_index
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

function gen_loop_body_batch_function(batch_func_name::Symbol,
                                      loop_body::Expr,
                                      iteration_space::Symbol,
                                      iteration_var_key::Symbol,
                                      iteration_var_val::Symbol,
                                      iter_space_value_type::Any,
                                      accessed_dist_arrays::Vector{Symbol},
                                      accessed_dist_array_buffers::Vector{Symbol},
                                      global_read_only_vars::Vector{Symbol},
                                      accumulator_vars::Vector{Symbol},
                                      renamed_accumulator_vars_map::Dict{Symbol, Symbol},
                                      update_global_accumulator_vars_stmt::Expr,
                                      is_iteration_var_val_reassigned::Bool)
    @assert isa(loop_body, Expr)
    @assert loop_body.head == :block

    remapped_loop_body = AstWalk.ast_walk(loop_body, remap_symbols_visit, renamed_accumulator_vars_map)

    reassign_iteration_var_val_stmt = nothing
    if is_iteration_var_val_reassigned
        reassign_iteration_var_val_stmt = :( oriongen_values[oriongen_i] = $iteration_var_val)
    end

    batch_loop_stmt = quote
        oriongen_start = 1 + oriongen_offset
        for oriongen_i in oriongen_start:(oriongen_start + oriongen_num_elements - 1)
            oriongen_key = oriongen_keys[oriongen_i]
            $iteration_var_val = oriongen_values[oriongen_i]
            OrionWorker.from_int64_to_keys(oriongen_key, oriongen_dims, $iteration_var_key)
            $remapped_loop_body
            $reassign_iteration_var_val_stmt
        end
    end



    batch_func = :(
        function $batch_func_name(oriongen_keys::Vector{Int64},
                                  oriongen_values::Vector{$iter_space_value_type},
                                  oriongen_dims::Vector{Int64},
                                  oriongen_offset::UInt64,
                                  oriongen_num_elements::UInt64,
                                  $iteration_var_key::Vector{Int64})
        $(batch_loop_stmt)
        $(update_global_accumulator_vars_stmt)
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

    for var_sym in accumulator_vars
        push!(batch_func.args[1].args, renamed_accumulator_vars_map[var_sym])
    end

    return batch_func
end

function gen_loop_body_batch_function_iter_dims(batch_func_name::Symbol,
                                                loop_body::Expr,
                                                iteration_space::Symbol,
                                                iteration_var_key::Symbol,
                                                iteration_var_val::Symbol,
                                                iter_space_value_type::Any,
                                                iterate_dims_length::Int64,
                                                accessed_dist_arrays::Vector{Symbol},
                                                accessed_dist_array_buffers::Vector{Symbol},
                                                global_read_only_vars::Vector{Symbol},
                                                accumulator_vars::Vector{Symbol},
                                                renamed_accumulator_vars_map::Dict{Symbol, Symbol},
                                                update_global_accumulator_vars_stmt::Expr,
                                                is_iteration_var_val_reassigned::Bool)

    @assert isa(loop_body, Expr)
    @assert loop_body.head == :block

    remapped_loop_body = AstWalk.ast_walk(loop_body, remap_symbols_visit, renamed_accumulator_vars_map)

    reassign_iteration_var_val_stmt = nothing
    if is_iteration_var_val_reassigned
        reassign_iteration_var_val_stmt = :(
            for oriongen_j in eachindex($iteration_var_val)
                oriongen_values[oriongen_i - length($iteration_var_val) - 1 + oriongen_j] = $(iteration_var_val)[oriongen_j]
            end
        )
    end
    batch_loop_stmt = quote
        oriongen_start = 1 + oriongen_offset
        oriongen_first_key = oriongen_keys[oriongen_start]
        oriongen_first_dim_keys = OrionWorker.from_int64_to_keys(oriongen_first_key, oriongen_dims)
        oriongen_prefix = oriongen_first_dim_keys[(end - $iterate_dims_length):end]
        $iteration_var_key = Vector{Vector{Int64}}()
        $iteration_var_val = Vector{$iter_space_value_type}()

        for oriongen_i in oriongen_start:(oriongen_start + oriongen_num_elements - 1)
            oriongen_key = oriongen_keys[i]
            oriongen_value = oriongen_values[i]
            oriongen_curr_dim_keys = OrionWorker.from_int64_to_keys(oriongen_key, oriongen_dims)
            oriongen_curr_prefix = oriongen_dim_keys[(end - $iterate_dims_length):end]

            if oriongen_curr_prefix == oriongen_prefix
                push!($iteration_var_key, oriongen_curr_dim_keys)
                push!($iteration_var_val, oriongen_value)
            else
                $remapped_loop_body
                $reassign_iteration_var_val_stmt

                $iteration_var_key = Vector{Vector{Int64}}()
                $iteration_var_val = Vector{$iter_space_value_type}()
                oriongen_prefix = oriongen_curr_prefix
            end
        end
    end

    batch_func = :(
    function $batch_func_name(oriongen_keys::Vector{Int64},
                              oriongen_values::Vector{$iter_space_value_type},
                              oriongen_dims::Vector{Int64},
                              oriongen_offset::UInt64,
                              oriongen_num_elements::UInt64,
                              oriongen_dim_keys::Vector{Int64})
        $(batch_loop_stmt)
        $(update_global_accumulator_vars_stmt)
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

    for var_sym in accumulator_vars
        push!(batch_func.args[1].args, renamed_accumulator_vars_map[var_sym])
    end

    return batch_func
end

function gen_prefetch_batch_function(prefetch_batch_func_name::Symbol,
                                     iteration_var_key::Symbol,
                                     iteration_var_val::Symbol,
                                     prefetch_stmts::Expr,
                                     iter_space_value_type::Any,
                                     global_read_only_vars::Vector{Symbol},
                                     accumulator_vars::Vector{Symbol},
                                     renamed_accumulator_vars_map::Dict{Symbol, Symbol})

    remapped_prefetch_stmts = AstWalk.ast_walk(prefetch_stmts, remap_symbols_visit, renamed_accumulator_vars_map)
    prefetch_batch_func = :(
        function $prefetch_batch_func_name(oriongen_keys::Vector{Int64},
                                           oriongen_values::Vector{$iter_space_value_type},
                                           oriongen_dims::Vector{Int64},
                                           oriongen_global_indexed_dist_array_ids::Vector{Int32},
                                           oriongen_global_indexed_dist_array_dims::Vector{Vector{Int64}},
                                           oriongen_offset::UInt64,
                                           oriongen_num_elements::UInt64,
                                           $iteration_var_key::Vector{Int64})::Vector{Vector{Int64}}
            oriongen_prefetch_point_dict = Dict{Int32, OrionWorker.DistArrayAccessSetRecorder}()
            for oriongen_idx in eachindex(oriongen_global_indexed_dist_array_ids)
                oriongen_dist_array_id = oriongen_global_indexed_dist_array_ids[oriongen_idx]
                oriongen_dist_array_dims = oriongen_global_indexed_dist_array_dims[oriongen_idx]
                oriongen_prefetch_point_dict[oriongen_dist_array_id] = OrionWorker.DistArrayAccessSetRecorder{length(oriongen_dist_array_dims)}(oriongen_dist_array_dims)
            end
            oriongen_dim_keys = Vector{Int64}(length(oriongen_dims))
            oriongen_start = 1 + oriongen_offset
            for oriongen_i in oriongen_start:(oriongen_start + oriongen_num_elements - 1)
                oriongen_key = oriongen_keys[oriongen_i]
                $iteration_var_val = oriongen_values[oriongen_i]
                OrionWorker.from_int64_to_keys(oriongen_key, oriongen_dims, $iteration_var_key)
                $remapped_prefetch_stmts
            end

            oriongen_prefetch_point_array = Vector{Vector{Int64}}()
            for oriongen_id in oriongen_global_indexed_dist_array_ids
                oriongen_point_set = oriongen_prefetch_point_dict[oriongen_id].keys_set
                push!(oriongen_prefetch_point_array, collect(oriongen_point_set))
            end
            return oriongen_prefetch_point_array
        end
    )

    for var_sym in global_read_only_vars
        push!(prefetch_batch_func.args[1].args[1].args, var_sym)
    end
    for var_sym in accumulator_vars
        push!(prefetch_batch_func.args[1].args[1].args, renamed_accumulator_vars_map[var_sym])
    end

    return prefetch_batch_func
end


function gen_prefetch_batch_function_iter_dims(prefetch_batch_func_name::Symbol,
                                               iteration_var_key::Symbol,
                                               iteration_var_val::Symbol,
                                               prefetch_stmts::Expr,
                                               iter_space_value_type::Any,
                                               iterate_dims_length::Int64,
                                               global_read_only_vars::Vector{Symbol},
                                               accumulator_vars::Vector{Symbol},
                                               renamed_accumulator_vars_map::Dict{Symbol, Symbol})

    remapped_prefetch_stmts = AstWalk.ast_walk(prefetch_stmts, remap_symbols_visit, renamed_accumulator_vars_map)
    prefetch_batch_func = :(
            function $prefetch_batch_func_name(oriongen_keys::Vector{Int64},
                                               oriongen_values::Vector{$iter_space_value_type},
                                               oriongen_dims::Vector{Int64},
                                               oriongen_global_indexed_dist_array_ids::Vector{Int32},
                                               oriongen_global_indexed_dist_array_dims::Vector{Vector{Int64}},
                                               oriongen_offset::UInt64,
                                               oriongen_num_elements::UInt64,
                                               $iteration_var_key::Vector{Int64})::Vector{Vector{Int64}}
            oriongen_prefetch_point_dict = Dict{Int32, OrionWorker.DistArrayAccessSetRecorder}()
            for oriongen_idx in eachindex(oriongen_global_indexed_dist_array_ids)
                oriongen_id = oriongen_global_indexed_dist_array_ids[oriongen_idx]
                oriongen_dims = oriongen_global_indexed_dist_array_dims[oriongen_idx]
                oriongen_prefetch_point_dict[oriongen_dist_array_id] = OrionWorker.DistArrayAccessSetRecorder{length(oriongen_dist_array_dims)}(oriongen_dist_array_dims)
            end

            oriongen_start = 1 + oriongen_offset
            oriongen_first_key = oriongen_keys[oriongebn_start]
            oriongen_first_dim_keys = OrionWorker.from_int64_to_keys(oriongen_first_key, oriongen_dims)
            oriongen_prefix = oriongen_first_dim_keys[(end - $iterate_dims_length):end]
            $iteration_var_key = Vector{Vector{Int64}}()
            $iteration_var_val = Vector{$iter_space_value_type}()
            for oriongen_i in oriongen_start:(oriongen_start + oriongen_num_elements - 1)
                oriongen_key = oriongen_keys[oriongen_i]
                oriongen_value = oriongen_values[oriongen_i]
                oriongen_dim_keys = OrionWorker.from_int64_to_keys(oriongen_key, oriongen_dims)
                oriongen_curr_prefix = oriongen_dim_keys[(end - $iterate_dims_length):end]

                if oriongen_curr_prefix == oriongen_prefix
                    push!($iteration_var_key, oriongen_curr_dim_keys)
                    push!($iteration_var_val, oriongen_value)
                else
                     $remapped_prefetch_stmts
                     for oriongen_j in eachindex($iteration_var_val)
                         oriongen_values[oriongen_i - length($iteration_var_val) - 1 + oriongen_j] = $(iteration_var_val)[oriongen_j]
                     end

                     $iteration_var_key = Vector{Vector{Int64}}()
                     $iteration_var_val = Vector{$iter_space_value_type}()
                     oriongen_prefix = oriongen_curr_prefix
                 end
             end

            oriongen_prefetch_point_array = Vector{Vector{Int64}}()
            for oriongen_id in oriongen_global_indexed_dist_array_ids
                origen_point_set = oriongen_prefetch_point_dict[id].keys_set
                push!(oriongen_prefetch_point_array, collect(oriongen_point_set))
            end
            return oriongen_prefetch_point_array
        end
    )

    dump(prefetch_batch_func)
    for var_sym in global_read_only_vars
        push!(prefetch_batch_func.args[1].args[1].args, var_sym)
    end

    for var_sym in accumulator_vars
        push!(prefetch_batch_func.args[1].args[1].args, renamed_accumulator_vars_map[var_sym])
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

function gen_group_by_key_batch_function(key_func_name::Symbol,
                                         key_batch_func_name::Symbol)
    key_batch_func = :(
        function $key_batch_func_name(keys::Vector{Int64},
                                      dims::Vector{Int64})
        group_keys = Vector{UInt64}(length(keys))
        dim_keys_vec = Vector{Int64}(length(dims))
        for i in eachindex(keys)
            key = keys[i]
            OrionWorker.from_int64_to_keys(key, dims, dim_keys_vec)
            dim_keys = tuple(dim_keys_vec...)
            group_key = $(key_func_name)(dim_keys)
            group_keys[i] = group_key
        end
        return group_keys
    end
   )
end

function gen_group_by_map_batch_function(key_func_name::Symbol,
                                         map_func_name::Symbol,
                                         map_batch_func_name::Symbol,
                                         num_dims::Integer,
                                         ValueType)
    map_batch_func = :(
       function $map_batch_func_name(dims::Vector{Int64},
                                     new_dims::Vector{Int64},
                                     keys::Vector{Int64},
                                     values::Vector,
                                     output_value_type)
            new_keys = Vector{Int64}()
            new_values = Vector{output_value_type}()
            dim_keys_vec = Vector{Int64}(length(dims))
            key_group_dict = Dict{UInt64, Vector{Tuple{NTuple{$num_dims, Int64}, $ValueType}}
                                  }()
            @assert length(keys) == length(values)
            for i in eachindex(keys)
                key = keys[i]
                value = values[i]
                OrionWorker.from_int64_to_keys(key, dims, dim_keys_vec)
                dim_keys = tuple(dim_keys_vec...)
                group_key = $(key_func_name)(dim_keys)
                if !haskey(key_group_dict, group_key)
                    key_group_dict[group_key] = Vector{Tuple{NTuple{$num_dims, Int64}, $ValueType}}()
                end
                push!(key_group_dict[group_key], (dim_keys, value))
            end
            for (group_key, group_values) in key_group_dict
                (new_dim_keys, new_value) = $(map_func_name)(group_values)
                new_key = OrionWorker.from_keys_to_int64(new_dim_keys, new_dims)
                push!(new_keys, new_key)
                push!(new_values, new_value)
            end
            return (new_keys, new_values)
      end
    )
    return map_batch_func
end

function gen_to_string_batch_function(to_string_func_name::Symbol,
                                      to_string_batch_func_name::Symbol)
    to_string_batch_func = :(
       function $to_string_batch_func_name(dims::Vector{Int64},
                                           keys::Vector{Int64},
                                           values::Vector)
            record_strings_vec = Vector{String}(length(keys))
            dim_keys_vec = Vector{Int64}(length(dims))
            for i in eachindex(keys)
                key = keys[i]
                value = values[i]
                OrionWorker.from_int64_to_keys(key, dims, dim_keys_vec)
                dim_keys = tuple(dim_keys_vec...)
                record_string = $(to_string_func_name)(dim_keys, value)
                record_strings_vec[i] = record_string
             end
             return record_strings_vec
       end
    )
    return to_string_batch_func
end

function gen_renamed_accumulator_vars(accumulator_vars::Vector{Symbol})
    update_global_accumulator_vars_stmts = quote end
    renamed_accumulator_vars_map = Dict{Symbol, Symbol}()
    if length(accumulator_vars) > 0
        for var_sym in accumulator_vars
            renamed_accumulator_var_sym = Symbol(:oriongen_xx_, var_sym)
            renamed_accumulator_vars_map[var_sym] = renamed_accumulator_var_sym
            update_global_accumulator_vars_stmt = :(global $(var_sym) = $(renamed_accumulator_var_sym))
            push!(update_global_accumulator_vars_stmts.args, update_global_accumulator_vars_stmt)
        end
    end
    return renamed_accumulator_vars_map, update_global_accumulator_vars_stmts
end
