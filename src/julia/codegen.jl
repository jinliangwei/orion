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
          for key in keys
            #println(key)
            dim_keys = OrionWorker.from_int64_to_keys(key, dims)
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
          for key in keys
            #println(key)
            dim_keys = OrionWorker.from_int64_to_keys(key, dims)
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
          for key in keys
            #println(key)
            dim_keys = OrionWorker.from_int64_to_keys(key, dims)
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
                  dim_keys = OrionWorker.from_int64_to_keys(key, parent_dims)
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
                                inherited_vars_to_mark_global::Set{Symbol},
                                ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}})

    @assert isa(loop_body, Expr)
    @assert loop_body.head == :block
    iteration_space_dist_array = eval(current_module(), iteration_space)
    iter_space_value_type = dist_array_get_value_type(iteration_space_dist_array)
    iteration_var_string = string(iteration_var)

    for var in inherited_vars_to_mark_global
        insert!(loop_body.args, 1, :(global $var))
    end

    loop_body_func = :(
        function $func_name($iteration_var)
        $loop_body
        end
    )

    println(loop_body_func)
    return loop_body_func
end

function gen_loop_body_batch_function(batch_func_name::Symbol,
                                      func_name::Symbol,
                                      iter_space_value_type::DataType)
    batch_loop_stmt = :(
    for i in 1:length(keys)
        key = keys[i]
        value = values[i]
        dim_keys = OrionWorker.from_int64_to_keys(key, dims)

        key_value = (dim_keys, value)
        $(func_name)(key_value)
        end
    )

    batch_func = :(
    function $batch_func_name(keys::Vector{Int64},
                              values::Vector{$iter_space_value_type},
                              dims::Vector{Int64})
        $(batch_loop_stmt)
    end
    )
    return batch_func
end

function gen_loop_body_batch_function_iter_dims(batch_func_name::Symbol,
                                                func_name::Symbol,
                                                iter_space_value_type::DataType,
                                                iterate_dims_length::Int64)

    batch_loop_stmt = quote
        if length(keys) == 0
          return
        end
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
            $(func_name)(key_value)
            dim_keys_vec = [dim_keys]
            value_vec = [value]
            prefix = curr_prefix
          end
        end
    end

    batch_func = :(
    function $batch_func_name(keys::Vector{Int64},
                              values::Vector{$iter_space_value_type},
                              dims::Vector{Int64})
        $(batch_loop_stmt)
    end
    )

    return batch_func
end

function gen_prefetch_function(prefetch_func_name::Symbol,
                               iterate_var::Symbol,
                               prefetch_stmts::Expr)
    prefetch_func = :(
        function $prefetch_func_name($iterate_var,
                                     oriongen_prefetch_point_dict::Dict{Int64, Set{Int64}},
                                     oriongen_prefetch_range_dict::Dict{Int64, Set{Tuple{Int64, UInt64}}})
            $prefetch_stmts
        end
    )
    return prefetch_func
end

function gen_prefetch_batch_function(prefetch_batch_func_name::Symbol,
                                     prefetch_func_name::Symbol,
                                     iter_space_value_type::DataType)
        prefetch_batch_func = :(
            function $prefetch_batch_func_name(keys::Vector{Int64},
                                               values::Vector{$iter_space_value_type},
                                               dims::Vector{Int64},
                                               global_indexed_dist_array_ids::Vector{Int32})::Tuple{Vector{Vector{Int64}},
                                                                                                    Vector{Vector{Tuple{Int64, UInt64}}}}
            prefetch_point_dict = Dict{Int32, Set{Int64}}()
            prefetch_range_dict = Dict{Int32, Set{Tuple{Int64, UInt64}}}()
            for id in global_indexed_dist_array_ids
                prefetch_point_dict[id] = Set{Int64}()
                prefetch_range_dict[id] = Set{Tuple{Int64, UInt64}}()
            end

            for i in 1:length(keys)
                key = keys[i]
                value = values[i]
                dim_keys = OrionWorker.from_int64_to_keys(key, dims)

                key_value = (dim_keys, value)
                $(prefetch_func_name)(key_value, prefetch_point_dict, prefetch_range_dict)
            end

            prefetch_point_array = Vector{Vector{Int64}}()
            prefetch_range_array = Vector{Vector{Tuple{Int64, UInt64}}}()
            for id in global_indexed_dist_array_ids
                point_set = prefetch_point_dict[id]
                push!(prefetch_point_array, collect(point_set))
                range_set = prefetch_range_dict[id]
                push!(prefetch_range_array, collect(range_set))
            end
            return (prefetch_point_array, prefetch_range_array)
        end
    )


    return prefetch_batch_func
end


function gen_prefetch_batch_function_iter_dims(prefetch_batch_func_name::Symbol,
                                               prefetch_func_name::Symbol,
                                               iter_space_value_type::DataType,
                                               iterate_dims_length::Int64)
        prefetch_batch_func = :(
            function $prefetch_batch_func_name(keys::Vector{Int64},
                                               values::Vector{$iter_space_value_type},
                                               dims::Vector{Int64},
                                               global_indexed_dist_array_ids::Vector{Int32})::Tuple{Vector{Vector{Int64}},
                                                                                                    Vector{Vector{Tuple{Int64, UInt64}}}}
            prefetch_point_dict = Dict{Int32, Set{Int64}}()
            prefetch_range_dict = Dict{Int32, Set{Tuple{Int64, UInt64}}}()
            for id in global_indexed_dist_array_ids
                prefetch_point_dict[id] = Set{Int64}()
                prefetch_range_dict[id] = Set{Tuple{Int64, UInt64}}()
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
                        $(prefetch_func_name)(key_value, prefetch_point_dict, prefetch_range_dict)
                        dim_keys_vec = [dim_keys]
                        value_vec = [value]
                        prefix = curr_prefix
                    end
                end
            end

            prefetch_point_array = Vector{Vector{Int64}}()
            prefetch_range_array = Vector{Vector{Tuple{Int64, UInt64}}}()
            for id in global_indexed_dist_array_ids
                point_set = prefetch_point_dict[id]
                push!(prefetch_point_array, collect(point_set))
                range_set = prefetch_range_dict[id]
                push!(prefetch_range_array, collect(range_set))
            end
            return (prefetch_point_array, prefetch_range_array)
        end
    )

    return prefetch_batch_func
end

function gen_access_count_function(access_count_func_name::Symbol,
                                   iterate_var::Symbol,
                                   access_stmts::Expr)
    access_count_func = :(
        function $access_count_func_name($iterate_var,
                                         oriongen_access_count_dict::Dict{Int64, Dict{Int64, UInt64}})
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
                                                   global_indexed_dist_array_ids::Vector{Int32})::Vector{Vector{Tuple{Int64, UInt64}}}
            access_count_dict = Dict{Int32, Dict{Int64, UInt64}}()
            for id in global_indexed_dist_array_ids
                access_count_dict[id] = Dict{Int64, UInt64}()
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
                access_count = access_count_dict[id]
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
                                               global_indexed_dist_array_ids::Vector{Int32})::Vector{Vector{Tuple{Int64, UInt64}}}

            access_count_dict = Dict{Int32, Dict{Int64, UInt64}}()
            for id in global_indexed_dist_array_ids
                access_count_dict[id] = Dict{Int64, UInt64}()
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
                        $(access_count_func_name)(key_value, access_count_point_dict, access_count_range_dict)
                        dim_keys_vec = [dim_keys]
                        value_vec = [value]
                        prefix = curr_prefix
                    end
                end
            end

            access_count_array = Vector{Vector{Tuple{Int64, UInt64}}}()
            for id in global_indexed_dist_array_ids
                access_count = access_count_dict[id]
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
