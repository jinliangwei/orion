# describe the information of a variable for code within a scope
# the only exception is is_accumulator, which is defined by the code
# within the code and the code in its ancestors' scope before it
@enum DepVecValue DepVecValue_any =
    1 DepVecValue_nonneg =
    2 DepVecValue_none =
    3

function get_1d_tile_size(iteration_space_partition_dim::Int64,
                          iteration_space_dist_array::DistArray)::Int64
    partition_dim_size = get(iteration_space_dist_array.iterate_dims)[iteration_space_partition_dim]
    tile_size = cld(partition_dim_size, iteration_space_dist_array.num_partitions_per_dim)
    return tile_size
end

function get_2d_tile_sizes(iteration_space_partition_dims::Tuple{Int64, Int64},
                           iteration_space_dist_array::DistArray)::Tuple{Int64, Int64}
    tile_size_1 = get_1d_tile_size(iteration_space_partition_dims[1], iteration_space_dist_array)
    tile_size_2 = get_1d_tile_size(iteration_space_partition_dims[2], iteration_space_dist_array)
    return (tile_size_1, tile_size_2)
end

function compute_dependence_vectors(dist_array_access_dict::Dict{Symbol, Vector{DistArrayAccess}},
                                    is_ordered::Bool)
    dep_vecs = Set{Tuple}()
    for (dist_array_sym, da_access_vec) in dist_array_access_dict
        dist_array = eval(current_module(), dist_array_sym)
        num_dims = dist_array_get_num_dims(dist_array)

        for index_a in eachindex(da_access_vec)
            for index_b = index_a:length(da_access_vec)
                da_access_a = da_access_vec[index_a]
                da_access_b = da_access_vec[index_b]
                if da_access_a.is_read && da_access_b.is_read ||
                    (!is_ordered && !da_access_a.is_read && !da_access_b.is_read)
                    continue
                end
                subscripts_a = da_access_a.subscripts
                subscripts_b = da_access_b.subscripts
                dep_vec = Vector{Any}(num_dims)
                @assert length(subscripts_a) == length(subscripts_b)

                fill!(dep_vec, DepVecValue_any)

                no_dep = false
                for i in eachindex(subscripts_a)
                    sub_a = subscripts_a[i]
                    sub_b = subscripts_b[i]
                    if sub_a.value_type == DistArrayAccessSubscript_value_any ||
                        sub_b.value_type == DistArrayAccessSubscript_value_any ||
                        sub_a.value_type == DistArrayAccessSubscript_value_unknown ||
                        sub_b.value_type == DistArrayAccessSubscript_value_unknown
                        continue
                    elseif sub_a.loop_index_dim == nothing
                        if sub_b.loop_index_dim == nothing
                            if sub_b.offset == sub_a.offset
                                continue
                            else
                                no_dep = true
                                break
                            end
                        else
                            continue
                        end
                    elseif sub_b.loop_index_dim == nothing
                        continue
                    elseif sub_b.loop_index_dim != sub_a.loop_index_dim
                        continue
                    else
                        dep_val = sub_a.offset - sub_b.offset
                        if dep_vec[sub_a.loop_index_dim] == dep_val ||
                            dep_vec[sub_a.loop_index_dim] == DepVecValue_any
                            dep_vec[sub_a.loop_index_dim] = dep_val
                        else
                            no_dep = true
                            break
                        end
                    end
                end
                if no_dep
                    continue
                end
                for i in eachindex(dep_vec)
                    dep_val = dep_vec[i]
                    if isa(dep_val, Number)
                        if dep_val > 0
                            break
                        elseif dep_val < 0
                            for j in eachindex(dep_vec)
                                if isa(dep_vec[j], Number)
                                    dep_vec[j] = -dep_vec[j]
                                end
                            end
                            break
                        end
                    else
                        if dep_val == DepVecValue_any
                            dep_vec[i] = DepVecValue_nonneg
                            break
                        end
                    end
                end
                push!(dep_vecs, tuple(dep_vec...))
            end
        end
    end
    return dep_vecs
end

function check_for_1d_parallelization(dep_vecs::Set{Tuple}, num_dims)::Vector{Int64}
    par_dims = Vector{Int64}()
    for i = 1:num_dims
        is_parallelizable = true
        for dep_vec in dep_vecs
            if dep_vec[i] != 0
                is_parallelizable = false
                break
            end
        end
        if is_parallelizable
            push!(par_dims, i)
        end
    end
    return par_dims
end

function check_for_2d_parallelization(dep_vecs::Set{Tuple}, num_dims)::Vector{Tuple{Int64, Int64}}
    par_dims = Vector{Tuple{Int64, Int64}}()
    for i = 1:num_dims
        for j = (i + 1):num_dims
            is_parallelizable = true
            for dep_vec in dep_vecs
                if dep_vec[i] != 0 &&
                    dep_vec[j] != 0
                    is_parallelizable = false
                    break
                end
            end
            if is_parallelizable
                push!(par_dims, (i, j))
            end
        end
    end
    return par_dims
end

function check_for_unimodular_parallelization(dep_vecs::Set{Tuple}, num_dims)::Vector{Int64}
    par_dims = Vector{Int64}()
    for i = 1:num_dims
        is_parallelizable = true
        for dep_vec in dep_vecs
            if dep_vec[i] == DepVecValue_any
                is_parallelizable = false
                break
            end
        end
        if is_parallelizable
            push!(par_dims, i)
        end
    end
    return par_dims
end

# Check whether a for loop is suitable for Naive, 1D or 2D parallelization
function determine_parallelization_scheme(dep_vecs::Set{Tuple}, num_dims)
    if isempty(dep_vecs)
        return ForLoopParallelScheme_naive, nothing
    end

    par_dims = check_for_1d_parallelization(dep_vecs, num_dims)
    if !isempty(par_dims)
        return ForLoopParallelScheme_1d, par_dims
    end

    par_dims = check_for_2d_parallelization(dep_vecs, num_dims)
    if !isempty(par_dims)
        return ForLoopParallelScheme_2d, par_dims
    end

    par_dims = check_for_unimodular_parallelization(dep_vecs, num_dims)
    if length(par_dims) >= 2
        return ForLoopParallelScheme_unimodular, par_dims
    end
    return ForLoopParallelScheme_none, par_dims
end

function parallelize_naive(iteration_space::Symbol,
                           iteration_var::Symbol,
                           dist_array_access_dict::Dict{Symbol, Vector{DistArrayAccess}},
                           inherited_vars_to_mark_global::Set{Symbol},
                           loop_body::Expr,
                           is_ordered::Bool,
                           ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}},
                           flow_graph::BasicBlock)
    iteration_space_dist_array = eval(current_module(), iteration_space)
    iteration_space_partition_info = isnull(iteration_space_dist_array.target_partition_info) ?
        iteration_space_dist_array.partition_info :
        get(iteration_space_dist_array.target_partition_info)
    println(iteration_space_partition_info.partition_type)
    if iteration_space_partition_info.partition_type == DistArrayPartitionType_naive ||
        iteration_space_partition_info.partition_type == DistArrayPartitionType_range ||
        iteration_space_partition_info.partition_type == DistArrayPartitionType_hash_executor ||
        iteration_space_partition_info.partition_type == DistArrayPartitionType_hash_server
        partition_dim = length(get(iteration_space_dist_array.iterate_dims))
        return parallelize_1d(iteration_space,
                              iteration_var,
                              dist_array_access_dict,
                              inherited_vars_to_mark_global,
                              loop_body,
                              is_ordered,
                              [partition_dim],
                              ssa_defs,
                              flow_graph)
    elseif iteration_space_partition_info.partition_type == DistArrayPartitionType_1d
        return parallelize_1d(iteration_space,
                              iteration_var,
                              dist_array_access_dict,
                              inherited_vars_to_mark_global,
                              loop_body,
                              is_ordered,
                              [get(iteration_space_partition_info.partition_dims)[1]],
                              ssa_defs,
                              flow_graph)
    elseif iteration_space_partition_info.partition_type == DistArrayPartitionType_2d
        return parallelize_2d(iteration_space,
                              iteration_var,
                              dist_array_access_dict,
                              inherited_vars_to_mark_global,
                              loop_body,
                              is_ordered,
                              [get(iteration_space_partition_info.partition_dims)],
                              ssa_defs,
                              flow_graph)
    else
        return parallelize_unimodular(iteration_space,
                                      iteration_var,
                                      dist_array_access_dict,
                                      inherited_vars_to_mark_global,
                                      loop_body,
                                      is_ordered,
                                      [get(iteration_space_partition_info.partition_dims)...],
                                      ssa_defs,
                                      flow_graph)
    end

end

function parallelize_1d(iteration_space::Symbol,
                        iteration_var::Symbol,
                        dist_array_access_dict::Dict{Symbol, Vector{DistArrayAccess}},
                        inherited_vars_to_mark_global::Set{Symbol},
                        loop_body::Expr,
                        is_ordered::Bool,
                        par_dims::Vector{Int64},
                        ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}},
                        flow_graph::BasicBlock)
    space_partition_dim = par_dims[end]
    iteration_space_dist_array = eval(current_module(), iteration_space)
    iteration_space_dims = get(iteration_space_dist_array.iterate_dims)
    tile_size = get_1d_tile_size(space_partition_dim, iteration_space_dist_array)
    iteration_space_dims_diff = length(iteration_space_dist_array.dims) - length(get(iteration_space_dist_array.iterate_dims))
    space_partitioned_dist_array_ids = Vector{Int32}()
    global_indexed_dist_array_ids = Vector{Int32}()
    global_indexed_dist_array_syms = Set{Symbol}()

    loop_partition_func_name = gen_unique_symbol()
    loop_partition_func = gen_1d_partition_function(loop_partition_func_name,
                                                    space_partition_dim + iteration_space_dims_diff,
                                                    tile_size)

    dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_1d,
                                                       string(loop_partition_func_name),
                                                       (space_partition_dim + iteration_space_dims_diff,),
                                                       (tile_size,),
                                                       DistArrayIndexType_none)
    parallelized_loop = quote end
    repartition_stmt = :(Orion.check_and_repartition($(esc(iteration_space)), $dist_array_partition_info))
    iteration_space_dist_array.target_partition_info = Nullable{DistArrayPartitionInfo}(dist_array_partition_info)
    push!(parallelized_loop.args, repartition_stmt)
    partition_func_set = Set{Expr}()
    accessed_da_syms = Set{Symbol}()
    for (da_sym, da_access_vec) in dist_array_access_dict
        push!(accessed_da_syms, da_sym)
        partition_dims = compute_dist_array_partition_dims(da_sym, da_access_vec,
                                                           iteration_space_dims)
        partition_dim = 0
        if !(da_sym in keys(dist_array_buffer_map)) && space_partition_dim in keys(partition_dims)
            partition_dim = partition_dims[space_partition_dim]
            push!(space_partitioned_dist_array_ids, eval(current_module(), da_sym).id)
        else
            push!(global_indexed_dist_array_ids, eval(current_module(), da_sym).id)
            push!(global_indexed_dist_array_syms, da_sym)
        end

        if partition_dim > 0
            partition_func_name = gen_unique_symbol()
            partition_func = gen_1d_partition_function(partition_func_name,
                                                       partition_dim,
                                                       tile_size)
            push!(partition_func_set, partition_func)
            dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_1d,
                                                               string(partition_func_name),
                                                               (partition_dim,),
                                                               (tile_size,),
                                                               DistArrayIndexType_none)
            repartition_stmt = :(Orion.check_and_repartition($(esc(da_sym)), $dist_array_partition_info))
            push!(parallelized_loop.args, repartition_stmt)
        else
            dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_hash_server,
                                                               DistArrayIndexType_range)
            repartition_stmt = :(Orion.check_and_repartition($(esc(da_sym)), $dist_array_partition_info))
            push!(parallelized_loop.args, repartition_stmt)
        end

        if da_sym in keys(dist_array_buffer_map)
            buffer_info = dist_array_buffer_map[da_sym]
            for helper_da_sym in buffer_info.helper_dist_array_vec
                @assert !(helper_da_sym in accessed_da_syms)
                dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_hash_server,
                                                                   DistArrayIndexType_range)
                repartition_stmt = :(Orion.check_and_repartition($(esc(helper_da_sym)), $dist_array_partition_info))
                push!(parallelized_loop.args, repartition_stmt)
            end
        end
    end

    loop_body_func_name = gen_unique_symbol()
    loop_batch_func_name = gen_unique_symbol()
    loop_body_func = gen_loop_body_function(loop_body_func_name,
                                            loop_body,
                                            iteration_space,
                                            iteration_var,
                                            inherited_vars_to_mark_global,
                                            ssa_defs)
    iter_space_value_type = dist_array_get_value_type(iteration_space_dist_array)
    if iteration_space_dist_array.dims == get(iteration_space_dist_array.iterate_dims)
        loop_batch_func = gen_loop_body_batch_function(loop_batch_func_name,
                                                       loop_body_func_name,
                                                       iter_space_value_type)
    else
        loop_batch_func = gen_loop_body_batch_function_iter_dims(loop_batch_func_name,
                                                                 loop_body_func_name,
                                                                 iter_space_value_type,
                                                                 length(get(iteration_space_dist_array.iterate_dims)))
    end

    prefetch_func = nothing
    prefetch_batch_func = nothing
    prefetch = false
    if !isempty(global_indexed_dist_array_syms)
        prefetch_stmts = get_prefetch_stmts(flow_graph, global_indexed_dist_array_syms)
        if prefetch_stmts != nothing
            prefetch = true
            prefetch_func_name = gen_unique_symbol()
            prefetch_batch_func_name = gen_unique_symbol()
            prefetch_func = gen_prefetch_function(prefetch_func_name,
                                                  iteration_var,
                                                  prefetch_stmts)
            if iteration_space_dist_array.dims == get(iteration_space_dist_array.iterate_dims)
                prefetch_batch_func = gen_prefetch_batch_function(prefetch_batch_func_name,
                                                                  prefetch_func_name,
                                                                  iter_space_value_type)
            else
                prefetch_batch_func = gen_prefetch_batch_function_iter_dims(prefetch_batch_func_name,
                                                                            prefetch_func_name,
                                                                            iter_space_value_type,
                                                                            length(get(iteration_space_dist_array.iterate_dims)))
            end
        end
    end
    remap_ssa_vars(loop_body_func.args[2], ssa_defs)
    if prefetch_func != nothing
        remap_ssa_vars(prefetch_func.args[2], ssa_defs)
    end
    eval_expr_on_all(loop_partition_func, :Main)
    for partition_func in partition_func_set
        eval_expr_on_all(partition_func, :Main)
    end
    eval_expr_on_all(loop_body_func, :Main)
    eval_expr_on_all(loop_batch_func, :Main)

    loop_batch_func_name_str = string(loop_batch_func_name)
    prefetch_batch_func_name_str = prefetch ? string(prefetch_batch_func_name) : ""
    buffered_dist_array_ids = Vector{Int32}()
    dist_array_buffer_ids = Vector{Int32}()
    num_buffers_each_dist_array = Vector{Int32}()

    for dist_array_id in global_indexed_dist_array_ids
        if dist_array_id in keys(dist_array_buffer_map)
            push!(buffered_dist_array_ids, dist_array_id)
            buff_info = dist_array_buffer_map[dist_array_id]
            push!(num_buffers_each_dist_array, length(buff_info.helper_buffer_vec))
            append!(dist_array_buffer_ids, buff_info.helper_buffer_vec)
        end
    end

    exec_loop_stmt = :(Orion.exec_for_loop($(iteration_space_dist_array.id),
                                           Orion.ForLoopParallelScheme_1d,
                                           $(space_partitioned_dist_array_ids),
                                           Vector{Int64}(),
                                           $(global_indexed_dist_array_ids),
                                           $(buffered_dist_array_ids),
                                           $(dist_array_buffer_ids),
                                           $(num_buffers_each_dist_array),
                                           $loop_batch_func_name_str,
                                           $prefetch_batch_func_name_str,
                                           $(is_ordered)))
    push!(parallelized_loop.args, exec_loop_stmt)
    return parallelized_loop
end

function parallelize_2d(iteration_space::Symbol,
                        iteration_var::Symbol,
                        dist_array_access_dict::Dict{Symbol, Vector{DistArrayAccess}},
                        inherited_vars_to_mark_global::Set{Symbol},
                        loop_body::Expr,
                        is_ordered::Bool,
                        par_dims::Vector{Tuple{Int64, Int64}},
                        ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}},
                        flow_graph::BasicBlock)
    println("parallelize_2d")
    par_dim = par_dims[end]
    space_partition_dim = par_dim[1]
    time_partition_dim = par_dim[2]

    iteration_space_dist_array = eval(current_module(), iteration_space)
    iteration_space_dims = get(iteration_space_dist_array.iterate_dims)
    space_partitioned_dist_array_ids = Vector{Int32}()
    time_partitioned_dist_array_ids = Vector{Int32}()
    global_indexed_dist_array_ids = Vector{Int32}()
    global_indexed_dist_array_syms = Set{Symbol}()
    tile_sizes = get_2d_tile_sizes(par_dim, iteration_space_dist_array)

    iteration_space_dims_diff = length(iteration_space_dist_array.dims) - length(get(iteration_space_dist_array.iterate_dims))

    loop_partition_func_name = gen_unique_symbol()
    loop_partition_func = gen_2d_partition_function(loop_partition_func_name,
                                                    space_partition_dim + iteration_space_dims_diff,
                                                    time_partition_dim + iteration_space_dims_diff,
                                                    tile_sizes[1],
                                                    tile_sizes[2])
    dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_2d,
                                                       string(loop_partition_func_name),
                                                       (space_partition_dim + iteration_space_dims_diff,
                                                        time_partition_dim + iteration_space_dims_diff),
                                                       tile_sizes,
                                                       DistArrayIndexType_none)
    parallelized_loop = quote end
    repartition_stmt = :(Orion.check_and_repartition($(esc(iteration_space)), $dist_array_partition_info))
    iteration_space_dist_array.target_partition_info = Nullable{DistArrayPartitionInfo}(dist_array_partition_info)
    push!(parallelized_loop.args, repartition_stmt)
    partition_func_set = Set{Expr}()
    accessed_da_syms = Set{Symbol}()
    for (da_sym, da_access_vec) in dist_array_access_dict
        push!(accessed_da_syms, da_sym)
        partition_dims = compute_dist_array_partition_dims(da_sym, da_access_vec,
                                                           iteration_space_dims)
        partition_dim = 0
        if !(da_sym in keys(dist_array_buffer_map)) &&
            (space_partition_dim in keys(partition_dims) ||
             time_partition_dim in keys(partition_dims))
            if space_partition_dim in keys(partition_dims)
                partition_dim = partition_dims[space_partition_dim]
                push!(space_partitioned_dist_array_ids, eval(current_module(), da_sym).id)
                tile_size = tile_sizes[1]
            else
                @assert time_partition_dim in keys(partition_dims)
                partition_dim = partition_dims[time_partition_dim]
                push!(time_partitioned_dist_array_ids, eval(current_module(), da_sym).id)
                tile_size = tile_sizes[2]
            end
        else
            push!(global_indexed_dist_array_ids, eval(current_module(), da_sym).id)
            push!(global_indexed_dist_array_syms, da_sym)
        end

        if partition_dim > 0
            partition_func_name = gen_unique_symbol()
            partition_func = gen_1d_partition_function(partition_func_name,
                                                       partition_dim,
                                                       tile_size)
            push!(partition_func_set, partition_func)
            dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_1d,
                                                               string(partition_func_name),
                                                               (partition_dim,),
                                                               (tile_size,),
                                                               DistArrayIndexType_none)
            repartition_stmt = :(Orion.check_and_repartition($(esc(da_sym)), $dist_array_partition_info))
            push!(parallelized_loop.args, repartition_stmt)
        else
            dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_server_hash,
                                                               DistArrayIndexType_range)
            repartition_stmt = :(Orion.check_and_repartition($(esc(da_sym)), $dist_array_partition_info))
            push!(parallelized_loop.args, repartition_stmt)
        end

        if da_sym in keys(dist_array_buffer_map)
            @assert !(da_sym in accessed_da_syms)
            buffer_info = dist_array_buffer_map[da_sym]
            for helper_da_sym in buffer_info.helper_dist_array_vec
                dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_server_hash,
                                                                   DistArrayIndexType_range)
                repartition_stmt = :(Orion.check_and_repartition($(esc(helper_da_sym)), $dist_array_partition_info))
                push!(parallelized_loop.args, repartition_stmt)
            end
        end
    end

    loop_body_func_name = gen_unique_symbol()
    loop_batch_func_name = gen_unique_symbol()
    loop_body_func = gen_loop_body_function(loop_body_func_name,
                                            loop_body,
                                            iteration_space,
                                            iteration_var,
                                            inherited_vars_to_mark_global,
                                            ssa_defs)

    iter_space_value_type = dist_array_get_value_type(iteration_space_dist_array)
    if iteration_space_dist_array.dims == get(iteration_space_dist_array.iterate_dims)
        loop_batch_func = gen_loop_body_batch_function(loop_batch_func_name,
                                                       loop_body_func_name,
                                                       iter_space_value_type)
    else
        loop_batch_func = gen_loop_body_batch_function_iter_dims(loop_batch_func_name,
                                                                 loop_body_func_name,
                                                                 iter_space_value_type,
                                                                 length(get(iteration_space_dist_array.iterate_dims)))
    end

    prefetch_func = nothing
    prefetch = false
    if !isempty(global_indexed_dist_array_syms)
        prefetch_stmts = get_prefetch_stmts(flow_graph, global_indexed_dist_array_syms)
        if prefetch_stmts != nothing
            prefetch = true
            prefetch_func_name = gen_unique_symbol()
            prefetch_batch_func_name = gen_unique_symbol()
            prefetch_func = gen_prefetch_function(prefetch_func_name,
                                                  iteration_var,
                                                  prefetch_stmts)
            if iteration_space_dist_array.dims == get(iteration_space_dist_array.iterate_dims)
                prefetch_batch_func = gen_prefetch_batch_function(prefetch_batch_func_name,
                                                                  prefetch_func_name,
                                                                  iter_space_value_type)
            else
                prefetch_batch_func = gen_prefetch_batch_function_iter_dims(prefetch_batch_func_name,
                                                                            prefetch_func_name,
                                                                            iter_space_value_type,
                                                                            length(get(iteration_space_dist_array.iterate_dims)))
            end
        end
    end

    eval_expr_on_all(loop_partition_func, :Main)
    for partition_func in partition_func_set
        eval_expr_on_all(partition_func, :Main)
    end
    eval_expr_on_all(loop_body_func, :Main)
    eval_expr_on_all(loop_batch_func, :Main)

    loop_batch_func_name_str = string(loop_batch_func_name)
    prefetch_batch_func_name_str = prefetch ? string(prefetch_batch_func_name) : ""
    buffered_dist_array_ids = Vector{Int32}()
    dist_array_buffer_ids = Vector{Int32}()
    num_buffers_each_dist_array = Vector{UInt64}()

    for dist_array_id in global_indexed_dist_array_ids
        if dist_array_id in keys(dist_array_buffer_map)
            push!(buffered_dist_array_ids, dist_array_id)
            buff_info = dist_array_buffer_map[dist_array_id]
            push!(num_buffers_each_dist_array, length(buff_info.helper_buffer_vec))
            append!(dist_array_buffer_ids, buff_info.helper_buffer_vec)
        end
    end

    exec_loop_stmt = :(Orion.exec_for_loop($(iteration_space_dist_array.id),
                                           Orion.ForLoopParallelScheme_2d,
                                           $(space_partitioned_dist_array_ids),
                                           $(time_partitioned_dist_array_ids),
                                           $(global_indexed_dist_array_ids),
                                           $(buffered_dist_array_ids),
                                           $(dist_array_buffer_ids),
                                           $(num_buffers_each_dist_array),
                                           $loop_batch_func_name_str,
                                           $prefetch_batch_func_name_str,
                                           $(is_ordered)))

    push!(parallelized_loop.args, exec_loop_stmt)
    println(iteration_space_dist_array.partition_info.partition_type)
    return parallelized_loop
end

function parallelize_unimodular(iteration_space::Symbol,
                                iteration_var::Symbol,
                                dist_array_access_dict::Dict{Symbol, Vector{DistArrayAccess}},
                                inherited_vars_to_mark_global::Set{Symbol},
                                loop_body::Expr,
                                is_ordered::Bool,
                                par_dims::Vector{Int64},
                                ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}},
                                flow_graph::BasicBlock)
    error("this is currently not supported")
end

# For a dist_array, if a subscript of all of its accesses are indexed by the same
# loop induction variable with the same offset, then assuming the iteration space
# is partitioned along that induction variable dimension (either time or space wise),
# partitioning the dist_array along that subscript's dimension guarantees that
# different workers do not access the same dist array element

# this function returns all such subscript dimensions and corresponding induction
# variable dimensions

function compute_dist_array_partition_dims(da_sym::Symbol,
                                           da_access_vec::Vector{DistArrayAccess},
                                           iteration_space_dims::Vector{Int64})::Dict{Int64, Int64}
    dist_array = eval(current_module(), da_sym)
    sub_dims = Set{Tuple}()
    for i = 1:dist_array_get_num_dims(dist_array)
        loop_index_dim = nothing
        for da_access in da_access_vec
            subscript = da_access.subscripts[i]
            if subscript.value_type == DistArrayAccessSubscript_value_static &&
                subscript.loop_index_dim != nothing &&
                subscript.offset == 0
                if loop_index_dim == nothing
                    loop_index_dim = subscript.loop_index_dim
                elseif loop_index_dim != subscript.loop_index_dim
                    loop_index_dim = nothing
                    break
                end
            else
                loop_index_dim = nothing
                break
            end
        end
        if loop_index_dim != nothing
            push!(sub_dims, (i, loop_index_dim))
        end
    end

    partition_dims = Dict{Int64, Int64}()
    da_dims = dist_array.dims
    for (sub_dim, loop_index_dim) in sub_dims
        if da_dims[sub_dim] == iteration_space_dims[loop_index_dim]
            partition_dims[loop_index_dim] = sub_dim
        end
    end
    return partition_dims
end

function static_parallelize(iteration_space::Symbol,
                            iteration_var::Symbol,
                            inherited_vars_to_mark_global::Set{Symbol},
                            loop_body::Expr,
                            is_ordered::Bool,
                            ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}},
                            flow_graph::BasicBlock)
    iteration_space = iteration_space
    iteration_space_dist_array = eval(current_module(), iteration_space)
    num_dims = length(get(iteration_space_dist_array.iterate_dims))
    dist_array_access_dict = get_dist_array_access(flow_graph, iteration_var, ssa_defs)
    dep_vecs = compute_dependence_vectors(dist_array_access_dict, is_ordered)
    par_scheme = determine_parallelization_scheme(dep_vecs, num_dims)
    if par_scheme[1] == ForLoopParallelScheme_naive
        println("parallel naive")
        return parallelize_naive(iteration_space,
                                 iteration_var,
                                 dist_array_access_dict,
                                 inherited_vars_to_mark_global,
                                 loop_body,
                                 is_ordered,
                                 ssa_defs,
                                 flow_graph)
    elseif par_scheme[1] == ForLoopParallelScheme_1d
        println("parallel 1d")
        return parallelize_1d(iteration_space,
                              iteration_var,
                              dist_array_access_dict,
                              inherited_vars_to_mark_global,
                              loop_body,
                              is_ordered,
                              par_scheme[2],
                              ssa_defs,
                              flow_graph)
    elseif par_scheme[1] == ForLoopParallelScheme_2d
        println("parallel 2d")
        return parallelize_2d(iteration_space,
                              iteration_var,
                              dist_array_access_dict,
                              inherited_vars_to_mark_global,
                              loop_body,
                              is_ordered,
                              par_scheme[2],
                              ssa_defs,
                              flow_graph)
    elseif par_scheme[1] == ForLoopParallelScheme_unimodular
        println("parallel unimodular")
        return parallelize_unimodular(iteration_space,
                                      iteration_var,
                                      dist_array_access_dict,
                                      inherited_vars_to_mark,
                                      loop_body,
                                      is_ordered,
                                      par_scheme[2],
                                      ssa_defs,
                                      flow_graph)
    end
    return nothing
end
