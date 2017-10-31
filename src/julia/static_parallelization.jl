# describe the information of a variable for code within a scope
# the only exception is is_accumulator, which is defined by the code
# within the code and the code in its ancestors' scope before it
@enum DepVecValue DepVecValue_any =
    1 DepVecValue_nonneg =
    2 DepVecValue_none =
    3

const default_tile_size = 1000

function compute_dependence_vectors(par_for_context::ParForContext)
    dep_vecs = Set{Tuple}()
    for (dist_array_sym, da_access_vec) in par_for_context.dist_array_access_dict
        dist_array = eval(current_module(), dist_array_sym)
        num_dims = dist_array.num_dims

        for index_a in eachindex(da_access_vec)
            for index_b = index_a:length(da_access_vec)
                da_access_a = da_access_vec[index_a]
                da_access_b = da_access_vec[index_b]
                if da_access_a.is_read && da_access_b.is_read ||
                    (!(par_for_context.is_ordered) &&
                     !da_access_a.is_read && !da_access_b.is_read)
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

function check_for_1d_parallelization(dep_vecs::Set{Tuple}, num_dims)
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

function check_for_2d_parallelization(dep_vecs::Set{Tuple}, num_dims)
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

function check_for_unimodular_parallelization(dep_vecs::Set{Tuple}, num_dims)
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
    if !isempty(par_dims)
        return ForLoopParallelScheme_unimodular, par_dims
    end
    return ForLoopParallelScheme_none, par_dims
end

function parallelize_2d(par_for_context::ParForContext,
                        par_for_scope::ScopeContext,
                        par_dims,
                        ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}})
    num_work_units = 0
    par_dim = par_dims[1]
    space_partition_dim = par_dim[1]
    time_partition_dim = par_dim[2]

    iteration_space = par_for_context.iteration_space
    iteration_space_dist_array = eval(current_module(), iteration_space)
    num_dims = iteration_space_dist_array.num_dims
    iteration_space_dims = iteration_space_dist_array.dims
    da_access_dict = par_for_context.dist_array_access_dict
    space_partitioned_dist_array_ids = Vector{Int32}()
    time_partitioned_dist_array_ids = Vector{Int32}()
    global_indexed_dist_array_ids = Vector{Int32}()

    loop_partition_func_name = gen_unique_symbol()
    loop_partition_func = gen_2d_partition_function(loop_partition_func_name,
                                                    space_partition_dim,
                                                    time_partition_dim,
                                                    default_tile_size,
                                                    default_tile_size)

    space_partition_func_name = gen_unique_symbol()
    space_partition_func = gen_1d_partition_function(space_partition_func_name,
                                                     space_partition_dim,
                                                     default_tile_size)
    time_partition_func_name = gen_unique_symbol()
    time_partition_func = gen_1d_partition_function(time_partition_func_name,
                                                    time_partition_dim,
                                                    default_tile_size)

    dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_2d,
                                                       loop_partition_func_name,
                                                       (space_partition_dim, time_partition_dim),
                                                       DistArrayIndexType_none)
    repartition_stmt = :(Orion.check_and_repartition($(esc(iteration_space)), $dist_array_partition_info))
    parallelized_loop = quote end
    push!(parallelized_loop.args, repartition_stmt)
    for (da_sym, da_access_vec) in da_access_dict
        partition_dims = compute_dist_array_partition_dims(da_sym, da_access_vec, num_dims,
                                                           iteration_space_dims)
        if space_partition_dim in partition_dims
            dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_1d,
                                                               space_partition_func_name,
                                                               (space_partition_dim,),
                                                               DistArrayIndexType_local)
            push!(space_partitioned_dist_array_ids, eval(current_module(), da_sym).id)
            repartition_stmt = :(Orion.check_and_repartition($(esc(da_sym)), $dist_array_partition_info))
            push!(parallelized_loop.args, repartition_stmt)
        elseif time_partition_dim in partition_dims
            dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_1d,
                                                               time_partition_func_name,
                                                               (time_partition_dim,),
                                                               DistArrayIndexType_local)
            push!(time_partitioned_dist_array_ids, eval(current_module(), da_sym).id)
            repartition_stmt = :(Orion.check_and_repartition($(esc(da_sym)), $dist_array_partition_info))
            push!(parallelized_loop.args, repartition_stmt)
        else
            dist_array_partition_info = DistArrayPartitionInfo(DistArrayPartitionType_naive,
                                                               nothing, nothing,
                                                               DistArrayIndexType_global)
            repartition_stmt = :(Orion.check_and_repartition($(esc(da_sym)), $dist_array_partition_info))
            push!(global_indexed_dist_array_ids, eval(current_module(), da_sym).id)
            push!(parallelized_loop.args, repartition_stmt)
        end
    end

    loop_body_func_name = gen_unique_symbol()
    loop_batch_func_name = gen_unique_symbol()
    loop_body_func, loop_batch_func = gen_loop_body_function(loop_body_func_name,
                                                             loop_batch_func_name,
                                                             par_for_context.loop_stmt.args[2],
                                                             par_for_context,
                                                             par_for_scope,
                                                             ssa_defs)
    eval_expr_on_all(loop_partition_func, :Main)
    eval_expr_on_all(space_partition_func, :Main)
    eval_expr_on_all(time_partition_func, :Main)
    eval_expr_on_all(loop_body_func, :Main)
    eval_expr_on_all(loop_batch_func, :Main)

    loop_batch_func_name_str = string(loop_batch_func_name)

    exec_loop_stmt = :(Orion.exec_for_loop($(iteration_space_dist_array.id),
                                           Orion.ForLoopParallelScheme_2d,
                                           $(space_partitioned_dist_array_ids),
                                           $(time_partitioned_dist_array_ids),
                                           $(global_indexed_dist_array_ids),
                                           $loop_batch_func_name_str,
                                           $(par_for_context.is_ordered)))
    push!(parallelized_loop.args, exec_loop_stmt)

#    for (var, var_info) in par_for_scope.inherited_var
#        println(var, var_info)
#        if var in keys(accumulator_info_dict)
#            var_str = string(var)
#            get_accumulator_value_expr = :(esc(var) = Orion.get_accumulator_value(Symbol($(var_str))))
            #push!(parallelized_loop.args, get_accumulator_value_expr)
#            println("get accumulator value for ", var)
#        end
#    end
    return parallelized_loop
end

function compute_dist_array_partition_dims(da_sym::Symbol,
                                           da_access_vec::Vector{DistArrayAccess},
                                           num_dims,
                                           iteration_space_dims::Vector{Int64})
    partition_dims = Set{Int64}([i for i in 1:num_dims])
    dist_array = eval(current_module(), da_sym)
    sub_dims = Set{Tuple}()
    for i = 1:dist_array.num_dims
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

    partition_dims = Set{Int64}()
    da_dims = dist_array.dims
    for (sub_dim, loop_index_dim) in sub_dims
        if da_dims[sub_dim] == iteration_space_dims[loop_index_dim]
            push!(partition_dims, sub_dim)
        end
    end
    return partition_dims
end

function static_parallelize(par_for_context::ParForContext,
                            par_for_scope::ScopeContext,
                            ssa_context::SsaContext)
    iteration_space = par_for_context.iteration_space
    iteration_space_dist_array = eval(current_module(), iteration_space)
    num_dims = iteration_space_dist_array.num_dims
    dep_vecs = compute_dependence_vectors(par_for_context)
    par_scheme = determine_parallelization_scheme(dep_vecs, num_dims)
    if par_scheme[1] == ForLoopParallelScheme_2d
        return parallelize_2d(par_for_context,
                       par_for_scope,
                       par_scheme[2],
                       ssa_context.ssa_defs)
    else
    end
    return nothing
end
