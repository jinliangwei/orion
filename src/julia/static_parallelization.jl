# describe the information of a variable for code within a scope
# the only exception is is_accumulator, which is defined by the code
# within the code and the code in its ancestors' scope before it
@enum ParallelSchemeType ParallelSchemeType_naive =
    1 ParallelSchemeType_1d =
    2 ParallelSchemeType_2d =
    3 ParallelSchemeType_unimodular =
    4 ParallelSchemeType_none =
    5

# could return one of the following:
# 1) :(:)
# 2) a constant
# 3) a Tuple, the first dim is loop index dim, the second is offset

function eval_subscript(expr::Any,
                        par_for_scope::ScopeContext,
                        par_for_context::ParForContext)
    if expr == :(:)
        return expr
    elseif isa(expr, Number)
        return expr
    elseif isa(expr, Symbol)
        if isconst(current_module(), expr)
            expr_val = eval(expr)
            if isa(expr_val, Integer)
                return expr_val
            else
                return :(:)
            end
        else
            sym_def = get_symbol_def(par_for_scope, expr)
            if sym_def == nothing
                return nothing
            else
                sym_val = eval_subscript(sym_def, par_for_scope, par_for_context)
                return sym_val
            end
        end
    else
        @assert isa(expr, Expr)
        if expr.head == :call &&
            (expr.args[1] == :+ ||
             expr.args[1] == :-)
            left = eval_subscript(expr.args[2], par_for_scope, par_for_context)
            right = eval_subscript(expr.args[3], par_for_scope, par_for_context)
            operator = expr.args[1]
            if isa(left, Tuple) &&
                isa(right, Integer)
                if operator == :+
                    return (left[1], left[2] + right)
                else
                    return (left[1], left[2] - right)
                end
            elseif isa(left, Integer) &&
                isa(right, Tuple)
                if operator == :+
                    return (right[1], left + right[2])
                else
                    return (right[1], left - right[2])
                end
            elseif isa(left, Integer) &&
                isa(right, Integer)
                if operator == :+
                    return left + right
                else
                    return left - right
                end
            else
                return nothing
            end
        elseif expr.head == :ref
            if isa(expr.args[1], Expr) &&
            isa(expr.args[2], Integer) &&
            expr.args[1].head == :(.) &&
            length(expr.args[1].args) == 2 &&
            expr.args[1].args[1] == par_for_context.iteration_var &&
            isa(expr.args[1].args[2], Expr) &&
            expr.args[1].args[2].head == :quote &&
            expr.args[1].args[2].args[1] == :key
                return (expr.args[2], 0)
            else
                return nothing
            end
        end
    end
end

function eval_all_subscripts(par_for_context::ParForContext,
                                       par_for_scope::ScopeContext)::Bool
    for da_access_vec in values(par_for_context.dist_array_access_dict)
        for da_access in da_access_vec
            for subscript in da_access.subscripts
                expr = subscript.expr
                sub_value = eval_subscript(expr, par_for_scope, par_for_context)
                println("eval ", expr, " result = ", sub_value)
                if sub_value == nothing
                    return false
                elseif sub_value == :(:)
                    subscript.expr = :(:)
                    subscript.offset = 0
                    subscript.loop_index_dim = nothing
                elseif isa(sub_value, Number)
                    subscript.offset = sub_value
                    subscript.loop_index_dim = nothing
                elseif isa(sub_value, Tuple)
                    subscript.offset = sub_value[2]
                    subscript.loop_index_dim = sub_value[1]
                else
                    return false
                end
            end
        end
    end
    return true
end

@enum DepVecValue DepVecValue_any =
    1 DepVecValue_nonneg =
    2 DepVecValue_none =
    3

function compute_dependence_vectors(par_for_context::ParForContext)
    dep_vecs = Set{Tuple}()
    for da_access_pair in par_for_context.dist_array_access_dict
        dist_array_sym = da_access_pair.first
        dist_array = eval(current_module(), dist_array_sym)
        num_dims = dist_array.num_dims
        da_access_vec = da_access_pair.second
        for da_access_a in da_access_vec
            for da_access_b in da_access_vec
                if da_access_a.is_read && da_access_b.is_read
                    continue
                end
                subscripts_a = da_access_a.subscripts
                subscripts_b = da_access_b.subscripts
                dep_vec = Vector{Any}(num_dims)

                fill!(dep_vec, DepVecValue_any)
                if length(subscripts_a) != length(subscripts_b)
                    return nothing
                end
                no_dep = false
                for i = 1:length(subscripts_a)
                    sub_a = subscripts_a[i]
                    sub_b = subscripts_b[i]
                    if sub_a.expr == :(:)
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
                            return nothing
                        end
                    elseif sub_b.expr == :(:)
                        continue
                    elseif sub_b.loop_index_dim == nothing
                        return nothing
                    elseif sub_b.loop_index_dim != sub_a.loop_index_dim
                        return nothing
                    else
                        dep_vec[sub_a.loop_index_dim] = sub_a.offset - sub_b.offset
                    end
                end
                if no_dep
                    continue
                end
                for i = 1:length(dep_vec)
                    dep_val = dep_vec[i]
                    if isa(dep_val, Number)
                        if dep_val > 0
                            break
                        elseif dep_val < 0
                            no_dep = true
                            break
                        end
                    else
                        if dep_val == DepVecValue_any
                            dep_vec[i] = DepVecValue_nonneg
                        end
                    end
                end
                if !no_dep
                    push!(dep_vecs, tuple(dep_vec...))
                end
            end
        end
    end
    return dep_vecs
end

# Prerequisite: each DistArray access subscript is either a constant, :(:) or a linear
# function of one dimension index.

# The for loop can be n-D parallelized if there exists n dimensions in the iteration
# space such that for any DistArray, two iterations doesn't depend on each other
# if all n dimensions differ.

# Currently we support embarrassingly parallel, 1D and 2D parallelization.
# We perform the following checks:
# 1) if all DistArray accesses are reads, the loop is embarrassingly parallel;
# 2) otherwise, we first exculde the DistArrays that are read-only;
# 3) for each DistArray accessed, there exists 1 dimension such that for different dimension
# indices, those accesses are independent. That is, all accesses share the same subscripts,
# each access's subscripts involve exactly one dimension index.
# 4) if all DistArray accesses involve one dimension index, the loop is 1D parallelized, if 2,
# then 2D parallelized.

function simple_parallelization(par_for_context::ParForContext)
    iteration_space = par_for_context.iteration_space
    iteration_space_dist_array = eval(current_module(), iteration_space)

    all_read_dist_arrays = Set{Symbol}()

    for da_access_vec_pair in par_for_context.dist_array_access_dict
        all_read = true
        da_sym = da_access_vec_pair.first
        da_access_vec = da_access_vec_pair.second
        for da_access in da_access_vec
            if !da_access.is_read
                all_read = false
                break
            end
        end
        if all_read
            push!(all_read_dist_arrays, da_sym)
        end
    end
    if length(all_read_dist_arrays) == length(par_for_context.dist_array_access_dict)
        return ParallelSchemeType_naive
    end

    dimension_set = Set{Int64}()
    dist_array_access_dict = Dict{Symbol, Tuple{Vector{DistArrayAccessSubscript}, Int64}}()
    for da_access_vec_pair in par_for_context.dist_array_access_dict
        dist_array_dimension_set = Set{Int64}()
        da_sym = da_access_vec_pair.first
        da_access_vec = da_access_vec_pair.second
        prev_subscripts = nothing
        for da_access in da_access_vec
            subscripts = da_access.subscripts
            if prev_subscripts != nothing
                if length(subscripts) != length(prev_subscripts)
                    return ParallelScheme_none
                end
                for i = 1:length(subscripts)
                    sub_a = subscripts[i]
                    sub_b = prev_subscripts[i]
                    if sub_a.expr != sub_b.expr ||
                        sub_a.offset != sub_b.offset ||
                        sub_a.loop_index_dim != sub_b.loop_index_dim
                        return ParallelSchemeType_none
                    end
                end
            else
                one_dim_index = false
                for sub in subscripts
                    if sub.loop_index_dim != nothing
                        one_dim_index = true
                        break
                    end
                end
                if !one_dim_index
                    return ParallelSchemeType_none
                end
            end
            prev_subscripts = subscripts
            for sub in subscripts
                if sub.loop_index_dim != nothing
                    push!(dist_array_dimension_set, sub.loop_index_dim)
                end
            end
        end
        if length(dist_array_dimension_set) != 1
            return ParallelSchemeType_none
        end
        dist_array_access_dict[da_sym] = (prev_subscripts, collect(dist_array_dimension_set)[1])
        union!(dimension_set, dist_array_dimension_set)
    end
    if length(dimension_set) == 1
        return (ParallelSchemeType_1d, dist_array_access_dict, dimension_set)
    elseif length(dimension_set) == 2
        return (ParallelSchemeType_2d, dist_array_access_dict, dimension_set)
    else
        return -1
    end
end

function parallelize_1d()
end


function parallelize_2d(par_for_context::ParForContext,
                        par_for_scope::ScopeContext,
                        parallel_scheme)
    da_access_sizes = Dict{Symbol, Int64}()
    da_access_dict = parallel_scheme[2]
    dimension_set = parallel_scheme[3]
    comm_size_dict = Dict{Int64, Int64}()

    for da_access_pair in da_access_dict
        da_sym = da_access_pair.first
        da_access_subscripts = da_access_pair.second[1]
        da_access_partition_dim = da_access_pair.second[2]
        dist_array = eval(current_module(), da_sym)
        dims = dist_array.dims
        access_size = 1
        for i = 1:length(da_access_subscripts)
            sub = da_access_subscripts[i]
            if sub.expr == :(:)
                access_size *= dims[i]
            end
        end
        partition_dim_size = dims[da_access_partition_dim]
        if !(da_access_partition_dim in keys(comm_size_dict))
            comm_size_dict[da_access_partition_dim] = 0
        end
        comm_size_dict[da_access_partition_dim] += access_size
    end

    time_partition_dim = nothing
    comm_size = nothing
    for comm_size_pair in comm_size_dict
        dim = comm_size_pair.first
        size = comm_size_pair.second
        if comm_size == nothing ||
            comm_size > size
            comm_size = size
            time_partition_dim = dim
        end
    end

    space_partition_dim = nothing
    for dim in dimension_set
        if dim != time_partition_dim
            space_partition_dim = dim
        end
    end
    println("time partiton dim = ", time_partition_dim,
            " space_partition_dim = ", space_partition_dim)
    space_dim_tile_size = 100
    time_dim_tile_size = 100
    random_access_dist_array_set = Set{Symbol}()

    for da_access_pair in da_access_dict
        da_sym = da_access_pair.first
        da_access_partition_dim = da_access_pair.second[2]
        dist_array = eval(current_module(), da_sym)
        da_access_subscripts = da_access_pair.second[1]
        if da_access_subscripts[1].loop_index_dim != da_access_partition_dim ||
            da_access_subscripts[1].offset != 0
            push!(random_access_dist_array_set, da_sym)
        else
            if length(da_access_subscripts) > 1
                for sub in da_access_subscripts[2:end]
                    if sub.expr != :(:)
                        push!(random_access_dist_array_set, da_sym)
                        break
                    end
                end
            end
        end
    end
    dump(par_for_context.loop_stmt)
    loop_func_name = gen_unique_symbol()
    gen_loop_body_function(loop_func_name,
                           par_for_context.loop_stmt.args[2],
                           par_for_context,
                           par_for_scope)
    #println(par_for_context
end

function static_parallelize(par_for_context::ParForContext,
                            par_for_scope::ScopeContext)
    iteration_space = par_for_context.iteration_space
    iteration_space_dist_array = eval(current_module(), iteration_space)

    ret = eval_all_subscripts(par_for_context, par_for_scope)
    if !ret
        println("no static parallelization")
        return
    end
    parallel_scheme = simple_parallelization(par_for_context)
    println("parallel scheme = ", parallel_scheme)
    if parallel_scheme == ParallelSchemeType_naive
        println("embarassingly parallel")
    elseif parallel_scheme == ParallelSchemeType_none
        println("try to apply unimodular transformation")
        dep_vecs = compute_dependence_vectors(par_for_context)
        println(dep_vecs)
    else
        @assert isa(parallel_scheme, Tuple)
        if parallel_scheme[1] == ParallelSchemeType_1d
            parallelize_1d()
        elseif parallel_scheme[1] == ParallelSchemeType_2d
            parallelize_2d(par_for_context, par_for_scope, parallel_scheme)
        else
            @assert false
        end
    end

    iteration_var = par_for_context.iteration_var
    partition_func_name = gen_unique_symbol()
#    partition_func = gen_space_time_partition_function(
#        partition_func_name,
#        [(1,), (1,)], [(1,), (2,)],
#        100, 100)
    #eval_expr_on_all(partition_func, :OrionGen)
    #space_time_repartition(iteration_space_dist_array,
    #                       string(partition_func_name))
end
