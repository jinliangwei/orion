macro share(ex::Expr)
    eval_expr_on_all(ex, :Main)
    esc(ex)
end

macro parallel_for(expr::Expr)
    return parallelize_for_loop(expr, false)
end

macro ordered_parallel_for(expr::Expr)
    return parallelize_for_loop(expr, true)
end

macro accumulator(expr::Expr)
    @assert is_variable_definition(expr)
    var = assignment_get_assigned_to(expr)
    @assert isa(var, Symbol)
    accumulator_info_dict[var] = AccumulatorInfo(var,
                                                 eval(assignment_get_assigned_from(expr)))
    ret = quote end
    push!(ret.args, esc(expr))
    var_str = string(var)
    define_var_expr = :(Orion.define_var(Symbol($var_str)))
    push!(ret.args, define_var_expr)
    return ret
end

function parallelize_for_loop(loop_stmt::Expr, is_ordered::Bool)
    println("parallelize_for loop")
    @assert is_for_loop(loop_stmt)
    iteration_var = for_get_iteration_var(loop_stmt)
    iteration_space = for_get_iteration_space(loop_stmt)

    @assert isa(iteration_space, Symbol)
    @assert isdefined(current_module(), iteration_space)
    @assert isa(eval(current_module(), iteration_space), DistArray)

    # find variables that need to be broadcast and marked global
    @time scope_context = get_scope_context!(nothing, loop_stmt)
    bc_vars = get_vars_to_broadcast(scope_context)

    loop_body = for_get_loop_body(loop_stmt)
    @time (flow_graph, _, ssa_context) = flow_analysis(loop_body)

    inherited_var = scope_context.inherited_var
    inherited_vars_to_mark_global = Set{Symbol}()
    global_vars_written = Set{Symbol}()
    for (var, var_info) in inherited_var
        println("inherited_var: ", var, " ", var_info)
        @assert !var_info.is_marked_local
        if var_info.is_assigned_to &&
            !var_info.is_marked_global
            push!(inherited_vars_to_mark_global, var)
        end
        if var_info.is_assigned_to &&
            var_info.is_mutated
            push!(global_vars_written, var)
        end
    end

    parallelized_loop = quote end
    define_dynamic_bc_vars_stmt = :(Orion.define_vars($bc_vars))
    push!(parallelized_loop.args, define_dynamic_bc_vars_stmt)
    println("before static_parallelize")
    exec_loop_stmts = static_parallelize(iteration_space,
                                         iteration_var,
                                         inherited_vars_to_mark_global,
                                         global_vars_written,
                                         loop_body,
                                         is_ordered,
                                         ssa_context.ssa_defs,
                                         flow_graph)

    if exec_loop_stmts == nothing
        error("loop not parallelizable")
    end
    push!(parallelized_loop.args, exec_loop_stmts)
    return parallelized_loop
end

macro dist_array(expr::Expr)
    ret_stmts = quote
        $(esc(expr))
    end

    @assert expr.head == :(=)
    dist_array_symbol = assignment_get_assigned_to(expr)
    @assert isa(dist_array_symbol, Symbol)
    symbol_str = string(dist_array_symbol)
    push!(ret_stmts.args,
          :(Orion.dist_array_set_symbol($(esc(dist_array_symbol)), $symbol_str)))
    return ret_stmts
end
