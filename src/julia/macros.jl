macro share(ex::Expr)
    if ex.head == :function
        eval_expr_on_all(ex, :Main)
    else
        error("Do not support sharing Expr of this kind")
    end
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

function parallelize_for_loop(loop_stmt, is_ordered::Bool)
    @assert is_for_loop(loop_stmt)
    iteration_var = for_get_iteration_var(loop_stmt)
    iteration_space = for_get_iteration_space(loop_stmt)

    @assert isa(iteration_space, Symbol)
    println(iteration_space)
    @assert isdefined(current_module(), iteration_space)
    @assert isa(eval(current_module(), iteration_space), DistArray)

    par_for_context = ParForContext(iteration_var,
                                    iteration_space,
                                    loop_stmt,
                                    is_ordered)

    @time scope_context = get_scope_context!(nothing, loop_stmt)

    @time (flow_graph, flow_graph_context, ssa_context) = flow_analysis(loop_stmt)
    parallelized_loop = gen_parallelized_loop(loop_stmt, scope_context, par_for_context,
                                             flow_graph,
                                             flow_graph_context, ssa_context)
    println(parallelized_loop)
    return parallelized_loop
end

macro objective(expr::Expr)
end

macro evaluate(args...)
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
