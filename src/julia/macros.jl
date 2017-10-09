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

macro accumulator(expr::Expr, combiner::Symbol)
    @assert is_variable_definition(expr)
    var = assignment_get_assigned_to(expr)
    @assert isa(var, Symbol)
    accumulator_info_dict[var] = AccumulatorInfo(var,
                                                 eval(assignment_get_assignment_from(expr)),
                                                 combiner)
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

function transform_loop(expr::Expr, context::ScopeContext)

#    print(scope_context)


    #push!(ret.args, define_static_bc_vars_stmt)


    return
    push!(ret.args, loop_transformed)


    #println(eval(current_module(), :num_iterations))
    #ret = :(for i = 1:$(esc(:num_iterations)) println(i) end)
    # TODO: generate and insert stmts for dynamic broadcast variables

    # parallelization
   # par_for_index = 1
   # for curr_par_for_context in scope_context.par_for_context
    #    parallelized_for_loop = static_parallelize(curr_par_for_context,
     #                                              scope_context.par_for_scope[par_for_index])
      #  print(scope_context)
       # par_for_index += 1
    #end

    #bc_expr_array = Array{Array{Expr, 1}, 1}()
#    for dynamic_bc_var in dynamic_bc_var_array
#        bc_expr_array = gen_stmt_broadcast_var(dynamic_bc_var)
        #push!(bc_expr_array, expr_array)
#        for bc_expr in bc_expr_array
#            push!(iterative_body.args, bc_expr)
#        end
#    end

 #   push!(ret.args,
 #         Expr(expr.head,
 #              esc(expr.args[1]),
 #              iterative_body
 #              )
    #         )

    #dump(ret)
    return ret
end

macro objective(expr::Expr)
end

macro evaluate(args...)
end
