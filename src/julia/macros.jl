macro share(ex::Expr)
    if ex.head == :function
        eval_expr_on_all(ex, :Main)
    else
        error("Do not support sharing Expr of this kind")
    end
    esc(ex)
end

macro transform(expr::Expr)
    if expr.head == :for
        context = ScopeContext()
        transform_loop(expr, context)
    elseif expr.head == :block
        transform_block(expr)
    else
        error("Expression ", expr.head, " cannot be parallelized (yet)")
    end
end

macro parallel_for(expr::Expr)
end

macro ordered_parallel_for(expr::Expr)
end

macro accumulator(expr::Expr)
end

function transform_loop(expr::Expr, context::ScopeContext)
    @time scope_context = get_scope_context!(nothing, expr)
#    print(scope_context)
    @time flow_analysis(expr)
    return

    iterative_body = quote
    end
    push!(iterative_body.args, Expr(:call, :println, "ran one iteration"))
    ret = quote
    end

    static_bc_var, dynamic_bc_var_array,
    accumulator_var_array = get_vars_to_broadcast(scope_context)
    println("broadcat list ", static_bc_var)
    define_static_bc_vars_stmt = :(Orion.define_vars($static_bc_var))
    #push!(ret.args, define_static_bc_vars_stmt)

    loop_transformed = gen_transformed_loop(expr, scope_context)
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

function transform_block(expr::Expr)
    return :(assert(false))
end

macro objective(expr::Expr)
end

macro evaluate(args...)
end
