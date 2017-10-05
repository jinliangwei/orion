type TransformContext
    scope_context::ScopeContext
    curr_par_for_index::Int64
    curr_child_scope_index::Int64
    TransformContext(scope_context::ScopeContext) = new(scope_context,
                                                        1, 1)
end

function gen_transformed_loop(expr::Expr, scope_context::ScopeContext)
    @assert expr.head == :for

    parallelized = false
    for par_for_index in eachindex(scope_context.par_for_context)
        par_for_context = scope_context.par_for_context[par_for_index]
        par_for_scope = scope_context.par_for_scope[par_for_index]
       # this_parallelized, par_loop = static_parallelize(par_for_context, par_for_scope)
        #parallelized |= this_parallelized
        #par_for_context.parallelized_loop = par_loop
    end

#    if !parallelized
#        return remove_orion_macros(expr)
#    end

    iteration_var = for_get_iteration_var(expr)
    loop_stmt = :(for i = 1:1 println("iteration = ", $(esc(iteration_var)),
                                      " step size = ", $(esc(:step_size))) end)

    transform_context = TransformContext(scope_context)
    loop_condition = for_get_loop_condition(expr)
    loop_stmt.args[1] = AstWalk.ast_walk(loop_condition, gen_transformed_visit, transform_context)

    stmt_array = for_get_loop_body(loop_stmt).args
    for stmt in for_get_loop_body(expr).args
        transformed_stmt = AstWalk.ast_walk(stmt, gen_transformed_visit,
                                           transform_context)
        push!(stmt_array, transformed_stmt)
    end
    return loop_stmt
end

function remove_orion_macros(expr::Expr)
    return expr
end

function transform_child_scope(expr::Expr, transform_context::TransformContext)
    child_scope_index = transform_context.curr_child_scope_index
    scope_context = transform_context.scope_context
    child_scope_context = scope_context.child_scope[child_scope_index]
    local transformed_expr
    if expr.head == :for
        iteration_var = for_get_iteration_var(expr)
        scope_stmt = :(for i = 1:1 end)
        scope_stmt.args[1] = for_get_loop_condition(expr)
        stmt_array = for_get_loop_body(scope_stmt).args

        child_transform_context = TransformContext(child_scope_context)

        for stmt in for_get_loop_body(expr).args
            transformed_stmt = AstWalk.ast_walk(stmt, gen_transformed_visit,
                                                transform_context)
            push!(stmt_array, transformed_stmt)
        end
        transformed_expr = scope_stmt
    elseif expr.head == :if
    elseif expr.head == :while
        scope_stmt = :(while a < 1 end)
        scope_stmt.args[1] = for_get_loop_condition(expr)
        stmt_array = while_get_loop_body(scope_stmt).args
    else
        error("syntax not supported", expr)
    end
    transform_context.curr_child_scope_index += 1
    return transformed_expr
end

function gen_transformed_visit(expr,
                               transform_context::TransformContext,
                               top_level::Integer,
                               is_top_level::Bool,
                               read::Bool)
    if isa(expr, Symbol)
        return esc(expr)
    elseif isa(expr, Expr)
        dump(expr)
        head = expr.head
        args = expr.args
        if head in Set([:for, :while, :if])
            return transform_child_scope(expr, transform_context)
        elseif head == :macrocall
            if macrocall_get_symbol(expr) == Symbol("@accumulator")
                return nothing
            elseif macrocall_get_symbol(expr) == Symbol("@parallel_for") ||
                macrocall_get_symbol(expr) == Symbol("@ordered_parallel_for")
                par_for_index = transform_context.curr_par_for_index
                par_for_context = transform_context.scope_context.par_for_context[par_for_index]
                par_for_scope = transform_context.scope_context.par_for_scope[par_for_index]
                transform_context.curr_par_for_index += 1
                return par_for_context.parallelized_loop
            else
                return esc(expr)
            end
        else
            return esc(expr)
        end
    else
        return esc(expr)
    end
end
