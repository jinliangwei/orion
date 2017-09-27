type TransformContext
    scope_context::ScopeContext
    stmt_array::Vector{Any}
    curr_par_for_index::Int64
    curr_child_scope_index::Int64
    TransformContext(scope_context::ScopeContext) = new(scope_context, Vector{Any}(),
                                                        0, 0)
end

function gen_transformed_loop(expr::Expr, scope_context::ScopeContext)
    @assert expr.head == :for
    loop_stmt = :(for i = 1:1 end)
    loop_stmt.args[1] = for_get_loop_range(expr)
    stmt_array = for_get_loop_body(loop_stmt).args
    transform_context = TransformContext(scope_context, stmt_array)
    for stmt in for_get_loop_body(expr).args
        AstWalk.ast_walk(stmt, gen_transformed_visit, tranform_context)
    end
end

function transform_child_scope(expr::Expr, transform_context::TransformContext)
    child_scope_index = transform_context.curr_child_scope_index
    scope_context = transform_scope.scope_context
    child_scope_context = scope_context.child_scope[child_scope_index]
    if expr.head == :for
        scope_stmt = :(for i = 1:1 end)
        scope_stmt.args[1] = for_get_loop_range(expr)
        stmt_array = for_get_loop_body(scope_stmt).args

        child_transform_context = TransformContext(child_scope_context, stmt_array)

        for stmt in for_get_loop_body(expr).args
            AstWalk.ast_walk(stmt, gen_transformed_visit, tranform_context)
        end

    elseif expr.head == :if
    elseif expr.head == :while
        scope_stmt = :(while a < 10 end)
        scope_stmt.args[1] = for_get_loop_range(expr)
        stmt_array = while_get_loop_body(scope_stmt).args
    else
        error("syntax not supported", expr)
    end

    transform_context.curr_child_scope_index += 1
end

function gen_transformed_visit(expr,
                               tranform_context::TransformContext,
                               top_level::Integer,
                               is_top_level::Bool,
                               read::Bool)

end
