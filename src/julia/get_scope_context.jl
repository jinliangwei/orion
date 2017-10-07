function get_scope_context_visit(expr::Any,
                                 scope_context::ScopeContext,
                                 top_level::Integer,
                                 is_top_level::Bool,
                                 read::Bool)
    if isa(expr, Symbol)
        if expr == :(:) ||
            (isdefined(current_module(), expr) &&
             isa(eval(current_module(), expr), Function))
            return expr
        end

        info = VarInfo()
        info.is_mutated = !read
        info.is_assigned_to = !read
        add_var!(scope_context, expr, info)
        return expr
    end
    if isa(expr, Number) || isa(expr, String) || isa(expr, QuoteNode)
        return expr
    end
    @assert isa(expr, Expr) "must be an Expr"
    head = expr.head
    args = expr.args
    if head == :macrocall
        return expr
    elseif head == :for
        loop_stmt = args[2]
        child_scope = get_scope_context!(scope_context, loop_stmt)
        add_child_scope!(scope_context, child_scope, true)
        return expr
    elseif expr.head in Set([:call, :invoke, :call1, :foreigncall])
        if call_get_func_name(expr) in Set([:(+), :(-), :(*), :(/)])
            return AstWalk.AST_WALK_RECURSE
        end
        for arg in expr.args[2:end]
            if isa(arg, Symbol)
                info = VarInfo()
                info.is_mutated = true
                add_var!(scope_context, arg, info)
            elseif isa(arg, Expr)
                if expr.head == :(.)
                    root_var = ref_dot_get_mutated_var(expr)
                    info = VarInfo()
                    info.is_mutated = true
                    add_var!(scope_context, root_var, info)
                end
            end
        end
        return AstWalk.AST_WALK_RECURSE
    elseif expr.head == :(.)
        if !read
            mutated_var = ref_dot_get_mutated_var(expr)
            if mutated_var != nothing
                info = VarInfo()
                info.is_mutated = true
                add_var!(scope_context, mutated_var, info)
            end
        end
        referenced_var = dot_get_referenced_var(expr)
        if isa(referenced_var, Symbol)
            info = VarInfo()
            add_var!(scope_context, referenced_var, info)
            return expr
        else
            return AstWalk.AST_WALK_RECURSE
        end
    elseif expr.head == :ref
        if !read
            mutated_var = ref_dot_get_mutated_var(expr)
            if mutated_var != nothing
                info = VarInfo()
                info.is_mutated = true
                add_var!(scope_context, mutated_var, info)
            end
        end
        referenced_var = ref_get_referenced_var(expr)
        if isa(referenced_var, Symbol) &&
            (!isdefined(current_module(), referenced_var) ||
             !isa(eval(current_module(), referenced_var), Module))
            info = VarInfo()
            add_var!(scope_context, referenced_var, info)
        end
        return AstWalk.AST_WALK_RECURSE
    elseif head == :block
        return AstWalk.AST_WALK_RECURSE
    else
        return AstWalk.AST_WALK_RECURSE
    end
end

# expr is contained in the scope denoted by scope_context
function get_scope_context!(scope_context::Any,
                            expr)
    if scope_context == nothing ||
        isa(expr, Expr) ||
        expr.head == :for ||
        expr.head == :while ||
        # for list comprehension
        expr.head == :generator
        child_scope_context = ScopeContext()
        child_scope_context.parent_scope = scope_context
        for arg in expr.args
            AstWalk.ast_walk(arg, get_scope_context_visit, child_scope_context)
        end
        return child_scope_context
    else
        AstWalk.ast_walk(expr, get_scope_context_visit, scope_context)
        return nothing
    end
end
