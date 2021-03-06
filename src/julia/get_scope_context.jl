function get_scope_context_visit(expr::Any,
                                 scope_context::ScopeContext)
    if isa(expr, Symbol)
        if is_keyword(expr)
            return expr
        end

        if isdefined(expr) &&
            isa(eval(current_module(), expr), Function)
            return expr
        end

        if isdefined(expr) &&
            isa(eval(current_module(), expr), DataType)
            return expr
        end

        info = VarInfo()
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
        if args[1] == Symbol("__dot__")
            scope_context.is_dotted = true
            for idx in eachindex(args)
                args[idx] = AstWalk.ast_walk(args[idx], get_scope_context_visit, scope_context)
            end
            scope_context.is_dotted = false
            return expr
        else
            return AstWalk.AST_WALK_RECURSE
        end
    elseif head == :for
        child_scope = get_scope_context!(scope_context, expr)
        add_child_scope!(scope_context, child_scope)
        return expr
    elseif head in Set([:call, :invoke, :call1, :foreigncall])
        if call_get_func_name(expr) in const_func_set
            return AstWalk.AST_WALK_RECURSE
        end
        for arg in args[2:end]
            if isa(arg, Symbol) &&
                !is_keyword(arg) &&
                !(isdefined(arg) &&
                  isa(eval(which(arg), arg), Function))
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
    elseif head in Set([:(=), :(.=), :(+=), :(/=), :(*=), :(-=)])
        assigned_to = assignment_get_assigned_to(expr)
        var_mutated_vec = Vector{Tuple{Symbol, DataType, Any}}()
        assigned_to_get_mutated_var_vec(assigned_to, var_mutated_vec)
        if head == :(.=) || scope_context.is_dotted
            for (var_mutated, mutated_type, _) in var_mutated_vec
                info = VarInfo()
                info.is_mutated = true
                add_var!(scope_context, var_mutated, info)
            end
        else
            for (var_mutated, mutated_type, _) in var_mutated_vec
                info = VarInfo()
                info.is_mutated = true
                info.is_assigned_to = true
                add_var!(scope_context, var_mutated, info)
            end
        end
        return AstWalk.AST_WALK_RECURSE
    elseif head == :(.)
        return AstWalk.AST_WALK_RECURSE
    elseif head == :ref
        return AstWalk.AST_WALK_RECURSE
    elseif head == :block
        return AstWalk.AST_WALK_RECURSE
    elseif head == :global
        for arg in args
            if isa(arg, Expr) && is_assignment(arg)
                assignment_to = assignment_get_assigned_to(arg)
                info = VarInfo()
                info.is_marked_global = true
                add_var!(scope_context, assignment_to, info)
            elseif isa(arg, Symbol)
                info = VarInfo()
                info.is_marked_global = true
                add_var!(scope_context, arg, info)
            else
                error("unsupported syntax")
            end
        end
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
