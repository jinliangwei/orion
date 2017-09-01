function get_vars_visit(expr::Any,
                        scope_context::ScopeContext,
                        top_level::Integer,
                        is_top_level::Bool,
                        read::Bool)
    if isa(expr, Symbol)
        if expr == Symbol("@accumulator") ||
            expr == Symbol("@parallel_for") ||
            expr == :(:)
            return
        end

        info = VarInfo()
        if !read
            info.is_modified = true
        end
        add_var!(scope_context, expr, info)
        return
    end
    if isa(expr, Number) || isa(expr, String)
        return
    end
    @assert isa(expr, Expr) "must be an Expr"
    head = expr.head
    args = expr.args
    if head == :macrocall
        if macrocall_get_symbol(args[1]) == Symbol("@accumulator")
            @assert args[2].head == :(=)
            var = args[2].args[1]
            @assert isa(var, Symbol)
            info = VarInfo()
            info.is_accumulator = true
            add_var!(scope_context, var, info)
            println("just befoer return")
            return AstWalk.AST_WALK_RECURSE
         elseif macrocall_get_symbol(args[1]) == Symbol("@parallel_for")
            @assert scope_context.parent_scope == nothing
            @assert length(args) == 2
            @assert args[2].head == :for
            loop_stmt = args[2]
            iteration_var = for_get_iteration_var(loop_stmt)
            iteration_space = for_get_iteration_space(loop_stmt)
            @assert isa(iteration_space, Symbol)
            @assert isdefined(iteration_space)
            @assert isa(eval(current_module(), iteration_space), DistArray)
            par_for_context = ParForContext(iteration_var,
                                            iteration_space,
                                            args[2])
            push!(scope_context.par_for_context, par_for_context)
            par_for_scope = get_vars!(scope_context, loop_stmt)
            add_child_scope!(scope_context, par_for_scope, true)
            return expr
        else
            error("unsupported macro call in Orion transform scope")
        end
    elseif head == :for
        @assert args[2].head == :block
        for ex in args[2].args
            child_scope = get_vars!(scope_context, ex)
            if child_scope != nothing
                add_child_scope!(scope_context, child_scope, false)
            end
        end
        return expr
    elseif head == :(=) ||
        head == :(+=) ||
        head == :(-=) ||
        head == :(*=) ||
        head == :(/=)
       if isa(args[1], Symbol)
            var = expr.args[1]
            info = VarInfo()
            info.is_assigned_to = true
           add_var!(scope_context, var, info)
       else
           @assert isa(args[1], Expr)
           @assert args[1].head == :ref "syntax not supported"
           var = ref_get_var(args[1])
           @assert isa(var, Symbol)
           info = VarInfo()
           info.is_modified = true
           add_var!(scope_context, var, info)
       end
        return AstWalk.AST_WALK_RECURSE
    elseif expr.head == :call
    elseif expr.head == :ref
    end
end

# expr is contained in the scope denoted by scope_context
function get_vars!(scope_context::Any, expr)
    if isa(expr, Expr)
        head = expr.head
        if head == :for ||
            head == :while ||
            # for list comprehension
            head == :generator
            child_scope_context = ScopeContext()
            child_scope_context.parent_scope = scope_context
            AstWalk.ast_walk(expr, get_vars_visit, child_scope_context)
            return child_scope_context
        else
            AstWalk.ast_walk(expr, get_vars_visit, scope_context)
            return nothing
        end
    else
        AstWalk.ast_walk(expr, get_vars_visit, scope_context)
        return nothing
    end
end

function translate_stmt(expr::Expr, context::ScopeContext)
    if expr.head == :macrocall
        if macrocall_get_module(expr) != :Orion
            return expr
        end
        return nothing
#        if macrocall_get_symbol(expr) == Symbol("@accumulator")
#            return macrocall_get_arguments(expr)[1]
#        elseif expr.args[1].args[2].args[1] == Symbol("@parallel_for")
#            return nothing
#        end
    elseif expr.head == :(=)
        return nothing
    elseif expr.head == :(+=)
        return nothing
    elseif expr.head == :line
        return nothing
    elseif expr.head == :for
        return nothing
    else
        return nothing
    end
end
