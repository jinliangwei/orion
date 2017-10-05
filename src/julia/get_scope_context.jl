type ScopeContextProcessInfo
    scope_context
    helper_context
    ScopeContextProcessInfo(scope_context) = new(scope_context, nothing)
end
function get_scope_context_visit(expr::Any,
                                 scope_context_info::ScopeContextProcessInfo,
                                 top_level::Integer,
                                 is_top_level::Bool,
                                 read::Bool)
    scope_context = scope_context_info.scope_context
    is_par_for = isa(scope_context_info.helper_context, ParForContext)

    if isa(expr, Symbol)
        if expr == Symbol("@accumulator") ||
            expr == Symbol("@parallel_for") ||
            expr == :(:) ||
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
        if macrocall_get_symbol(expr) == Symbol("@accumulator")
            @assert is_variable_definition(args[2])
            var = assignment_get_assigned_to(args[2])
            @assert isa(var, Symbol)
            info = VarInfo()
            info.is_accumulator = true
            add_var!(scope_context, var, info)
            scope_context.accumulator_info_dict[var] = AccumulatorInfo(var,
                                                                       assignment_get_assigned_from(args[2]),
                                                                       args[3])
            return AstWalk.AST_WALK_RECURSE
        elseif macrocall_get_symbol(expr) == Symbol("@parallel_for") ||
            macrocall_get_symbol(expr) == Symbol("@ordered_parallel_for")
            @assert scope_context.parent_scope == nothing
            @assert length(args) == 2
            @assert is_for_loop(args[2]) println(args[2])
            loop_stmt = args[2]
            iteration_var = for_get_iteration_var(loop_stmt)
            iteration_space = for_get_iteration_space(loop_stmt)
            @assert isa(iteration_space, Symbol)
            @assert isdefined(iteration_space)
            @assert isa(eval(current_module(), iteration_space), DistArray)

            par_for_context = ParForContext(iteration_var,
                                            iteration_space,
                                            loop_stmt,
                                            macrocall_get_symbol(expr) == Symbol("@ordered_parallel_for"))
            push!(scope_context.par_for_context, par_for_context)
            par_for_scope = get_scope_context!(scope_context, loop_stmt, par_for_context)
            add_child_scope!(scope_context, par_for_scope, true)
            return expr
        else
            return expr
        end
    elseif head == :for
        loop_stmt = args[2]
        child_scope = get_scope_context!(scope_context, loop_stmt)
        return expr
    elseif head in Set([:(=), :(+=), :(-=), :(*=), :(/=), :(.*=), :(./=)])
        return AstWalk.AST_WALK_RECURSE
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
        if isa(referenced_var, Symbol)
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
                            expr,
                            helper_context = false)
    if isa(expr, Expr)
        head = expr.head
        if head == :for ||
            head == :while ||
            # for list comprehension
            head == :generator
            child_scope_context = ScopeContext()
            child_scope_context.parent_scope = scope_context
            child_scope_context_info = ScopeContextProcessInfo(child_scope_context)
            child_scope_context_info.helper_context = helper_context
            for arg in expr.args
                AstWalk.ast_walk(arg, get_scope_context_visit, child_scope_context_info)
            end
            return child_scope_context
        else
            scope_context_info = ScopeContextProcessInfo(scope_context)
            AstWalk.ast_walk(expr, get_scope_context_visit, scope_context_info)
            return nothing
        end
    else
        scope_context_info = ScopeContextProcessInfo(scope_context)
        AstWalk.ast_walk(expr, get_scope_context_visit, scope_context_info)
        return nothing
    end
end
