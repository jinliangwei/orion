# translate

# expr is contained in the scope denoted by scope_context
function get_vars!(scope_context::ScopeContext, expr)
    if isa(expr, Symbol)
        add_inherited_var(expr, VarInfo())
        return
    end
    if isa(expr, Number) || isa(expr, String)
        return
    end
    @assert isa(expr, Expr)
    if expr.head == :macrocall
        temp_scope_context = ScopeContext()
        for arg in expr.args[2:length(expr.args)]
            get_vars!(temp_scope_context, arg)
        end
        if macrocall_get_module(expr) == :Orion
            if macrocall_get_symbol(expr) == Symbol("@accumulator")
                @assert expr.args[2].head == :(=) "Orion.@accumulator may only be applied to assignment"
                info = VarInfo()
                var = expr.args[2].args[1]
                @assert isa(var, Symbol)
                if isa(expr.args[2].args[2], Number)
                    info.ValueType = typeof(expr.args[2].args[2])
                    info.value = expr.args[2].args[2]
                end
                info.is_accumulator = true
                info.assigned_to = true
                add_local_var!(scope_context, var, info)
                merge_scope!(scope_context, temp_scope_context)
            elseif macrocall_get_symbol(expr) == Symbol("@parallel_for")
                @assert expr.args[2].head == :for
                iteration_var = for_get_iteration_var(expr)
                iteration_space = for_get_iteration_space(expr)
                @assert isa(iteration_space, Symbol)
                @assert isa(eval(current_module(), iteration_space), DistArray)
                par_for_context = ParForContext(iteration_var, iteration_space)
                push!(scope_context.par_for_context, par_for_context)
                merge_scope!(scope_context, temp_scope_context)
                push!(scope_context.par_for_scope, pop!(scope_context.child_scope))
            else
                error("unsupported macro call in Orion transform scope")
            end
        else
            merge_scope!(scope_context, temp_scope_context)
        end
    elseif expr.head == :for
        #TODO: deal with loop header
        temp_scope_context = ScopeContext()
        @assert expr.args[2].head == :block
        for ex in expr.args[2].args
            get_vars!(temp_scope_context, ex)
        end
        push!(scope_context.child_scope, temp_scope_context)
    elseif expr.head == :(=)
        println("process assignment ", expr)
        if isa(expr.args[1], Symbol)
            var = expr.args[1]
            info = VarInfo()
            info.assigned_to = true
            add_local_var!(scope_context, var, info)
        else
            @assert isa(expr.args[1], Expr)
            @assert expr.args[1].head == :ref "syntax not supported"
            var = ref_get_var(expr.args[1])
            println(var)
            println(typeof(var))
            @assert isa(var, Symbol)
            info = VarInfo()
            info.modified = true
            add_inherited_var!(scope_context, var, info)

            subscripts = ref_get_subscripts(expr)
            for s in subscripts
                if isa(s, Expr)
                    get_vars!(scope_context, s)
                elseif isa(s, Symbol)
                    if s == :(:)
                        continue
                    end
                    info = VarInfo()
                    add_inherited_vars(scope_context, var, info)
                end
            end
        end
        get_vars!(scope_context, expr.args[2])
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
