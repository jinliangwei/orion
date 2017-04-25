# translate

# expr is contained in the scope denoted by scope_context
function get_vars!(scope_context::ScopeContext, expr)
    if isa(expr, Symbol)
        add_var!(scope_context, expr, VarInfo())
        return
    end
    if isa(expr, Number) || isa(expr, String)
        return
    end
    @assert isa(expr, Expr)
    if expr.head == :macrocall
        for arg in expr.args[2:length(expr.args)]
            dump(arg)
            get_vars!(scope_context, arg)
        end
        println("got vars from all macro args")
        if macrocall_get_module(expr) == :Orion
            println(macrocall_get_symbol(expr))
            if macrocall_get_symbol(expr) == Symbol("@accumulator")
                @assert expr.args[2].head == :(=) "Orion.@accumulator may only be applied to assignment"
                info = VarInfo()
                var = expr.args[2].args[1]
                @assert isa(var, Symbol)
                info.is_accumulator = true
                info.is_assigned_to = true
                add_var!(scope_context, var, info)
            elseif macrocall_get_symbol(expr) == Symbol("@parallel_for")
                @assert scope_context.parent_scope == nothing
                @assert expr.args[2].head == :for
                iteration_var = for_get_iteration_var(expr)
                iteration_space = for_get_iteration_space(expr)
                @assert isa(iteration_space, Symbol)
                @assert isa(eval(current_module(), iteration_space), DistArray)
                par_for_context = ParForContext(iteration_var, iteration_space, expr.args[2])
                push!(scope_context.par_for_context, par_for_context)
                par_for_scope = scope_context.child_scope[length(scope_context.child_scope)]
                scope_context.child_scope = scope_context.child_scope[1:(length(scope_context.child_scope) - 1)]
                prepare_par_for_scope(par_for_scope)
                push!(scope_context.par_for_scope, par_for_scope)
            else
                error("unsupported macro call in Orion transform scope")
            end
        end
        println("done processing macro")
    elseif expr.head == :for
        #TODO: deal with loop header
        temp_scope_context = ScopeContext(scope_context)
        @assert expr.args[2].head == :block
        for ex in expr.args[2].args
            get_vars!(temp_scope_context, ex)
        end
        add_child_scope!(scope_context, temp_scope_context, false)
    elseif expr.head == :(=) ||
        expr.head == :(+=) ||
        expr.head == :(-=) ||
        expr.head == :(*=) ||
        expr.head == :(/=)
        #println("process assignment ", expr)
        if isa(expr.args[1], Symbol)
            var = expr.args[1]
            info = VarInfo()
            info.is_assigned_to = true
            add_var!(scope_context, var, info)
        else
            @assert isa(expr.args[1], Expr)
            @assert expr.args[1].head == :ref "syntax not supported"
            var = ref_get_var(expr.args[1])
            @assert isa(var, Symbol)
            info = VarInfo()
            info.is_modified = true
            add_var!(scope_context, var, info)

            subscripts = ref_get_subscripts(expr)
            for s in subscripts
                if isa(s, Symbol) && s == :(:)
                    continue
                end
                get_vars!(scope_context, s)
            end
        end
        get_vars!(scope_context, expr.args[2])
    elseif expr.head == :call
        println("deal with function call")
        dump(expr)
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
