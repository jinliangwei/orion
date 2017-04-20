# translate
function translate_iterative_stmt(expr::Expr, context::Context)
    if expr.head == :macrocall
        if macrocall_get_module(expr) != Orion
            return esc(expr)
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
