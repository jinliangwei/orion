@inline function macrocall_get_symbol(macrocall_expr::Expr)::Symbol
    macro_identifier = macrocall_expr.args[1]
    if isa(macro_identifier, Symbol)
        return macro_identifier
    elseif macro_identifier.head == :(.)
        if isa(macro_identifier.args[2], QuoteNode)
            return macro_identifier.args[2].value
        elseif isa(macro_identifier.args[2], Expr) &&
            macro_identifier.args[2].head == :quote
            return macro_identifier.args[2].args[1]
        else
            dump(macro_identifier)
            error("Unknown macro form ", macrocall_expr, " ", macro_identifier)
        end
    else
        dump(macro_identifier)
        error("Unknown macro form")
    end
end

@inline function macrocall_get_module(macrocall_expr::Expr)::Symbol
    macro_identifier = macrocall_expr.args[1]
    if isa(macro_identifier, Symbol)
        return Symbol(which(macro_identifier))
    elseif macro_identifier.head == :(.)
        return macro_identifier.args[1]
    else
        dump(macro_identifier)
        error("Unknown macro form")
    end
end

@inline function for_get_iteration_var(loop_stmt::Expr)::Symbol
    return loop_stmt.args[1].args[1]
end

@inline function for_get_iteration_space(loop_stmt::Expr)
    return loop_stmt.args[1].args[2]
end

@inline function ref_get_referenced_var(ref_expr::Expr)
    return ref_expr.args[1]
end

function ref_get_root_var(ref_expr::Expr)
    expr = ref_expr
    while !isa(expr, Symbol)
        @assert isa(expr, Expr)
        expr = expr.args[1]
    end
    return expr
end

@inline function ref_get_subscripts(ref_expr::Expr)::Array
    return ref_expr.args[2:length(ex.args)]
end

@inline function assignment_get_assigned_to(expr::Expr)
    return expr.args[1]
end

@inline function assignment_get_assigned_from(expr::Expr)
    return expr.args[2]
end

@inline function for_get_loop_range(expr::Expr)
    return expr.args[1]
end

@inline function for_get_loop_body(expr::Expr)
    return expr.args[2]
end

@inline function while_get_loop_condition(expr::Expr)
    return expr.args[1]
end

@inline function while_get_loop_body(expr::Expr)
    return expr.args[2]
end

@inline function is_variable_definition(expr)::Bool
    return isa(expr, Expr) && expr.head == :(=)
end

@inline function is_for_loop(expr)::Bool
    return isa(expr, Expr) && expr.head == :for
end

@inline function is_ref(expr)::Bool
    return isa(expr, Expr) && expr.head == :ref
end

@inline function is_dot(expr)::Bool
    return isa(expr, Expr) && expr.head == :(.)
end

@inline function is_dist_array(sym::Symbol)::Bool
    return isdefined(current_module(), sym) &&
        typeof(eval(current_module(), sym)) <: DistArray
end
