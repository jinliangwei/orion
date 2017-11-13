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

@inline function ref_get_referenced_var(ref_expr::Expr)
    return ref_expr.args[1]
end

@inline function ref_get_subscripts(ref_expr::Expr)::Array
    return ref_expr.args[2:length(ref_expr.args)]
end

@inline function ref_dot_get_mutated_var(ref_expr::Expr)
    referenced_var = ref_get_referenced_var(ref_expr)
    if isa(referenced_var, Symbol)
        return referenced_var
    else
        @assert isa(referenced_var, Expr)
        if referenced_var.head == :(.)
            return dot_get_mutated_var(referenced_var)
        else
            @assert referenced_var.head == :ref
            return nothing
        end
    end
end

@inline function dot_get_referenced_var(ref_expr::Expr)
    return ref_expr.args[1]
end

@inline function is_assignment(expr::Expr)
    return expr.head in Set([:(=), :(+=), :(-=), :(*=), :(/=), :(.*=), :(./=)])
end

@inline function assignment_get_assigned_to(expr::Expr)
    return expr.args[1]
end

@inline function assignment_get_assigned_from(expr::Expr)
    return expr.args[2]
end

@inline function for_get_iteration_var(loop_stmt::Expr)::Symbol
    return loop_stmt.args[1].args[1]
end

@inline function for_get_iteration_space(loop_stmt::Expr)
    return loop_stmt.args[1].args[2]
end

@inline function for_get_loop_condition(expr::Expr)
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

@inline function if_get_condition(expr::Expr)
    return expr.args[1]
end

@inline function if_get_true_branch(expr::Expr)
    return expr.args[2]
end

@inline function if_get_false_branch(expr::Expr)
    if length(expr.args) == 2
        return nothing
    else
        return expr.args[3]
    end
end

@inline function call_get_func_name(expr::Expr)
    return expr.args[1]
end

@inline function call_get_arguments(expr::Expr)
    return expr.args[2:end]
end

@inline function block_get_stmts(expr::Expr)
    return expr.args
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

@inline function is_keyword(sym::Symbol)::Bool
    return sym in Set([:end, :(:)])
end
