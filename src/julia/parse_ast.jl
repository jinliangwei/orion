@inline function macrocall_oget_symbol(macro_identifier::Symbol)::Symbol
    return macro_identifier
end

@inline function macrocall_get_symbol(macro_identifier::Expr)::Symbol
    if macro_identifier.head == :(.)
        if typeof(macro_identifier.args[2]) == QuoteNode
            return macro_identifier.args[2].value
        elseif typeof(macro_identifier.args[2]) == Expr
            @assert macro_identifier.args[2].head == :quote
            return macro_identifier.args[2].args[1]
        else
            dump(macro_identifier)
            error("Unknown macro form")
        end
    else
        dump(macro_identifier)
        error("Unknown macro form")
    end
end

@inline function macrocall_get_module(macro_identifier::Symbol)::Symbol
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

@inline function ref_get_var(ref_stmt::Expr)
    return ref_stmt.args[1]
end

@inline function ref_get_subscripts(ref_stmt::Expr)::Array
    return ref_stmt.args[2:length(ex.args)]
end
