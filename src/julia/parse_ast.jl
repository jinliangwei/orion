@inline function macrocall_get_symbol(ex::Expr)::Symbol
    if isa(ex.args[1], Symbol)
        return ex.args[1]
    elseif ex.args[1].head == :(.)
        if typeof(ex.args[1].args[2]) == QuoteNode
            return ex.args[1].args[2].value
        else
            return ex.args[1].args[2].args[1]
        end
    else
        error("Unknown macro form")
    end
end

@inline function macrocall_get_module(ex::Expr)::Symbol
    if isa(ex.args[1], Symbol)
        return ex.args[1]
    elseif ex.args[1].head == :(.)
        return ex.args[1].args[1]
    else
        error("Unknown macro form")
    end
end

@inline function for_get_iteration_var(ex::Expr)::Symbol
    return ex.args[2].args[1].args[1]
end

@inline function for_get_iteration_space(ex::Expr)
    return ex.args[2].args[1].args[2]
end

@inline function ref_get_var(ex::Expr)
    return ex.args[1]
end

@inline function ref_get_subscripts(ex::Expr)::Array
    return ex.args[2:length(ex.args)]
end

@inline function macro_get_arguments(ex::Expr)::Array
    return ex.args[2:length(ex.args)]
end
