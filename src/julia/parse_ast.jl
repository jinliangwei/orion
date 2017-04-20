@inline function macrocall_get_symbol(ex::Expr)::Symbol
    if ex.args[1].head == :(.)
        if typeof(ex.args[1].args[2]) == QuoteNode
            return ex.args[1].args[2].value
        else
            dump(ex.args[1].args[2].args[1])
            return ex.args[1].args[2].args[1]
        end
    else
        return ex.args[1]
    end
end

@inline function macrocall_get_module(ex::Expr)::Module
    if ex.args[1].head == :(.)
        return Module(ex.args[1].args[1])
    else
        return which(ex.args[1])
    end
end

@inline function macro_get_arguments(ex::Expr)::Array
    return ex.args[2:length(ex.args)]
end
