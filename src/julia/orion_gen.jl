module OrionGen
const var_dict = Dict{Symbol, Any}()

function read(var::Symbol)
    return var_dict[var]
end

function write(var::Symbol, value)
    var_dict[var] = value
end

function delete(var::Symbol)
    delete!(var_dict, var)
end
end
