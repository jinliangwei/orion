module Ast
using Sugar

function parse_map_function(
    func::Function,
    arg_types::Tuple, # may have multiple methods for this function
    flatten::Bool = false
)::Tuple{DataType, UInt64, String}
    lambda_info = code_typed(func, arg_types)
    @assert length(methods(func)) == 1
    rettype = lambda_info[1].rettype
    if flatten
        rettype = rettype.parameters[1]
    end
    @assert length(rettype.parameters) == 2
    key_type = fieldtype(rettype, 1)
    value_type = fieldtype(rettype, 2)
    num_dims = length(key_type.parameters)
    dump(value_type)
    #    a = Sugar.sugared(func, (AbstractString,), code_typed)
#    println(a)
#    println(Base.unsafe_convert(String, ast[1].code))
    #    print(typeof(ast.code))
    #    print(fieldnames(ast))
    #    print(ast.code)
    return (value_type, num_dims)
end

end
