module Ast
using Sugar

function parse_map_function(
    func::Function,
    arg_types::Tuple, # may have multiple methods for this function
    flatten::Bool = false
)::Tuple{DataType, UInt64}
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
    return (value_type, num_dims)
end

function test_sugar(func::Function, arg_types::Tuple)
    #sugared_func = Sugar.sugared(func, (AbstractString,), code_typed)
    #println(sugared_func)
    #a = Sugar.get_lambda(code_typed, func, arg_types)
    #println(a)
    println(Base.return_types(func, arg_types))
    println(Sugar.get_static_parameters(func, (String,)))
    #println(Sugar.get_source(Sugar.get_method(func, (String,))))
    func_ast = Sugar.macro_form(func, (String,))[1]
    println(func_ast)

    buff = IOBuffer()

    #print(buff, func_ast)
    #println(takebuf_array(buff))
    #print(buff, func_ast)
    #func_ast_str = takebuf_string(buff)
    #println(func_ast_str)
    #eval(parse(func_ast_str))
    serialize(buff, func_ast)
    seekstart(buff)
    dfunc_ast = deserialize(buff)
    eval(dfunc_ast)


    #eval(func_ast)
    #print(Sugar.macro_form(func, (String,))[1])
    #eval("function tf()\n  return 2\nend")
    a = parse_line("1,2,0.2")
    println(a[1])

end

end
