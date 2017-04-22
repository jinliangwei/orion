module Ast

function parse_map_function(
    func::Function,
    arg_types::Tuple # may have multiple methods for this function
)::Tuple{DataType, UInt64, Bool}
    if !all(isleaftype, arg_types)
        error("Not all types are concrete: $arg_types")
    end
    rettype_array = Base.return_types(func, arg_types)
    @assert length(rettype_array) == 1
    rettype = first(rettype_array)
    local flatten_results = false
    if issubtype(rettype, Array)
        rettype = rettype.parameters[1]
        flatten_results = true
    end
    @assert issubtype(rettype, Tuple)
    local num_dims
    if length(rettype.parameters) == 1
        num_dims = 0
    else
        @assert length(rettype.parameters) == 2
        key_type = fieldtype(rettype, 1)
        @assert issubtype(key_type, Tuple)
        num_dims = length(key_type.parameters)
    end
    value_type = fieldtype(rettype, 2)

    return (value_type, num_dims, flatten_results)
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