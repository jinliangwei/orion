# A map function maps a DistArray to another DistArray.
# The function signature depends on the map type:
# 1) map - map_func(key::Tuple, value::ValueType)::Tuple{Tuple, NewValueType}
# 2) map_fixed_keys - map_func(key::Tuple, value::OldValueType)::NewValueType
# 3) map_values - map_func(value::OldValueType)::NewValueType
# 4) map_values_new_keys - map_func(value::OldValueType)::Tuple{Tuple, NewValueType}
# The function may also return an array of the above types, in this case we set flatten to true.
# The new and old value types must be a primitive type.

function parse_map_function(
    func::Function,
    arg_types::Tuple # may have multiple methods for this function
)::Tuple{DataType, UInt64, Bool, Bool}
#    if !all(isleaftype, arg_types)
#        error("Not all types are concrete: $arg_types")
#    end
    rettype_array = Base.return_types(func, arg_types)
    println(length(rettype_array))
    @assert length(rettype_array) == 1
    rettype = first(rettype_array)
    local flatten_results = false
    if issubtype(rettype, Array)
        rettype = rettype.parameters[1]
        flatten_results = true
    end
    local preserving_keys
    local new_value_type
    local num_dims
    if issubtype(rettype, Tuple)
        preserving_keys = false
        @assert length(rettype.parameters) == 2
        key_type = fieldtype(rettype, 1)
        @assert issubtype(key_type, Tuple)
        @assert !(first(key_type.parameters) <: Vararg) "TODO: the number of dimensions of a key must be a compile time constant"
        num_dims = length(key_type.parameters)
        new_value_type = fieldtype(rettype, 2)
    else
        preserving_keys = true
        num_dims = 0
        new_value_type = rettype
    end
    return (new_value_type, num_dims, flatten_results, preserving_keys)
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
