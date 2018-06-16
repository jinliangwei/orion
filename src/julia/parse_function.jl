# A map function maps a DistArray to another DistArray.
# The function signature depends on the map type:
# 1) map - map_func(key::Tuple, value::ValueType)::Tuple{Tuple, NewValueType}
# 2) map_fixed_keys - map_func(key::Tuple, value::OldValueType)::NewValueType
# 3) map_values - map_func(value::OldValueType)::NewValueType
# 4) map_values_new_keys - map_func(value::OldValueType)::Tuple{Tuple, NewValueType}
# The function may also return an array of the above types, in this case we set flatten to true.
# The new and old value types must be a primitive type.

function parse_map_new_keys_function(
    func::Function,
    arg_types::Tuple, # may have multiple methods for this function
    flatten::Bool
)::Tuple{Any, Int64}
    rettype_array = Base.return_types(func, arg_types)
    if length(rettype_array) == 0
        return (Any, 0)
    end
    @assert length(rettype_array) == 1
    rettype = first(rettype_array)
    if flatten
        @assert issubtype(rettype, Array)
        rettype = rettype.parameters[1]
    end
    local num_dims
    @assert issubtype(rettype, Tuple)
    @assert length(rettype.parameters) == 2
    key_type = fieldtype(rettype, 1)
    @assert issubtype(key_type, Tuple)
    @assert !issubtype(first(key_type.parameters), Vararg) "TODO: the number of dimensions of a key must be a compile time constant"
    num_dims = length(key_type.parameters)
    new_value_type = fieldtype(rettype, 2)
    return (new_value_type, num_dims)
end

function parse_map_fixed_keys_function(
    func::Function,
    arg_types::Tuple, # may have multiple methods for this function
    flatten::Bool
)::Any
    rettype_array = Base.return_types(func, arg_types)
    if length(rettype_array) == 0
        return Any
    end
    @assert length(rettype_array) == 1
    rettype = first(rettype_array)
    if flatten
        @assert issubtype(rettype, Array)
        rettype = rettype.parameters[1]
    end
    return rettype
end
