import Base

@enum DistArrayParentType DistArrayParentType_text_file =
    1 DistArrayParentType_dist_array =
    2 DistArrayParentType_init =
    3

@enum DistArrayInitType DistArrayInitType_empty =
    1 DistArrayInitType_uniform_random = 2

type DistArray{T} <: AbstractArray{T}
    id::Int32
    parent_type::DistArrayParentType
    flatten_results::Bool
    map::Bool
    num_dims::UInt64
    ValueType::DataType
    file_path::String
    parent_id::Int32
    init_type::DistArrayInitType
    mapper_func_module::Module
    mapper_func_name::String
    is_materialized::Bool
    dims::Array{Int64, 1}

    DistArray(id::Integer,
              parent_type::DistArrayParentType,
              flatten_results::Bool,
              map::Bool,
              num_dims::Integer,
              ValueType::DataType,
              file_path::String,
              parent_id::Integer,
              init_type::DistArrayInitType,
              mapper_func_module::Module,
              mapper_func_name::String,
              is_materialized::Bool) = new(id,
                                           parent_type,
                                           flatten_results,
                                           map,
                                           num_dims,
                                           ValueType,
                                           file_path,
                                           parent_id,
                                           init_type,
                                           mapper_func_module,
                                           mapper_func_name,
                                           is_materialized,
                                           zeros(Int64, num_dims))

end

const dist_arrays = Dict{Int32, DistArray}()

function dist_array_parent_type_to_int32(t::DistArrayParentType)::Int32
    local ret::Int32
    if t == DistArrayParentType_text_file
        ptr_val = cglobal((:ORION_TASK_DIST_ARRAY_PARENT_TYPE_TEXT_FILE, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif t == DistArrayParentType_dist_array
        ptr_val = cglobal((:ORION_TASK_DIST_ARRAY_PARENT_TYPE_DIST_ARRAY, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif t == DistArrayParentType_init
        ptr_val = cglobal((:ORION_TASK_DIST_ARRAY_PARENT_TYPE_INIT, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    else
        error("Unknown ", t)
    end
    return ret
end

function dist_array_init_type_to_int32(t::DistArrayInitType)::Int32
    local ret::Int32
    if t == DistArrayInitType_empty
        ptr_val = cglobal((:ORION_TASK_DIST_ARRAY_INIT_TYPE_EMPTY, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif t == DistArrayInitType_random
        ptr_val = cglobal((:ORION_TASK_DIST_ARRAY_INIT_TYPE_RANDOM, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    else
        error("Unknown ", t)
    end
    return ret
end

function text_file(
    file_path::AbstractString,
    parser_func::Function)::DistArray

    parser_func_module = which(Base.function_name(parser_func))
    parser_func_name = string(Base.function_name(parser_func))

    ValueType, num_dims, flatten_results =
        parse_map_function(parser_func, (String,))

    id = length(dist_arrays)
    dist_array = DistArray{ValueType}(
        id,
        DistArrayParentType_text_file,
        flatten_results,
        true,
        num_dims,
        ValueType,
        file_path,
        -1,
        DistArrayInitType_empty,
        parser_func_module,
        parser_func_name,
        false)
    dist_arrays[id] = dist_array
    return dist_array
end

function rand(ValueType::DataType, dims...)::DistArray
    id = length(dist_arrays)
    dist_array = DistArray{ValueType}(
        id,
        DistArrayParentType_init,
        false,
        false,
        length(dims),
        ValueType,
        "",
        -1,
        DistArrayInitType_uniform_random,
        Module(),
        "",
        false)
    dist_array.dims = [dims...]
    dist_arrays[id] = dist_array
    return dist_array
end

function rand(dims...)::DistArray
    rand(Float32, dims...)
end

function materialize(dist_array::DistArray)
    if dist_array.is_materialized
        return
    end

    if dist_array.parent_type == DistArrayParentType_text_file
        ccall((:orion_create_dist_array, lib_path),
              Void, (Int32, Int32, Bool, Bool, UInt64, Int32,
                     Cstring, Int32, Int32, Int32, Cstring, Ptr{Int64}),
              dist_array.id,
              dist_array_parent_type_to_int32(dist_array.parent_type),
              dist_array.map,
              dist_array.flatten_results,
              dist_array.num_dims,
              data_type_to_int32(dist_array.ValueType),
              dist_array.file_path,
              dist_array.parent_id,
              dist_array_init_type_to_int32(dist_array.init_type),
              module_to_int32(dist_array.mapper_func_module),
              dist_array.mapper_func_name,
              dist_array.dims)
    elseif dist_array.parent_type == DistArrayParentType_dist_array

    elseif dist_array.parent_type == DistArrayParentType_init
        ccall((:orion_create_dist_array, lib_path),
              Void, (Int32, Int32, Bool, Bool, UInt64, Int32,
                     Cstring, Int32, Int32, Int32, Cstring, Ptr{Int64}),
              dist_array.id,
              dist_array_parent_type_to_int32(dist_array.parent_type),
              dist_array.map,
              dist_array.flatten_results,
              dist_array.num_dims,
              data_type_to_int32(dist_array.ValueType),
              "",
              -1,
              dist_array_init_type_to_int32(dist_array.init_type),
              -1,
              "",
              dist_array.dims)
   end
              dist_array.is_materialized = true
end


function Base.size(dist_array::DistArray)
    return tuple(dist_array.dims...)
end
