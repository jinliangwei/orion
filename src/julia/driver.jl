export helloworld

using .Ast

@enum DistArrayParentType DistArrayParentType_text_file =
    1 DistArrayParentType_dist_array =
    2 DistArrayParentType_init =
    3

function dist_array_parent_type_to_int32
end

@enum DistArrayInitType DistArrayInitType_empty =
    1 DistARrayInitType_uniform_random = 2

type DistArray{T} <: AbstractArray{T}
    id::Int32
    parent_type::DistArrayParentType
    flatten_results::Bool
    value_only::Bool
    parse::Bool
    num_dims::UInt64
    ValueType::DataType
    file_path::String
    parent_id::Int32
    parser_func::String
    parser_func_name::String
    is_materialized::Bool
    function DistArray(id::Integer,
                       parent_type::DistArrayParentType,
                       flatten_results::Bool,
                       value_only::Bool,
                       parse::Bool,
                       num_dims::UInt64,
                       value_type::DataType)
        new(id, parent_type, flatten_results, value_only, parse,
            num_dims, value_type, "", -1, "", "", false)
    end
end

dist_arrays = Dict{Int32, DistArray}()

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
end

function local_helloworld()
    println("hello world!")
end

function glog_init(glogconfig::Ptr{Void})
    ccall((:orion_glog_init, lib_path), Void, (Ptr{Void},), glogconfig)
end

function init(
    master_ip::AbstractString,
    master_port::Integer,
    comm_buff_capacity::Integer)
    ccall((:orion_init, lib_path),
          Void, (Cstring, UInt16, UInt64), master_ip, master_port,
          comm_buff_capacity)
end

function execute_code(
    executor_id::Integer,
    code::AbstractString,
    ResultType::DataType)
    result_buff = Array{ResultType}(1)
    ccall((:orion_execute_code_on_one, lib_path),
          Void, (Int32, Cstring, Int32, Ptr{Void}),
          executor_id, code, data_type_to_int32(ResultType),
          result_buff);
    return result_buff[1]
end

function stop()
    ccall((:orion_stop, lib_path), Void, ())
end

function data_type_to_int32(ResultType::DataType)::Int32
    if ResultType == Void
        ptr_val = cglobal((:ORION_TYPE_VOID, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif ResultType == Int8
        ptr_val = cglobal((:ORION_TYPE_INT8, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif ResultType == UInt8
        ptr_val = cglobal((:ORION_TYPE_UINT8, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif ResultType == Int16
        ptr_val = cglobal((:ORION_TYPE_INT16, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif ResultType == UInt16
        ptr_val = cglobal((:ORION_TYPE_UINT16, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif ResultType == Int32
        ptr_val = cglobal((:ORION_TYPE_INT32, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif ResultType == UInt32
        ptr_val = cglobal((:ORION_TYPE_UINT32, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif ResultType == Int16
        ptr_val = cglobal((:ORION_TYPE_INT64, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif ResultType == UInt64
        ptr_val = cglobal((:ORION_TYPE_UINT64, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif ResultType == Float32
        ptr_val = cglobal((:ORION_TYPE_FLOAT32, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif ResultType == Float64
        ptr_val = cglobal((:ORION_TYPE_FLOAT64, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif ResultType == String
        ptr_val = cglobal((:ORION_TYPE_STRING, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    else
        ret = 12
    end
    return ret
end

function eval_expr_on_all(ex::Expr, ResultType::DataType)
    buff = IOBuffer()
    serialize(buff, ex)
    buff_array = takebuf_array(buff)
    result_buff = Array{ResultType}(1)
    ccall((:orion_eval_expr_on_all, lib_path),
          Void, (Ptr{UInt8}, UInt64, Int32, Ptr{Void}),
          buff_array, length(buff_array),
          data_type_to_int32(ResultType),
          result_buff);
end

function text_file(
    file_path::AbstractString,
    mapper::Function,
    arg_types::Tuple=(AbstractString,),
    flatten_results::Bool=false)

    id = length(dist_arrays)
    parent_type = DistArrayParentType_text_file
    ValueType, num_dims =
        Ast.parse_map_function(mapper, arg_types, flatten_results)

    dist_array = DistArray{ValueType}(
        id,
        parent_type,
        flatten_results,
        false,
        true,
        num_dims,
        ValueType)
    dist_array.file_path = file_path
end

function text_file(file_path::AbstractString)
    return DistArray{String}()
end

function get_dimensions(array::DistArray)
    return 1, 1
end

function rand(dims...)
end

function materialize(dist_array::DistArray)
    if (dist_array.parent_type == DistArrayParentType_text_file)
    elseif (dist_array.parent_type == DistArrayParentType_dist_array)
    elseif (dist_array.parent_type == DistArrayParentType_init)
    end
end

macro iterative(loop)
    return :(println("loop replaced"))
end
