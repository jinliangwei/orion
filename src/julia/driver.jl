export helloworld

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
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
          executor_id, code, get_result_type_value(ResultType),
          result_buff);
    return result_buff[1]
end

function stop()
    ccall((:orion_stop, lib_path), Void, ())
end

function get_result_type_value(ResultType::DataType)::Int32
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

type DistArray{T} <: AbstractArray{T}
end

function text_file(file_path::AbstractString,
                   mapper::Function)
    return DistArray{Float64}()
end

function text_file(file_path::AbstractString)
    return DistArray{String}()
end

function get_dimensions(array::DistArray)
    return 1, 1
end

function rand(dims...)
end

macro iterative(loop)
    return :(println("loop replaced"))
end
