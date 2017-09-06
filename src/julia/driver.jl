export helloworld, local_helloworld, glog_init

function module_to_int32(m::Symbol)::Int32
    local ret = -1
    println(m)
    if m == :Core
        ptr_val = cglobal((:ORION_JULIA_MODULE_CORE, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif m == :Base
        ptr_val = cglobal((:ORION_JULIA_MODULE_BASE, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif m == :Main
        ptr_val = cglobal((:ORION_JULIA_MODULE_MAIN, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    elseif m == :OrionGen
        ptr_val = cglobal((:ORION_JULIA_MODULE_ORION_GEN, lib_path), Int32)
        ret = unsafe_load(ptr_val)
    else
        error("Unknown module ", m)
    end
    return ret
end

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
end

function local_helloworld()
    println("hello world!")
end

function glog_init()
    ccall((:orion_glog_init, lib_path), Void, ())
end

function init(
    master_ip::AbstractString,
    master_port::Integer,
    comm_buff_capacity::Integer,
    num_executors::Integer)
    ccall((:orion_init, lib_path),
          Void, (Cstring, UInt16, UInt64, UInt64), master_ip, master_port,
          comm_buff_capacity, num_executors)
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
    local ret
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

function eval_expr_on_all(ex, eval_module::Symbol)
    buff = IOBuffer()
    serialize(buff, ex)
    buff_array = takebuf_array(buff)
    result_array = ccall((:orion_eval_expr_on_all, lib_path),
                         Any, (Ref{UInt8}, UInt64, Int32),
                         buff_array, length(buff_array),
                         module_to_int32(eval_module))
    #result_array = Base.cconvert(Array{Array{UInt8}}, result_bytes)

    println("result type = ", typeof(result_array),
            " length = ", length(result_array))
    #println("result = ", result_array[1])
end

function create_accumulator(var::Symbol)
    println("created accumulator variable ", string(var))
end

function define_var(var::Symbol)
    @assert isdefined(current_module(), var)
    value = eval(current_module(), var)
    typ = typeof(eval(current_module(), var))
    println("define variable ", var,
            " value = ", value,
            " type = ", typ)

    buff = IOBuffer()
    serialize(buff, value)
    buff_array = takebuf_array(buff)
    ccall((:orion_define_var, lib_path),
          Void, (Cstring, Ptr{UInt8}, UInt64),
          string(var), buff_array, length(buff_array))
end

function define_var(var_set::Set{Symbol})
    for var in var_set
        define_var(var)
    end
end
