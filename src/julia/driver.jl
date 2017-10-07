export helloworld, local_helloworld, glog_init

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
    _num_executors::Integer)
    ccall((:orion_init, lib_path),
          Void, (Cstring, UInt16, UInt64, UInt64), master_ip, master_port,
          comm_buff_capacity, _num_executors)
    global const num_executors = _num_executors
    load_constants()
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
    if length(result_array) > 0
        println("typeof(result[1]) = ", typeof(result_array[1]))
    end
end

function create_accumulator(var::Symbol, init_val)
    println("created accumulator variable ", string(var),
            " init val = ", init_val)
end

function define_var(var::Symbol)
    @assert isdefined(current_module(), var)
    value = eval(current_module(), var)

    expr = :($var = $value)

    eval_expr_on_all(expr, :Main)

    #buff = IOBuffer()
    #serialize(buff, value)
    #buff_array = takebuf_array(buff)
    #ccall((:orion_define_var, lib_path),
     #     Void, (Cstring, Ptr{UInt8}, UInt64),
     #     string(var), buff_array, length(buff_array))
end

function define_vars(var_set::Set{Symbol})
    for var in var_set
        define_var(var)
    end
end

@enum ParallelSchemeType ParallelSchemeType_naive =
    1 ParallelSchemeType_1d =
    2 ParallelSchemeType_2d =
    3 ParallelSchemeType_unimodular =
    4 ParallelSchemeType_none =
    5

function exec_for_loop(iteration_space_id::Integer,
                       parallel_scheme_type::ParallelSchemeType,
                       space_partitioned_dist_array_ids::Vector{Int32},
                       time_partitioned_dist_array_ids::Vector{Int32},
                       global_indexed_dist_array_ids::Vector{Int32},
                       loop_batch_func_name::AbstractString)

end
