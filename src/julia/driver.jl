export helloworld, local_helloworld, glog_init

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
    global const num_executors = 4
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
    _num_executors::Integer,
    _num_servers::Integer)
    ccall((:orion_init, lib_path),
          Void, (Cstring, UInt16, UInt64), master_ip, master_port,
          comm_buff_capacity)
    global const num_executors = _num_executors
    global const num_servers = _num_servers
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
    return result_array
end

function define_var(var::Symbol)
    @assert isdefined(current_module(), var)
    value = eval(current_module(), var)

    expr = :($var = $value)

    eval_expr_on_all(expr, :Main)
end

function define_vars(var_set::Set{Symbol})
    for var in var_set
        define_var(var)
    end
end

function exec_for_loop(iteration_space_id::Integer,
                       parallel_scheme::ForLoopParallelScheme,
                       space_partitioned_dist_array_ids::Vector{Int32},
                       time_partitioned_dist_array_ids::Vector{Int32},
                       global_indexed_dist_array_ids::Vector{Int32},
                       buffered_dist_array_ids::Vector{Int32},
                       dist_array_buffer_ids::Vector{Int32},
                       num_buffers_each_dist_array::Vector{UInt64},
                       loop_batch_func_name::AbstractString,
                       prefetch_batch_func_name::AbstractString,
                       is_ordered::Bool)
    ccall((:orion_exec_for_loop, lib_path),
          Void, (Int32,
                 Int32,
                 Ref{Int32}, UInt64,
                 Ref{Int32}, UInt64,
                 Ref{Int32}, UInt64,
                 Ref{Int32}, UInt64,
                 Ref{Int32}, Ref{UInt64},
                 Cstring, Cstring, Bool),
          iteration_space_id,
          for_loop_parallel_scheme_to_int32(parallel_scheme),
          space_partitioned_dist_array_ids, length(space_partitioned_dist_array_ids),
          time_partitioned_dist_array_ids, length(time_partitioned_dist_array_ids),
          global_indexed_dist_array_ids, length(global_indexed_dist_array_ids),
          buffered_dist_array_ids, length(buffered_dist_array_ids),
          dist_array_buffer_ids, num_buffers_each_dist_array,
          loop_batch_func_name, prefetch_batch_func_name, is_ordered)
end

function get_aggregated_value(var_sym::Symbol, combiner_func::Symbol)
    @assert which(combiner_func) == Base
    combiner_str = string(combiner_func)
    value = ccall((:orion_get_accumulator_value, lib_path),
                  Any, (Cstring, Cstring),
                  string(var_sym), combiner_str)

    return value
end

function reset_var_value(var_sym::Symbol, value)
    expr = :($var_sym = $value)
    eval_expr_on_all(expr, :Main)
end

function reset_accumulator(var_sym::Symbol)
    reset_var_value(var_sym, accumulator_info_dict[var_sym].init_value)
end

type DistArrayBufferInfo
    buffer_id::Int32
    apply_buffer_func::Symbol
    helper_buffer_vec::Vector{Int32}
    helper_dist_array_vec::Vector{Int32}
    DistArrayBufferInfo(buffer_id::Int32,
                        apply_buffer_func::Symbol) =
                            new(buffer_id,
                                apply_buffer_func,
                                Vector{Int32}(),
                                Vector{Int32}())
end

dist_array_buffer_map = Dict{Int32, DistArrayBufferInfo}()

function set_write_buffer(dist_array::DistArray,
                          apply_func::Function,
                          dist_array_buffers...)
    apply_func_name = Base.function_name(apply_func)
    buffer_info = DistArrayBufferInfo(dist_array_buffers[1].id,
                                      apply_func_name)
    for dist_array_buffer_helper in dist_array_buffers[2:end]
        if isa(dist_array_buffer_helper, DistArrayBuffer)
            push!(buffer_info.helper_buffer_vec, dist_array_buffer_helper.id)
        else
            push!(buffer_info.helper_dist_array_vec, dist_array_buffer_helper.id)
        end
    end
    dist_array_buffer_map[dist_array.id] = buffer_info
end

function reset_write_buffer(dist_array::DistArray)
    delete!(dist_array_buffer_map, dist_array.id)
end
