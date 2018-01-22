module OrionWorker

include("src/julia/dist_array_accessor.jl")
include("src/julia/dist_array.jl")
include("src/julia/constants.jl")
include("src/julia/dist_array_buffer.jl")

function worker_init(_num_executors::Integer,
                     _num_servers::Integer)
    global const num_executors = _num_executors
    global const num_servers = _num_servers
    load_constants()
end

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
end

end

using OrionWorker

function orionres_define_dist_array(id::Int32,
                                    ValueType::DataType,
                                    symbol::AbstractString,
                                    dims::Vector{Int64},
                                    is_dense::Bool,
                                    is_buffer::Bool,
                                    init_value)
    dist_array = OrionWorker.create_dist_array_on_worker(id,
                                                         ValueType,
                                                         symbol,
                                                         dims,
                                                         is_dense,
                                                         is_buffer,
                                                         init_value)
    dist_array_symbol = Symbol(symbol)
    eval(:(global $dist_array_symbol = $dist_array))
end


function orionres_set_dist_array_dims(dist_array::OrionWorker.AbstractDistArray,
                                      dims::Vector{Int64})
    dist_array.dims = copy(dims)
end

function orionres_get_dist_array_value_type{T, N}(dist_array::OrionWorker.AbstractDistArray{T, N})::DataType
    return T
end
