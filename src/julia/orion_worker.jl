module OrionWorker

include("src/julia/dist_array.jl")
include("src/julia/constants.jl")
include("src/julia/dist_array_buffer.jl")

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
end

end

using OrionWorker

function orionres_define_dist_array(ValueType::DataType,
                                    symbol::AbstractString,
                                    dims::Vector{Int64},
                                    is_dense::Bool,
                                    access_ptr,
                                    is_buffer::Bool)
    dist_array = OrionWorker.create_dist_array_for_access(ValueType,
                                                          symbol,
                                                          dims,
                                                          is_dense,
                                                          access_ptr,
                                                          is_buffer)
    dist_array_symbol = Symbol(symbol)
    eval(:(global $dist_array_symbol = $dist_array))
end


function orionres_set_dist_array_dims(dist_array::DistArray,
                                      dims::Vector{Int64})
    dist_array.dims = copy(dims)
end

function orionres_get_dist_array_value_type(dist_array::DistArray)::DataType
    return dist_array.ValueType
end

function orionres_dist_array_create_and_append_partition(dist_array::DistArray)::Vector
    partition = Vector{dist_array.ValueType}()
    push!(dist_array.partitions, partition)
    return partition
end

function orionres_dist_array_delete_partitions(dist_array::DistArray,
                                               partitions::Vector{Any})
    index_vec = Vector{Int64}()
    for partition_idx in eachindex(dist_array.partitions)
        partition = dist_array.partitions[partition_idx]
        if partition in partitions
            push!(index_vec, partition_idx)
        end
    end
    deleteat!(dist_array.partitions, index_vec)
end

function orionres_dist_array_clear_partition(partition::Vector)
    resize!(partition, 0)
end
