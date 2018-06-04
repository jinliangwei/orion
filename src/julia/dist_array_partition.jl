function dist_array_get_partition{T, N}(dist_array::AbstractDistArray{T, N},
                                        ptr_str::String)::Vector{T}
    return dist_array.partitions[ptr_str]
end

function dist_array_create_and_add_partition{T, N}(
    dist_array::AbstractDistArray{T, N},
    ptr_str::String)
    dist_array.partitions[ptr_str] = Vector{T}()
    dist_array.sparse_indexed_partitions[ptr_str] = Dict{Int64, T}()
end

function dist_array_delete_partition(dist_array::AbstractDistArray,
                                     ptr_str::String)
    delete!(dist_array.partitions, ptr_str)
    delete!(dist_array.sparse_indexed_partitions, ptr_str)
end

function dist_array_clear_partition{T, N}(dist_array::AbstractDistArray{T, N},
                                          ptr_str::String)
    dist_array.partitions[ptr_str] = Vector{T}()
    dist_array.sparse_indexed_partitions[ptr_str] = Dict{Int64, T}()
end

function dist_array_set_partition{T, N}(dist_array::AbstractDistArray{T, N},
                                        ptr_str::String,
                                        partition::Vector{T})
    dist_array.partitions[ptr_str] = partition
end

function dist_array_serialize_partition{T, N}(dist_array::AbstractDistArray{T, N},
                                              ptr_str::String)::Vector{UInt8}
    buffer = IOBuffer()
    serialize(buffer, dist_array.partitions[ptr_str])
    return take!(buffer)
end

function dist_array_modulo_serialize_partition{T, N}(dist_array::AbstractDistArray{T, N},
                                                   ptr_str::String,
                                                   key_vec::Vector{Int64},
                                                   num_dests::UInt64)::Tuple{Vector{Vector{Int64}},
                                                                             Vector{Vector{UInt8}}
                                                                            }
    values = dist_array.partitions[ptr_str]
    @assert length(key_vec) == length(values)
    key_vec_by_dest = [Vector{Int64}() for i = 1:num_dests]
    values_by_dest = [Vector{T}() for i = 1:num_dests]
    serialized_values_by_dest = Vector{Vector{UInt8}}(num_dests)
    for idx in eachindex(key_vec)
        key = key_vec[idx]
        value = values[idx]
        dest_idx = key % num_dests + 1
        push!(key_vec_by_dest[dest_idx], key)
        push!(values_by_dest[dest_idx], value)
    end
    for dest_idx in eachindex(values_by_dest)
        buffer = IOBuffer()
        serialize(buffer, dist_array.partitions[ptr_str])
        serialized_value = take!(buffer)
        serialized_values_by_dest[dest_idx] = serialized_value
    end
    return (key_vec_by_dest, serialized_values_by_dest)
end

function dist_array_deserialize_partition{T, N}(dist_array::AbstractDistArray{T, N},
                                                ptr_str::String,
                                                serialized_bytes::Vector{UInt8})
    buffer = IOBuffer(serialized_bytes)
    partition = deserialize(buffer)
    dist_array.partitions[ptr_str] = partition
end

function dist_array_deserialize_and_append_partition{T, N}(dist_array::AbstractDistArray{T, N},
                                                           ptr_str::String,
                                                           serialized_bytes::Vector{UInt8})
    buffer = IOBuffer(serialized_bytes)
    partition = deserialize(buffer)
    append!(dist_array.partitions[ptr_str], partition)
end

function dist_array_deserialize_and_overwrite_partition{T, N}(dist_array::AbstractDistArray{T, N},
                                                              ptr_str::String,
                                                              key_vec::Vector{Int64},
                                                              serialized_bytes::Vector{UInt8})
    buffer = IOBuffer(serialized_bytes)
    values = deserialize(buffer)
    dist_array_partition_set_values(dist_array, ptr_str, key_vec, values)
end

function dist_array_partition_get_value_array_by_keys{T, N}(dist_array::AbstractDistArray{T, N},
                                                            ptr_str::String,
                                                            key_vec::Vector{Int64})::Vector{T}
    sparse_indexed_partition = dist_array.sparse_indexed_partitions[ptr_str]
    value_vec = Vector{T}(length(key_vec))
    for idx in eachindex(key_vec)
        key = key_vec[idx]
        value = sparse_indexed_partition[key]
        value_vec[idx] = value
    end
    return value_vec
end

function dist_array_partition_set_values{T, N}(dist_array::AbstractDistArray{T, N},
                                               ptr_str::String,
                                               key_vec::Vector{Int64},
                                               values::Vector{T})
    sparse_indexed_partition = dist_array.sparse_indexed_partitions[ptr_str]
    @assert length(values) == length(key_vec)
    for idx in eachindex(key_vec)
        key = key_vec[idx]
        value = values[idx]
        sparse_indexed_partition[key] = value
    end
end

function dist_array_partition_get_and_serialize_value{T, N}(dist_array::AbstractDistArray{T, N},
                                                            ptr_str::String,
                                                            key::Int64)::Vector{UInt8}
    sparse_indexed_partition = dist_array.sparse_indexed_partitions[ptr_str]
    value = sparse_indexed_partition[key]
    buffer = IOBuffer()
    serialize(buffer, value)
    return take!(buffer)
end

function dist_array_partition_get_and_serialize_values{T, N}(
    dist_array::AbstractDistArray{T, N},
    ptr_str::String,
    key_vec::Vector{Int64})::Vector{UInt8}

    value_vec = dist_array_partition_get_value_array_by_keys(
        dist_array, ptr_str, key_vec)
    buffer = IOBuffer()
    serialize(buffer, value_vec)
    return take!(buffer)
end

function dist_array_partition_dense_to_sparse{T, N}(
    dist_array::AbstractDistArray{T, N},
    ptr_str::String,
    key_vec::Vector{Int64})
    values = dist_array.partitions[ptr_str]
    dist_array.sparse_indexed_partitions[ptr_str] = Dict(zip(key_vec, values))
end

function dist_array_partition_sparse_to_dense{T, N}(
    dist_array::AbstractDistArray{T, N},
    ptr_str::String)::Vector{Int64}
    partition = dist_array.sparse_indexed_partitions[ptr_str]

    key_vec = sort(collect(keys(partition)))
    values = Vector{T}(length(key_vec))
    for idx in eachindex(key_vec)
        key = key_vec[idx]
        value = partition[key]
        values[idx] = value
    end
    dist_array.partitions[ptr_str] = values
    return key_vec
end

function dist_array_shrink_partition_to_fit{T, N}(dist_array::AbstractDistArray{T, N},
                                                  ptr_str::String)
    partition = dist_array.partitions[ptr_str]
    new_partition = Vector{T}(length(partition))
    new_partition .= partition
    delete!(dist_array.partitions, ptr_str)
    dist_array.partitions[ptr_str] = new_partition
end
