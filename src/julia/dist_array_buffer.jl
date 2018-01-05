type DistArrayBuffer{T} <: AbstractDistArray{T}
    id::Int32
    dims::Vector{Int64}
    is_dense::Bool
    ValueType::DataType
    init_value
    symbol
    access_ptr
    is_materialized::Bool
    DistArrayBuffer(id::Integer,
                    dims::Vector{Int64},
                    is_dense::Bool,
                    init_value::T) = new(
                        id,
                        dims,
                        is_dense,
                        T,
                        init_value,
                        nothing,
                        nothing,
                        false)
    DistArrayBuffer() = new(-1,
                            zeros(Int64, 0),
                            false,
                            T,
                            nothing,
                            nothing,
                            nothing,
                            false)
end

function dist_array_set_symbol(dist_array::AbstractDistArray,
                               symbol)
    buff = IOBuffer()
    serialize(buff, dist_array.ValueType)
    buff_array = takebuf_array(buff)
    dist_array.symbol = symbol
end

function materialize(dist_array_buffer::DistArrayBuffer)
    if dist_array_buffer.is_materialized
        return
    end
    buff = IOBuffer()
    serialize(buff, dist_array_buffer.ValueType)
    buff_array = takebuf_array(buff)
    ccall((:orion_create_dist_array_buffer, lib_path),
          Void, (Int32, Ref{Int64}, UInt64, Bool, Int32,
                 Any, Cstring, Ref{UInt8}, UInt64),
          dist_array_buffer.id,
          dist_array_buffer.dims,
          length(dist_array_buffer.dims),
          dist_array_buffer.is_dense,
          data_type_to_int32(dist_array_buffer.ValueType),
          dist_array_buffer.init_value,
          dist_array_buffer.symbol,
          buff_array,
          length(buff_array))
    dist_array_buffer.is_materialized = true
end

function create_dist_array_buffer(dims::Vector{Int64}, init_value,
                                  is_dense::Bool = false)
    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1
    ValueType = typeof(init_value)
    dist_array_buffer = DistArrayBuffer{ValueType}(id,
                                                   dims,
                                                   is_dense,
                                                   init_value)
end

function create_dist_array_for_access(ValueType::DataType,
                                      symbol::AbstractString,
                                      dims::Vector{Int64},
                                      is_dense::Bool,
                                      access_ptr,
                                      is_buffer::Bool)::AbstractDistArray
    local dist_array
    if is_buffer
        dist_array = DistArrayBuffer{ValueType}()
    else
        dist_array = DistArray{ValueType}()
    end
    dist_array.symbol = symbol
    dist_array.dims = copy(dims)
    dist_array.is_dense = is_dense
    dist_array.access_ptr = access_ptr
    return dist_array
end

function setindex!(dist_array::AbstractDistArray, value, index::Integer)
    value_array = Vector{ValueType}(1)
    value_array[1] = value
    access_ptr = dist_array.access_ptr
    ccall((:orion_dist_array_write, lib_path),
          Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
          access_ptr, index[1], 1, value_array)
end

function setindex!(dist_array::AbstractDistArray,
                   values,
                   index...)
    array_size = length(values)
    value_array = reshape(values, (array_size,))
    access_ptr = dist_array.access_ptr

    access_ptr = dist_array.access_ptr
    dims = dist_array.dims
    (keys_vec, per_write_size, _) = subscript_to_index(dims,
                                                       index...)

    write_offset = 1
    for key_begin in keys_vec
        write_array = value_array[write_offset:(write_offset + per_write_size - 1)]
        write_offset += per_write_size
        ccall((:orion_dist_array_write, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Any),
              access_ptr, key_begin, per_write_size, write_array)
    end
end

function subscript_to_index(dist_array_dims::Vector{Int64},
                            index...)
    num_dist_array_dims = length(dist_array_dims)
    @assert num_dist_array_dims == length(index)
    keys_vec = Vector{Int64}()

    result_dims = Vector{Int64}()
    for i in eachindex(index)
        dim_i = index[i]
        if isa(dim_i, Integer)
        elseif isa(dim_i, Vector)
            if length(dim_i) == 0
                push!(result_dims, length(dim_i))
            else
                if isa(dim_i[i], Bool)
                    num_elements = Base.reduce((x, y) -> x += (y ? 1 : 0), a)
                    push!(result_dims, num_elements)
                else
                    push!(result_dims, length(dim_i))
                end
            end
        elseif dim_i == Colon()
            push!(result_dims, dist_array_dims[i])
        elseif isa(dim_i, UnitRange)
            push!(result_dims, max(0, dim_i[end] - dim_i[1] + 1))
        else
            error("not support this subscripts")
        end
    end

    per_access_size = 1
    num_dims_per_access = 0
    # column major!!
    for i in eachindex(index)
        dim_i = index[i]
        if dim_i == Colon()
            per_access_size *= dist_array_dims[i]
        else
            break
        end
        num_dims_per_access += 1
    end

    access_dims_begin = num_dims_per_access + 1
    num_accesses_by_dim = result_dims[access_dims_begin:end]
    result_array_size = reduce((x, y) -> x * y, 1, result_dims)
    num_accesses = fld(result_array_size, per_access_size)
    if result_array_size == 0
        return (keys_vec, per_access_size)
    end

    key_begin = Vector{Int64}(length(dist_array_dims))
    fill!(key_begin, 1)
    key_begin_index = Vector{Int64}(length(dist_array_dims))
    fill!(key_begin_index, 1)

    for i = 1:num_accesses
        for j = access_dims_begin:length(dist_array_dims)
            index_this_dim = key_begin_index[j]
            if index[j] == Colon()
                key_begin[j] = index_this_dim
            elseif isa(index[j], Vector)
                if isa(index[j][1], Integer)
                    key_begin[j] = index[j][index_this_dim]
                else
                    @assert isa(index[j][1], Bool)
                    while !index[j][index_this_dim]
                        index_this_dim += 1
                        @assert index_this_dim <= length(index[j])
                    end
                    key_begin[j] = index_this_dim
                    key_begin_index[j] = index_this_dim
                end
            elseif isa(index[j], Integer)
                key_begin[j] = index[j]
            elseif isa(index[j], UnitRange)
                key_begin[j] = index[j][index_this_dim]
            else
                @assert false
            end
            key_begin_int64 = from_keys_to_int64(key_begin, dist_array_dims)
            push!(keys_vec, key_begin_int64)
            key_begin_index[access_dims_begin] += 1

            j = access_dims_begin
            while j < length(dist_array_dims)
                if (isa(index[j], Vector) && isa(index[j][1], Bool) &&
                    key_begin_index[j] == length(index[j])) ||
                    key_begin_index[j] == (num_accesses_by_dim[j - num_dims_per_access] + 1)
                    key_begin_index[j] = 1
                    key_begin_index[j + 1] += 1
                else
                    break
                end
                j += 1
            end
        end
    end
    return (keys_vec, per_access_size, result_dims)
end

function getindex(dist_array::DistArray, index::Integer)
    access_ptr = dist_array.access_ptr
    if dist_array.is_dense
        value_array = Vector{ValueType}(1)
        ccall((:orion_dist_array_read_dense, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Any),
              access_ptr, index, 1, value_array)
        return value_array[1]
    else
        key_array = nothing
        value_array = nothing
        ccall((:orion_dist_array_read_sparse, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Ref{Any}, Ref{Any}),
              access_ptr, index, 1, key_array, value_array)
        kv_dict = Dict{Int64, dist_array.ValueType}()

        if isa(value_array, Vector) &&
            length(value_array) == 1
            kv_dict[key_array[1]] = value_array[1]
        end
    end
end

function getindex(dist_array::DistArray,
                  index...)
    ValueType = dist_array.ValueType
    access_ptr = dist_array.access_ptr
    (keys_vec, per_read_size, result_dims) = subscript_to_index(dist_array.dims,
                                                                index...)

    result_array_size = reduce((x, y) -> x * y, 1, result_dims)

    if result_array_size == 1
        return getindex(dist_array, keys_vec[1])
    end

    if dist_array.is_dense
        read_array = Vector{ValueType}(per_read_size)
        result_array = Vector{ValueType}(result_array_size)
        read_offset = 1
        for key_begin in keys_vec
            ccall((:orion_dist_array_read_dense, lib_path),
                  Void, (Ptr{Void}, Int64, UInt64, Any),
                  access_ptr, key_begin, per_read_size, read_array)
            result_array[read_offset:(read_offset + per_read_size - 1)] = read_array
            read_offset += per_read_size
        end
        result_array = reshape(result_array, tuple(result_dims...))
        return result_array
    else
        key_array = nothing
        value_array = nothing
        result_dict = Dict{Int64, ValueType}()
        for key_begin in keys_vec
            key_array = nothing
            value_array = nothing
            ccall((:orion_dist_array_read_sparse, lib_path),
                  Void, (Ptr{Void}, Int64, UInt64, Ref{Any}, Ref{Any}),
                  access_ptr, key_begin, per_read_size, key_array, value_array)
            if isa(value_array, Vector)
                for idx in eachindex(key_array)
                    result_dict[key_array[idx]] = value_array[idx]
                end
            end
            read_offset += per_read_size
        end
        return result_dict
    end
end

function getindex(dist_array::DistArrayBuffer, index::Integer)
    access_ptr = dist_array.access_ptr
    value_array = Vector{ValueType}(1)
    if dist_array.is_dense
        ccall((:orion_dist_array_read_dense, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Any),
              access_ptr, index, 1, value_array)
    else
        ccall((:orion_dist_array_read_sparse_with_init_value, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Any),
              access_ptr, index, 1, value_array)
    end
    return value_array[1]
end

function getindex(dist_array::DistArrayBuffer,
                  index...)
    ValueType = dist_array.ValueType
    access_ptr = dist_array.access_ptr
    (keys_vec, per_read_size, result_dims) = subscript_to_index(dist_array.dims,
                                                                index...)


    result_array_size = reduce((x, y) -> x * y, 1, result_dims)

    if result_array_size == 1
        return getindex(dist_array, keys_vec[1])
    end
    read_array = Vector{ValueType}(per_read_size)
    result_array = Vector{ValueType}(result_array_size)

    read_offset = 1
    if dist_array.is_dense
        for key_begin in keys_vec
            ccall((:orion_dist_array_read_dense, lib_path),
                  Void, (Ptr{Void}, Int64, UInt64, Any),
                  access_ptr, key_begin, per_read_size, read_array)
            result_array[read_offset:(read_offset + per_read_size - 1)] = read_array
            read_offset += per_read_size
        end
    else
        for key_begin in keys_vec
            ccall((:orion_dist_array_read_sparse_with_init_value, lib_path),
                  Void, (Ptr{Void}, Int64, UInt64, Any),
                  access_ptr, key_begin, per_read_size, read_array)
            result_array[read_offset:(read_offset + per_read_size - 1)] = read_array
            read_offset += per_read_size
        end
    end

    result_array = reshape(result_array, tuple(result_dims...))
    return result_array
end
