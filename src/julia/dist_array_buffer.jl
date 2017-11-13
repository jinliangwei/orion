type DistArrayBuffer{T} <: AbstractDistArray{T}
    id::Int32
    dims::Vector{Int64}
    is_dense::Bool
    init_value
    symbol
    access_ptr
    DistArrayBuffer(id::Integer,
                    dims::Vector{Int64},
                    is_dense::Bool,
                    init_value::T) = new(
                        id,
                        dims,
                        is_dense,
                        init_value,
                        nothing,
                        nothing)
    DistArrayBuffer() = new(-1,
                            zeros(Int64, 0),
                            false,
                            nothing,
                            nothing,
                            nothing)
end

function dist_array_set_symbol(dist_array::AbstractDistArray,
                               symbol)
    dist_array.symbol = symbol
end

function materialize(dist_array_buffer::DistArrayBuffer)
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

function set_write_buffer(dist_array::DistArray,
                          apply_func::Function,
                          dist_array_buffers::DistArrayBuffer...)
end

function create_dist_array_for_access(ValueType::DataType,
                                      symbol::AbstractString,
                                      dims::Vector,
                                      is_dense::Bool,
                                      access_ptr,
                                      is_buffer::Bool)::AbstractDistArray
    if is_buffer
        dist_array = DistArrayBuffer{ValueType}()
    else
        dist_array = DistArray{ValueType}()
    end
    dist_array.symbol = symbol
    dist_array.dims = dims
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
    (keys_vec, per_write_size) = subscript_to_index(index...)

    write_offset = 1
    for key_begin in keys_vec
        write_array = value_array[write_offset:(write_offset + per_write_size - 1)]
        write_offset += per_write_size
        key_begin_int64 = from_keys_to_int64(key_begin, dims)
        ccall((:orion_dist_array_write, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
              access_ptr, key_begin_int64, per_write_size, write_array)
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
            push!(result_dims, dims[i])
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
            per_access_size *= dims[i]
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

    key_begin = Vector{Int64}(length(dims))
    fill!(key_begin, 1)
    key_begin_index = Vector{Int64}(length(dims))
    fill!(key_begin_index, 1)

    for i = 1:num_accesses
        for j = access_dims_begin:length(dims)
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
            key_begin_int64 = from_keys_to_int64(key_begin, dims)
            push!(keys_vec, key_begin_int64)
            key_begin_index[access_dims_begin] += 1

            j = access_dims_begin
            while j < length(dims)
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
    return (keys_vec, per_access_size)
end

function getindex(dist_array::AbstractDistArray, index::Integer)
    value_array = Vector{ValueType}(1)
    access_ptr = dist_array.access_ptr
    ccall((:orion_dist_array_read, lib_path),
          Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
          access_ptr, index, 1, value_array)
    return value_array[1]
end

function getindex(dist_array::AbstractDistArray,
                  index...)
    ValueType = dist_array.ValueType
    access_ptr = dist_array.access_ptr
    (keys_vec, per_read_size) = subscript_to_index(index...)

    read_array = Vector{ValueType}(per_read_size)

    result_array_size = reduce((x, y) -> x * y, 1, result_dims)
    result_array = Vector{ValueType}(result_array_size)

    read_offset = 1
    for key_begin in keys_vec
        ccall((:orion_dist_array_read, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
              access_ptr, key_begin, per_read_size, read_array)
        result_array[read_offset:(read_offset + per_read_size - 1)] = read_array
        read_offset += per_read_size
    end
    result_array = reshape(result_array, tuple(result_dims...))
    return result_array
end
