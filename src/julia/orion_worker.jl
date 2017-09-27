module OrionWorker

include("dist_array.jl")
include("constants.jl")

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
end

function from_int64_to_keys(key::Int64, dims::Vector{Int64})
    dim_keys = []
    for dim in dims
        key_this_dim = key % dim + 1
        push!(dim_keys, key_this_dim)
        key = fld(key, dim)
    end
    return dim_keys
end

function from_keys_to_int64(key::Tuple, dims::Vector{Int64})
    key_int = 0
    for i = length(dims):-1:2
        key_int += key[i] - 1
        key_int *= dims[i - 1]
    end
    key_int += key[1] - 1
    return key_int
end

# DistArray accesses
function dist_array_get_dims(partition_ptr::Ptr{Void})
    dim_array = ccall((:orion_dist_array_get_dims, lib_path),
                      Any, (Ptr{Void},), partition_ptr)
    @assert isa(dim_array, Vector{Int64})
    return dim_array
end

function dist_array_get_value_type(partition_ptr::Ptr{Void})::DataType
    value_type_int32 = ccall((:orion_dist_array_get_value_type, lib_path),
                             Int32, (Ptr{Void},), partition_ptr)
    return int32_to_data_type(value_type_int32)
end

# TODO: currently doesn't allow any other Array except vectors to appear
# in the index; doesn't allow boolean vectors
function dist_array_read(partition_ptr::Ptr{Void},
                         index::Tuple)::Array
    num_dist_array_dims = length(index)
    result_dims = Vector{Int64}()
    dims = dist_array_get_dims(partition_ptr)
    println("dims = ", dims)
    ValueType = dist_array_get_value_type(partition_ptr)
    println(ValueType)
    if length(index) == 1
        value_array = Array{ValueType}(1)
        ccall((:orion_dist_array_read, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
              partition_ptr, index[1], 1, value_array)
        return value_array
    end

    @assert length(index) == length(dims)
    for i in eachindex(index)
        dim_i = index[i]
        if isa(dim_i, Integer)
        elseif isa(dim_i, Vector)
            push!(result_dims, length(dim_i))
        elseif dim_i == :(:)
            push!(result_dims, dims[i])
        elseif isa(dim_i, UnitRange)
            push!(result_dims, max(0, dim_i[end] - dim_i[1] + 1))
        else
            @assert false
        end
    end

    println("result dims = ", result_dims)
    array_size = reduce((x, y) -> x * y, 1, result_dims)
    result_array = Vector{ValueType}(array_size)

    per_read_size = 1
    num_dims_per_read = 0
    for i in eachindex(index)
        dim_i = index[i]
        if dim_i == :(:)
            per_read_size *= dims[i]
        else
            break
        end
        num_dims_per_read += 1
    end

    println("per_read_size = ", per_read_size)
    println("num_dims_per_read = ", num_dims_per_read)

    read_array = Vector{ValueType}(per_read_size)
    read_dims_begin = num_dims_per_read + 1
    read_offset = 1

    num_reads_by_dim = Vector{Int64}()
    for i = read_dims_begin:length(dims)
        dim_i = index[i]
        if dim_i == :(:)
            push!(num_reads_by_dim, dims[i])
        elseif isa(dim_i, Vector)
            push!(num_reads_by_dim, length(dim_i))
        elseif isa(dim_i, Integer)
            push!(num_reads_by_dim, 1)
        elseif isa(dim_i, UnitRange)
            push!(num_reads_by_dim, max(0, dim_i[end] - dim_i[1] + 1))
        else
            @assert false
        end
    end

    num_reads = fld(array_size, per_read_size)
    println("num_reads_by_dim = ", num_reads_by_dim)
    key_begin = Vector{Int64}(length(dims))
    fill!(key_begin, 1)
    key_begin_index = Vector{Int64}(length(dims))
    fill!(key_begin_index, 1)
    println("num_reads = ", num_reads)
    for i = 1:num_reads
        for j = read_dims_begin:length(dims)
            index_this_dim = key_begin_index[j]
            println("j = ", j, " index_this_dim = ", index_this_dim)
            if index[j] == :(:)
                key_begin[j] = index_this_dim
            elseif isa(index[j], Vector)
                key_begin[j] = index[j][index_this_dim]
            elseif isa(index[j], Integer)
                key_begin[j] = index[j]
            elseif isa(index[j], UnitRange)
                key_begin[j] = index[j][index_this_dim]
            else
                @assert false
            end
        end
        println("key_begin = ", key_begin)
        key_begin_int64 = from_keys_to_int64(tuple(key_begin...), dims)
        ccall((:orion_dist_array_read, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
              partition_ptr, key_begin_int64, per_read_size, read_array)

        result_array[read_offset:(read_offset + per_read_size - 1)] = read_array
        read_offset += per_read_size
        if read_dims_begin <= length(dims)
            key_begin_index[read_dims_begin] += 1
            j = read_dims_begin
            while j < length(dims)
                if key_begin_index[j] == (num_reads_by_dim[j - num_dims_per_read] + 1)
                    key_begin_index[j] = 1
                    key_begin_index[j + 1] += 1
                else
                    break
                end
                j += 1
            end
        end
    end
    result_array = reshape(result_array, tuple(result_dims...))
    return result_array
end

function dist_array_write(partition_ptr::Ptr{Void},
                          index::Tuple,
                          values::Array)
    num_dist_array_dims = length(index)
    array_size = length(values)
    write_dims = Vector{Int64}()
    dims = dist_array_get_dims(partition_ptr)
    ValueType = dist_array_get_value_type(partition_ptr)
    value_array = reshape(values, (array_size,))

    if length(index) == 1
        value_array = Array{ValueType}(1)
        ccall((:orion_dist_array_write, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
              partition_ptr, index[1], 1, values)
    end

    @assert length(index) == length(dims)
    for i in eachindex(index)
        dim_i = index[i]
        if isa(dim_i, Integer)
        elseif isa(dim_i, Vector)
            push!(result_dims, length(dim_i))
        elseif dim_i == :(:)
            push!(result_dims, dims[i])
        elseif isa(dim_i, UnitRange)
            push!(result_dims, max(0, dim_i[end] - dim_i[1] + 1))
        else
            @assert false
        end
    end

    per_write_size = 1
    num_dims_per_write = 0
    for i in eachindex(index)
        dim_i = index[i]
        if dim_i != :(:)
            break
        end
        per_write_size *= dims[i]
        num_dims_per_write += 1
    end

    write_dims_begin = num_dims_per_write + 1
    write_offset = 1

    num_writes_by_dim = Vector{Int64}()
    for i = read_dims_begin:length(dims)
        dim_i = index[i]
        if dim_i == :(:)
            push!(num_writes_by_dim, dims[i])
        elseif isa(dim_i, Vector)
            push!(num_writes_by_dim, length(dim_i))
        elseif isa(dim_i, Integer)
            push!(num_writes_by_dim, 1)
        elseif isa(dim_i, UnitRange)
            push!(num_writes_by_dim, max(0, dim_i[end] - dim_i[1] + 1))
        else
            @assert false
        end
    end

    num_writes = fld(array_size, per_write_size)

    key_begin = Vector{Int64}(length(dims))
    fill!(key_begin, 1)
    key_begin_index = Vector{Int64}(length(dims))
    fill!(key_begin_index, 1)
    for i = 1:num_writes
        for j = write_dims_begin:length(dims)
            index_this_dim = key_begin_index[j]
            if index[j] == :(:)
                key_begin[j] = index_this_dim
            elseif isa(index[j], Vector)
                key_begin[j] = index[j][index_this_dim]
            elseif isa(index[j], Integer)
                key_begin[j] = index[j]
            elseif isa(index[j], UnitRange)
                key_begin[j] = index[j][index_this_dim]
            else
                @assert false
            end
        end
        write_array = value_array[write_offset:(write_offset + per_write_size - 1)]
        write_offset += per_write_size
        key_begin_int64 = from_keys_to_int64(tuple(key_begin...), dims)
        ccall((:orion_dist_array_write, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
              partition_ptr, key_begin_int64, per_write_size, write_array)
        if write_dims_begin <= length(dims)
            key_begin_index[write_dims_begin] += 1
            j = write_dims_begin
            while j < length(dims)
                if key_begin_index[j] == (num_reads_by_dim[j - num_dims_per_read] + 1)
                    key_begin_index[j] = 1
                    key_begin_index[j + 1] += 1
                else
                    break
                end
                j += 1
            end
        end
    end
end

end
