import Base: size, getindex, setindex!
abstract DistArrayBuffer{T, N} <: AbstractDistArray{T, N}

type DenseDistArrayBuffer{T, N} <: DistArrayBuffer{T, N}
    id::Int32
    dims::Vector{Int64}
    init_value::T
    symbol::Nullable{Symbol}
    partitions::Vector{DistArrayPartition{T}}
    accessor::Nullable{DenseDistArrayAccessor{T, N}}
    is_materialized::Bool
    DenseDistArrayBuffer(id::Integer,
                         dims::Vector{Int64},
                         init_value::T) = new(
                             id,
                             copy(dims),
                             init_value,
                             Nullable{Symbol}(),
                             Vector{DistArrayPartition{T}}(),
                             Nullable{DenseDistArrayAccessor{T, N}}(),
                             false)
end

type SparseDistArrayBuffer{T, N} <: DistArrayBuffer{T, N}
    id::Int32
    dims::Vector{Int64}
    init_value::T
    symbol::Nullable{Symbol}
    partitions::Vector{DistArrayPartition{T}}
    accessor::Nullable{SparseInitDistArrayAccessor{T, N}}
    is_materialized::Bool
    SparseDistArrayBuffer(id::Integer,
                          dims::Vector{Int64},
                          init_value::T) = new(
                              id,
                              copy(dims),
                              init_value,
                              Nullable{Symbol}(),
                              Vector{DistArrayPartition{T}}(),
                              Nullable{SparseInitDistArrayAccessor{T, N}}(),
                              false)
end

function Base.getindex(dist_array::DistArrayBuffer,
                       I...)
    @assert !isnull(dist_array.accessor)
    accessor = get(dist_array.accessor)
    return getindex(accessor, I...)
end

function Base.setindex!(dist_array::DistArrayBuffer,
                        v, I...)
    @assert !isnull(dist_array.accessor)
    accessor = get(dist_array.accessor)
    setindex!(accessor, v, I...)
end

function materialize{T, N}(dist_array_buffer::DistArrayBuffer{T, N})
    if dist_array_buffer.is_materialized
        return
    end
    buff = IOBuffer()
    serialize(buff, T)
    buff_array = takebuf_array(buff)
    ccall((:orion_create_dist_array_buffer, lib_path),
          Void, (Int32,
                 Ref{Int64},
                 UInt64,
                 Bool,
                 Int32,
                 Any,
                 Cstring,
                 Ref{UInt8},
                 UInt64),
          dist_array_buffer.id,
          dist_array_buffer.dims,
          N,
          isa(dist_array_buffer, DenseDistArrayBuffer),
          data_type_to_int32(T),
          dist_array_buffer.init_value,
          string(get(dist_array_buffer.symbol)),
          buff_array,
          length(buff_array))
    dist_array_buffer.is_materialized = true
end

function create_dense_dist_array_buffer{T, N}(dims::NTuple{N, Int64},
                                              init_value::T)
    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1
    dist_array_buffer = DenseDistArrayBuffer{T, N}(id,
                                                   [dims...],
                                                   init_value)
end

function create_sparse_dist_array_buffer{T, N}(dims::NTuple{N, Int64},
                                              init_value::T)
    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1
    dist_array_buffer = SparseDistArrayBuffer{T, N}(id,
                                                   [dims...],
                                                   init_value)
end

function Base.size(dist_array_buffer::DistArrayBuffer)
    return tuple(dist_array_buffer.dims...)
end

function create_dist_array_buffer_accessor{T, N}(dist_array_buffer::DenseDistArrayBuffer{T, N})
    num_values = reduce(*, dist_array_buffer.dims)
    values = Vector{T}(num_values)
    for i = 1:num_values
        values[i] = dist_array_buffer.init_value
    end
    dist_array_buffer.accessor = Nullable{DenseDistArrayAccessor{T, N}}(
        DenseDistArrayAccessor{T, N}(0, values,
                                  [dist_array_buffer.dims...]))
end

function create_dist_array_buffer_accessor{T, N}(dist_array_buffer::SparseDistArrayBuffer{T, N})
    dist_array_buffer.accessor = Nullable{SparseInitDistArrayAccessor{T, N}}(
        SparseInitDistArrayAccessor{T, N}(dist_array_buffer.init_value,
                                          [dist_array_buffer.dims...]))

end

function delete_dist_array_accessor{T, N}(dist_array_buffer::DistArrayBuffer{T, N})
    dist_array_buffer.accessor = Nullable{DistArrayAccessor{T, N}}()
end

function dist_array_get_accessor_keys_values_vec{T, N}(dist_array_buffer::SparseDistArrayBuffer{T, N})::Tuple{Vector{Int64},
                                                                                                              Vector{T}}
    dist_array_accessor_get_keys_values_vec(get(dist_array_buffer.accessor))
end

function dist_array_get_accessor_values_vec{T, N}(dist_array_buffer::DenseDistArrayBuffer{T, N})::Vector{T}
    dist_array_accessor_get_values_vec(get(dist_array_buffer.accessor))
end

function create_dist_array_on_worker(id::Int32,
                                     ValueType::DataType,
                                     symbol::AbstractString,
                                     dims::Vector{Int64},
                                     is_dense::Bool,
                                     is_buffer::Bool,
                                     init_value)::AbstractDistArray

    local dist_array
    if is_buffer
        if is_dense
            dist_array = DenseDistArrayBuffer{ValueType, length(dims)}(id,
                                                                       dims,
                                                                       init_value)
        else
            dist_array = SparseDistArrayBuffer{ValueType, length(dims)}(id,
                                                                       dims,
                                                                       init_value)
        end
    else
        if is_dense
            dist_array = DenseDistArray{ValueType, length(dims)}(id,
                                                                 DistArrayParentType_init,
                                                                 DistArrayMapType_no_map)
        else
            dist_array = SparseDistArray{ValueType, length(dims)}(id,
                                                                  DistArrayParentType_init,
                                                                  DistArrayMapType_no_map)
        end
    end
    dist_array.symbol = symbol
    dist_array.dims = copy(dims)
    return dist_array
end
