import Base: linearindexing, size, getindex, setindex!
abstract DistArrayAccessor{T, N} <: AbstractArray{T, N}

immutable DenseDistArrayAccessor{T, N} <: DistArrayAccessor{T, N}
    key_begin::Int64
    values::Vector{T}
    dims::NTuple{N, Int64}
    DenseDistArrayAccessor(key_begin::Int64,
                           values::Vector{T},
                           dims::Vector{Int64}) = new(key_begin,
                                                      values,
                                                      tuple(dims...))

    DenseDistArrayAccessor(dims::Vector{Int64}) = new(0,
                                                      Vector{T}(reduce(*, dims)),
                                                      tuple(dims...))
end

immutable SparseDistArrayAccessor{T, N} <: DistArrayAccessor{T, N}
    key_value::Dict{Int64, T}
    dims::NTuple{N, Int64}
    SparseDistArrayAccessor(keys::Vector{Int64},
                            values::Vector{T},
                            dims::Vector{Int64}) = new(Dict(zip(keys, values)),
                                                       tuple(dims...))

    SparseDistArrayAccessor(dims::Vector{Int64}) = new(Dict{Int64, T}(),
                                                       tuple(dims...))
end

immutable SparseInitDistArrayAccessor{T, N} <: DistArrayAccessor{T, N}
    init_value::T
    key_value::Dict{Int64, T}
    dims::NTuple{N, Int64}
    SparseInitDistArrayAccessor(init_value::T,
                                keys::Vector{Int64},
                                values::Vector{T},
                                dims::Vector{Int64}) = new(init_value,
                                                           Dict(zip(keys, values)),
                                                           tuple(dims...))

    SparseInitDistArrayAccessor(init_value::T,
                                dims::Vector{Int64}) = new(init_value,
                                                           Dict{Int64, T}(),
                                                           tuple(dims...))
end

immutable DistArrayCacheAccessor{T, N} <: DistArrayAccessor{T, N}
    dist_array_id::Int32
    key_value::Dict{Int64, T}
    dims::NTuple{N, Int64}
    DistArrayCacheAccessor(dist_array_id::Int32,
                           keys::Vector{Int64},
                           values::Vector{T},
                           dims::Vector{Int64}) = new(dist_array_id,
                                                      Dict(zip(keys, values)),
                                                      tuple(dims...))

    DistArrayCacheAccessor(dist_array_id::Int32,
                           dims::Vector{Int64}) = new(dist_array_id,
                                                      Dict{Int64, T}(),
                                                      tuple(dims...))
end

function dist_array_cache_fetch(dist_array_id::Int32,
                                key::Int64,
                                ValueType::DataType)
    local value
    ccall((:orion_request_dist_array_data, lib_path),
          Void, (Int32,
                 Int64,
                 Int32,
                 Ref{Any}),
          dist_array_id,
          key,
          data_type_to_int32(ValueType),
          value)
end

function Base.linearindexing{T<:DistArrayAccessor}(::Type{T})
    return Base.LinearFast()
end

function Base.size(accessor::DistArrayAccessor)
    return accessor.dims
end

function Base.getindex(accessor::DenseDistArrayAccessor,
                       i::Int)
    return accessor.values[i - accessor.key_begin]
end

function Base.setindex!(accessor::DenseDistArrayAccessor,
                        v, i::Int)
    accessor.values[i - accessor.key_begin] = v
end

function Base.similar{T, N}(accessor::DenseDistArrayAccessor{T, N},
                            ::Type{T}, dims::NTuple{N, Int64})
    return DenseDistArrayAccessor{T, N}(dims)
end

function getindex(accessor::SparseDistArrayAccessor,
                  i::Int64)
    return accessor.key_value[i]
end

function setindex!(accessor::SparseDistArrayAccessor,
                   v, i::Int64)
    accessor.key_value[i] = v
end

function Base.similar{T, N}(accessor::SparseDistArrayAccessor{T, N},
                            ::Type{T}, dims::NTuple{N, Int64})
    return SparseDistArrayAccessor{T, N}(dims)
end

function getindex(accessor::SparseInitDistArrayAccessor,
                  i::Int64)
    return get(accessor.key_value, i, accessor.init_value)
end

function setindex!(accessor::SparseInitDistArrayAccessor,
                   v, i::Int64)
    return accessor.key_value[i] = v
end

function Base.similar{T, N}(accessor::SparseInitDistArrayAccessor{T, N},
                            ::Type{T}, dims::NTuple{N, Int64})
    return SparseInitDistArrayAccessor{T, N}(accessor.init_value, dims)
end

function getindex{T, N}(accessor::DistArrayCacheAccessor{T, N},
                  i::Int64)
    if haskey(accessor.key_value, i)
        return accessor.key_value[i]
    else
        value = dist_array_cache_fetch(accessor.dist_array_id,
                                       key,
                                       T)
        @assert value != nothing
        key_value[i] = value
        return value
    end
end

function setindex!(accessor::DistArrayCacheAccessor,
                   v, i::Int64)
    accessor.key_value[i] = v
end

function Base.similar{T, N}(accessor::DistArrayCacheAccessor{T, N},
                            ::Type{T}, dims::NTuple{N, Int64})
    return DistArrayCacheAccessor{T, N}(accessor.dist_array_id, dims)
end

function dist_array_accessor_get_values_vec(dist_array_accessor::DenseDistArrayAccessor)::Vector
    return dist_array_accessor.values
end

function dist_array_accessor_get_keys_vec(dist_array_accessor::DistArrayAccessor)::Vector{Int64}
    return [k in keys(dist_array_accessor.key_value)]
end

function dist_array_accessor_get_values_vec(dist_array_accessor::DistArrayAccessor)::Vector
    return [v in values(dist_array_accessor.key_value)]
end
