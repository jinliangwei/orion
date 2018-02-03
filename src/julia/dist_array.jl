import Base
import Base: copy, ==
import Base: size, getindex, setindex!

@enum DistArrayParentType DistArrayParentType_text_file =
    1 DistArrayParentType_dist_array =
    2 DistArrayParentType_init =
    3

@enum DistArrayInitType DistArrayInitType_empty =
    1 DistArrayInitType_uniform_random =
    2 DistArrayInitType_normal_random =
    3

@enum DistArrayMapType DistArrayMapType_no_map =
    1 DistArrayMapType_map =
    2 DistArrayMapType_map_fixed_keys =
    3 DistArrayMapType_map_values =
    4 DistArrayMapType_map_values_new_keys =
    5

@enum DistArrayPartitionType DistArrayPartitionType_naive =
    1 DistArrayPartitionType_1d =
    2 DistArrayPartitionType_2d =
    3 DistArrayPartitionType_2d_unimodular =
    4 DistArrayPartitionType_range =
    5 DistArrayPartitionType_hash_server =
    6 DistArrayPartitionType_hash_executor =
    7

@enum DistArrayIndexType DistArrayIndexType_none =
    1 DistArrayIndexType_range =
    2

@enum ForLoopParallelScheme ForLoopParallelScheme_naive =
    1 ForLoopParallelScheme_1d =
    2 ForLoopParallelScheme_2d =
    3 ForLoopParallelScheme_unimodular =
    4 ForLoopParallelScheme_none =
    5

immutable DistArrayPartitionInfo
    partition_type::DistArrayPartitionType
    partition_func_name::Nullable{String}
    partition_dims::Nullable{Tuple}
    tile_sizes::Nullable{Tuple}
    index_type::DistArrayIndexType
    DistArrayPartitionInfo(partition_type::DistArrayPartitionType,
                           index_type::DistArrayIndexType) = new(
                               partition_type,
                               Nullable{String}(),
                               Nullable{Tuple}(),
                               Nullable{Tuple}(),
                               index_type)
    DistArrayPartitionInfo(partition_type::DistArrayPartitionType,
                           partition_func_name::String,
                           partition_dims::Tuple,
                           tile_sizes::Tuple,
                           index_type::DistArrayIndexType) = new(
    partition_type,
    partition_func_name,
    partition_dims,
    tile_sizes,
    index_type)
end

function is_partition_equal(partition_a::DistArrayPartitionInfo,
                            partition_b::DistArrayPartitionInfo)::Bool

return partition_a.partition_type == partition_b.partition_type &&
    (
        (
            isnull(partition_a.partition_dims) && isnull(partition_b.partition_dims)
        ) || (
            get(partition_a.partition_dims) == get(partition_b.partition_dims)
        )
    ) && (
        (
            isnull(partition_a.tile_sizes) && isnull(partition_b.tile_sizes)
        ) || (
            get(partition_a.tile_sizes) == get(partition_b.tile_sizes)
        )
    ) &&
        partition_a.index_type == partition_b.index_type
end

immutable DistArrayMapInfo
    parent_id::Nullable{Int32}
    flatten_results::Bool
    map_func_module::Module
    map_func_name::String

    DistArrayMapInfo(flatten_results::Bool,
                     map_func_module::Module,
                     map_func_name::String) = new(Nullable{Int32}(),
                                                  flatten_results,
                                                  map_func_module,
                                                  map_func_name)
    DistArrayMapInfo(parent_id::Int32,
                     flatten_results::Bool,
                     map_func_module::Module,
                     map_func_name::String) = new(parent_id,
                                                  flatten_results,
                                                  map_func_module,
                                                  map_func_name)
end

immutable DistArrayInitInfo
    init_type::DistArrayInitType
    random_init_type::DataType
end

immutable DistArrayPartition{T}
    values::Vector{T}
end

function dist_array_partition_get_values(partition::DistArrayPartition)::Vector
    return partition.values
end

function dist_array_partition_set_values(partition::DistArrayPartition,
                                         values::Vector)
    partition.values = values
end

abstract AbstractDistArray{T, N} <: AbstractArray{T, N}

abstract DistArray{T, N} <: AbstractDistArray{T, N}

type ValueOnlyDistArray{T} <: DistArray{T, 0}
    id::Int32
    symbol::Nullable{Symbol}
    partitions::Vector{DistArrayPartition{T}}
    is_materialized::Bool
    parent_type::DistArrayParentType
    file_path::Nullable{String}
    map_type::DistArrayMapType
    map_info::Nullable{DistArrayMapInfo}
    init_info::Nullable{DistArrayInitInfo}
    partition_info::DistArrayPartitionInfo
    target_partition_info::Nullable{DistArrayPartitionInfo}

    ValueOnlyDistArray(id::Integer,
                       parent_type::DistArrayParentType,
                       map_type::DistArrayMapType) = new(
                           id,
                           Nullable{Symbol}(),
                           Vector{Vector{T}}(),
                           false,
                           parent_type,
                           Nullable{String}(),
                           map_type,
                           Nullable{DistArrayMapInfo}(),
                           Nullable{DistArrayInitInfo}(),
                           DistArrayPartitionInfo(DistArrayPartitionType_naive,
                                                  DistArrayIndexType_none),
                           Nullable{DistArrayPartitionInfo}())
end

function copy{T}(dist_array::ValueOnlyDistArray{T})::ValueOnlyDistArray{T}
    new_dist_array = ValueOnlyDistArray{T, N}(
        dist_array.id,
        dist_array.parent_type,
        dist_array.map_type)
    new_dist_array.symbol = dist_array.symbol
    new_dist_array.partitions = dist_array.partitions
    new_dist_array.is_materialized = dist_array.is_materialized
    new_dist_array.file_path = dist_array.file_path
    new_dist_array.map_type = dist_array.map_type
    new_dist_array.map_info = dist_array.map_info
    new_dist_array.init_info = dist_array.init_info
    new_dist_array.partition_info = dist_array.partition_info
    new_dist_array.target_partition_info = dist_array.target_partition_info
    return new_dist_array
end

type DenseDistArray{T, N} <: DistArray{T, N}
    id::Int32
    dims::Vector{Int64}
    symbol::Nullable{Symbol}
    partitions::Vector{DistArrayPartition{T}}
    is_materialized::Bool
    parent_type::DistArrayParentType
    file_path::Nullable{String}
    map_type::DistArrayMapType
    map_info::Nullable{DistArrayMapInfo}
    init_info::Nullable{DistArrayInitInfo}
    partition_info::DistArrayPartitionInfo
    target_partition_info::Nullable{DistArrayPartitionInfo}
    accessor::Nullable{DenseDistArrayAccessor{T, N}}
    cache_accessor::Nullable{DistArrayCacheAccessor{T, N}}
    iterate_dims::Nullable{Vector{Int64}}
    num_partitions_per_dim::Int64

    DenseDistArray(id::Integer,
                   parent_type::DistArrayParentType,
                   map_type::DistArrayMapType) = new(
                       id,
                       Vector{Int64}(N),
                       Nullable{Symbol}(),
                       Vector{DistArrayPartition{T}}(),
                       false,
                       parent_type,
                       Nullable{String}(),
                       map_type,
                       Nullable{DistArrayMapInfo}(),
                       Nullable{DistArrayInitInfo}(),
                       DistArrayPartitionInfo(DistArrayPartitionType_naive,
                                              DistArrayIndexType_none),
                       Nullable{DistArrayPartitionInfo}(),
                       Nullable{DenseDistArrayAccessor{T, N}}(),
                       Nullable{DistArrayCacheAccessor{T, N}}(),
                       Nullable{Vector{Int64}}(),
                       num_executors * 2)

    DenseDistArray(id::Integer,
                   dims::Vector{Int64},
                   parent_type::DistArrayParentType,
                   map_type::DistArrayMapType) = new(
                       id,
                       copy(dims),
                       Nullable{Symbol}(),
                       Vector{DistArrayPartition{T}}(),
                       false,
                       parent_type,
                       Nullable{String}(),
                       map_type,
                       Nullable{DistArrayMapInfo}(),
                       Nullable{DistArrayInitInfo}(),
                       DistArrayPartitionInfo(DistArrayPartitionType_naive,
                                              DistArrayIndexType_none),
                       Nullable{DistArrayPartitionInfo}(),
                       Nullable{DenseDistArrayAccessor{T, N}}(),
                       Nullable{DistArrayCacheAccessor{T, N}}(),
                       Nullable{Vector{Int64}}(),
                       num_executors * 2)
end

function copy{T, N}(dist_array::DenseDistArray{T, N})::DenseDistArray{T, N}
    new_dist_array = DenseDistArray{T, N}(
        dist_array.id,
        dist_array.parent_type,
        dist_array.map_type)

    new_dist_array.dims = copy(dist_array.dims)
    new_dist_array.symbol = dist_array.symbol
    new_dist_array.partitions = dist_array.partitions
    new_dist_array.is_materialized = dist_array.is_materialized
    new_dist_array.file_path = dist_array.file_path
    new_dist_array.map_type = dist_array.map_type
    new_dist_array.map_info = dist_array.map_info
    new_dist_array.init_info = dist_array.init_info
    new_dist_array.partition_info = dist_array.partition_info
    new_dist_array.target_partition_info = dist_array.target_partition_info
    new_dist_array.accessor = dist_array.accessor
    new_dist_array.cache_accessor = dist_array.cache_accessor
    new_dist_array.iterate_dims = Nullable{Vector{Int64}}(copy(get(dist_array.iterate_dims)))
    new_dist_array.num_partitions_per_dim = dist_array.num_partitions_per_dim
    return new_dist_array
end

type SparseDistArray{T, N} <: DistArray{T, N}
    id::Int32
    dims::Vector{Int64}
    symbol::Nullable{Symbol}
    partitions::Vector{DistArrayPartition{T}}
    is_materialized::Bool
    parent_type::DistArrayParentType
    file_path::Nullable{String}
    map_type::DistArrayMapType
    map_info::Nullable{DistArrayMapInfo}
    init_info::Nullable{DistArrayInitInfo}
    partition_info::DistArrayPartitionInfo
    target_partition_info::Nullable{DistArrayPartitionInfo}
    accessor::Nullable{SparseDistArrayAccessor{T, N}}
    cache_accessor::Nullable{DistArrayCacheAccessor{T, N}}
    iterate_dims::Nullable{Vector{Int64}}
    num_partitions_per_dim::Int64

   SparseDistArray(id::Integer,
                   parent_type::DistArrayParentType,
                   map_type::DistArrayMapType) = new(
                       id,
                       Vector{Int64}(N),
                       Nullable{Symbol}(),
                       Vector{DistArrayPartition{T}}(),
                       false,
                       parent_type,
                       Nullable{String}(),
                       map_type,
                       Nullable{DistArrayMapInfo}(),
                       Nullable{DistArrayInitInfo}(),
                       DistArrayPartitionInfo(DistArrayPartitionType_naive,
                                              DistArrayIndexType_none),
                       Nullable{DistArrayPartitionInfo}(),
                       Nullable{SparseDistArrayAccessor{T, N}}(),
                       Nullable{DistArrayCacheAccessor{T, N}}(),
                       Nullable{Vector{Int64}}(),
                       num_executors * 2)

    SparseDistArray(id::Integer,
                    dims::Vector{Int64},
                    parent_type::DistArrayParentType,
                    map_type::DistArrayMapType) = new(
                        id,
                        copy(dims),
                        Nullable{Symbol}(),
                        Vector{DistArrayPartition{T}}(),
                        false,
                        parent_type,
                        Nullable{String}(),
                        map_type,
                        Nullable{DistArrayMapInfo}(),
                        Nullable{DistArrayInitInfo}(),
                        DistArrayPartitionInfo(DistArrayPartitionType_naive,
                                               DistArrayIndexType_none),
                        Nullable{DistArrayPartitionInfo}(),
                        Nullable{SparseDistArrayAccessor{T, N}}(),
                        Nullable{DistArrayCacheAccessor{T, N}}(),
                        Nullable{Vector{Int64}}(),
                        num_executors * 2)
end

function copy{T, N}(dist_array::SparseDistArray{T, N})::SparseDistArray{T, N}
    new_dist_array = SparseDistArray{T, N}(
        dist_array.id,
        dist_array.parent_type,
        dist_array.map_type)
    new_dist_array.dims = copy(dist_array.dims)
    new_dist_array.symbol = dist_array.symbol
    new_dist_array.partitions = dist_array.partitions
    new_dist_array.is_materialized = dist_array.is_materialized
    new_dist_array.file_path = dist_array.file_path
    new_dist_array.map_type = dist_array.map_type
    new_dist_array.map_info = dist_array.map_info
    new_dist_array.init_info = dist_array.init_info
    new_dist_array.partition_info = dist_array.partition_info
    new_dist_array.target_partition_info = dist_array.target_partition_info
    new_dist_array.accessor = dist_array.accessor
    new_dist_array.cache_accessor = dist_array.cache_accessor
    new_dist_array.iterate_dims = Nullable{Vector{Int64}}(copy(get(dist_array.iterate_dims)))
    new_dist_array.num_partitions_per_dim = dist_array.num_partitions_per_dim
    return new_dist_array
end

const dist_arrays = Dict{Int32, AbstractDistArray}()
dist_array_id_counter = 0

function dist_array_get_num_dims{T, N}(dist_array::AbstractDistArray{T, N})
    return N
end

function dist_array_get_value_type{T, N}(dist_array::AbstractDistArray{T, N})
    return T
end

function dist_array_set_num_partitions_per_dim(dist_array::AbstractDistArray,
                                               num_partitions_per_dim::Int)
    return dist_array.num_partitions_per_dim = num_partitions_per_dim
end

function text_file(file_path::AbstractString,
                   map_func::Function;
                   is_dense::Bool = false,
                   with_line_number::Bool = false,
                   new_keys::Bool = true,
                   flatten_results::Bool = false,
                   num_dims::Int64 = 0)::DistArray
    map_type = DistArrayMapType_no_map
    ValueType = String
    key_num_dims = 0
    local map_func_module, map_func_name
    map_func_module = which(Base.function_name(map_func))
    map_func_name = string(Base.function_name(map_func))
    if with_line_number && new_keys
        ValueType, key_num_dims = parse_map_new_keys_function(map_func, (Int64, AbstractString), flatten_results)
        map_type = DistArrayMapType_map
    elseif !with_line_number && new_keys
        ValueType, key_num_dims = parse_map_new_keys_function(map_func, (AbstractString,), flatten_results)
        map_type = DistArrayMapType_map_values_new_keys
    elseif with_line_number && !new_keys
        ValueType = parse_map_fixed_keys_function(map_func, (Int64, AbstractString), flatten_results)
        map_type = DistArrayMapType_map_fixed_keys
    else
        ValueType = parse_map_fixed_keys_function(map_func, (AbstractString,), flatten_results)
        map_type = DistArrayMapType_map_values
    end

    if new_keys
        if num_dims == 0
            @assert key_num_dims > 0
            num_dims = key_num_dims
        else
            @assert key_num_dims == num_dims
        end
    end

    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1
    local dist_array
    if new_keys
        if is_dense
            dist_array = DenseDistArray{ValueType, num_dims}(id, DistArrayParentType_text_file,
                                                              map_type)
        else
            println(typeof(num_dims))
            dist_array = SparseDistArray{ValueType, num_dims}(id, DistArrayParentType_text_file,
                                                              map_type)
        end
    else
        dist_array = ValueOnlyDistArray{ValueType}(id, DistArrayParentType_text_file, map_type)
    end
    dist_array.file_path = Nullable{String}(file_path)
    if map_type != DistArrayMapType_no_map
        dist_array.map_info = Nullable{DistArrayMapInfo}(
        DistArrayMapInfo(
            flatten_results,
            map_func_module,
            map_func_name))
    end
    dist_arrays[id] = dist_array
    return dist_array
end

function text_file(file_path::AbstractString;
                   is_dense::Bool = false,
                   with_line_number::Bool = false,
                   new_keys::Bool = true,
                   flatten_results::Bool = false,
                   num_dims::Int64 = 0)::DistArray
    map_type = DistArrayMapType_no_map
    ValueType = String
    key_num_dims = 0
    if new_keys && num_dims == 0
        @assert key_num_dims > 0
        num_dims = key_num_dims
    end

    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1
    local dist_array
    if new_keys
        if is_dense
            dist_array = DenseDistArray{ValueType, num_dims}(id,  DistArrayParentType_text_file,
                                                   map_type)
        else
            dist_array = SparseDistArray{ValueType, num_dims}(id, DistArrayParentType_text_file,
                                                    map_type)
        end
    else
        dist_array = ValueOnlyDistArray{ValueType}(id, DistArrayParentType_text_file,
                                                   map_type)
    end
    if map_type != DistArrayMapType_no_map
        dist_array.map_info = Nullable{DistArrayMapInfo}(
        DistArrayMapInfo(
            flatten_results,
            map_func_module,
            map_func_name))
    end
    dist_arrays[id] = dist_array
    return dist_array
end

function rand(ValueType::DataType, dims...)::DenseDistArray
    @assert ValueType == Float32 || ValueType == Float64
    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1
    dist_array = DenseDistArray{ValueType, length(dims)}(
        id,
        DistArrayParentType_init,
        DistArrayMapType_no_map)
    dist_array.init_info = Nullable{DistArrayInitInfo}(
        DistArrayInitInfo(
            DistArrayInitType_uniform_random,
            ValueType
        ))
    dist_array.partition_info = DistArrayPartitionInfo(DistArrayPartitionType_range,
                                                       DistArrayIndexType_none)

    dist_array.dims = [dims...]
    dist_array.iterate_dims = Nullable{Vector{Int64}}([dims...])
    dist_arrays[id] = dist_array
    return dist_array
end

function rand(dims...)::DistArray
    rand(Float32, dims...)
end

function randn(ValueType::DataType, dims...)::DenseDistArray
    @assert ValueType == Float32 || ValueType == Float64
    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1
    dist_array = DenseDistArray{ValueType, length(dims)}(
        id,
        DistArrayParentType_init,
        DistArrayMapType_no_map)
    dist_array.init_info = Nullable{DistArrayInitInfo}(
        DistArrayInitInfo(
            DistArrayInitType_normal_random,
            ValueType
        ))
    dist_array.partition_info = DistArrayPartitionInfo(DistArrayPartitionType_range,
                                                       DistArrayIndexType_none)
    dist_array.dims = [dims...]
    dist_array.iterate_dims = Nullable{Vector{Int64}}([dims...])
    dist_arrays[id] = dist_array
    return dist_array
end

function randn(dims...)::DistArray
    randn(Float32, dims...)
end

function call_create_dist_array{T, N}(dist_array::DistArray{T, N},
                                   value_type_buff_array::Vector{UInt8})
    flatten_results = false
    file_path = ""
    parent_id = -1
    init_type = DistArrayInitType_empty
    map_func_module = Module(:Main)
    map_func_name = ""
    random_init_type = Float32
    is_dense = isa(dist_array, DenseDistArray) ? true : false

    if !isnull(dist_array.map_info)
        map_info = get(dist_array.map_info)
        flatten_results = map_info.flatten_results
        if !isnull(map_info.parent_id)
            parent_id = get(map_info.parent_id)
        end
        map_func_module = map_info.map_func_module
        map_func_name = map_info.map_func_name
    end

    if !isnull(dist_array.init_info)
        init_info = get(dist_array.init_info)
        init_type = init_info.init_type
        random_init_type = init_info.random_init_type
    end

    if !isnull(dist_array.file_path)
        file_path = get(dist_array.file_path)
    end
    println(dist_array.dims)
    ccall((:orion_create_dist_array, lib_path),
          Void, (Int32, # id
                 Int32, # parent_type
                 Int32, # map_type
                 Int32, # partition_scheme
                 Bool, # flatten_results
                 UInt64, # num_dims
                 Int32, # value_type
                 Cstring, # file_path
                 Int32, # parent_id
                 Int32, # init_type
                 Int32, # map_func_module
                 Cstring, # map_func_name
                 Ptr{Int64}, # dims
                 Int32, # random_init_type
                 Bool, # is_dense
                 Cstring, # symbol
                 Ref{UInt8}, # value_type_bytes
                 UInt64 # value_type_size
                 ),
          dist_array.id,
          dist_array_parent_type_to_int32(dist_array.parent_type),
          dist_array_map_type_to_int32(dist_array.map_type),
          dist_array_partition_type_to_int32(dist_array.partition_info.partition_type),
          flatten_results,
          N,
          data_type_to_int32(T),
          file_path,
          parent_id,
          dist_array_init_type_to_int32(init_type),
          module_to_int32(Symbol(map_func_module)),
          map_func_name,
          N == 0 ? C_NULL : dist_array.dims,
          data_type_to_int32(random_init_type),
          is_dense,
          string(get(dist_array.symbol)),
          value_type_buff_array,
          length(value_type_buff_array))
end

function materialize{T, N}(dist_array::DistArray{T, N})
    if dist_array.is_materialized
        return
    end

    if dist_array.parent_type == DistArrayParentType_dist_array
        dist_array_to_create = process_dist_array_map(dist_array)
    else
        dist_array_to_create = dist_array
    end

    buff = IOBuffer()
    serialize(buff, eltype(dist_array_to_create))
    buff_array = takebuf_array(buff)

    call_create_dist_array(dist_array_to_create, buff_array)
    dist_array.is_materialized = true
    if isnull(dist_array.iterate_dims)
        dist_array.iterate_dims = Nullable{Vector{Int64}}(copy(dist_array.dims))
    end
end

function process_dist_array_map{T, N}(dist_array::DistArray{T, N})::DistArray{T, N}
    @assert dist_array.parent_type == DistArrayParentType_dist_array
    processed_dist_array = copy(dist_array)
    origin_dist_array = dist_array
    map_func_names = Vector{String}()
    map_func_modules = Vector{Module}()
    map_types = Vector{DistArrayMapType}()
    map_flattens = Vector{Bool}()

    while !origin_dist_array.is_materialized &&
        origin_dist_array.parent_type == DistArrayParentType_dist_array
        map_info = get(origin_dist_array.map_info)
        insert!(map_func_names, 1, map_info.map_func_name)
        insert!(map_func_modules, 1, map_info.map_func_module)
        insert!(map_types, 1, origin_dist_array.map_type)
        insert!(map_flattens, 1, map_info.flatten_results)
        parent_id = get(map_info.parent_id)
        origin_dist_array = dist_arrays[parent_id]
    end

    map_func_name_sym = gen_unique_symbol()
    flatten_results = reduce((x, y) -> x || y, false, map_flattens)
    map_values_only = (map_types[1] == DistArrayMapType_map_values) ||
        (map_types[1] == DistArrayMapType_map_values_new_keys)
    new_keys = reduce((x, y) -> x || y,
                      false,
                      Base.map(x -> (x == DistArrayMapType_map) || (x == DistArrayMapType_map_values_new_keys),
                               map_types))
    processed_dist_array.map_type =
        map_values_only ?
        (new_keys ? DistArrayMapType_map_values_new_keys : DistArrayMapType_map_values) :
        (new_keys ? DistArrayMapType_map : DistArrayMapType_map_fixed_keys)


    if origin_dist_array.parent_type == DistArrayParentType_init
        processed_dist_array.parent_type = DistArrayParentType_init
        processed_dist_array.init_info = origin_dist_array.init_info
        processed_dist_array.map_info = Nullable{DistArrayMapInfo}(
            DistArrayMapInfo(flatten_results,
                             Module(:Main),
                             string(map_func_name_sym)))
    elseif origin_dist_array.parent_type == DistArrayParentType_dist_array
        processed_dist_array.parent_type = DistArrayParentType_dist_array
        processed_dist_array.map_info = Nullable{DistArrayMapInfo}(
        DistArrayMapInfo(origin_dist_array.id,
                         flatten_results,
                         Module(:Main),
                         string(map_func_name_sym)))
        if origin_dist_array.partition_info.partition_type == DistArrayPartitionType_hash_server
            check_and_repartition(origin_dist_array,
                                  PartitionInfo(DistArrayPartitionType_hash_executor,
                                                DistArrayIndexType_none))
        end
    else
        @assert false "not supported"
    end

    if map_values_only
        generated_map_func = gen_map_values_function(map_func_name_sym,
                                                     map_func_names,
                                                     map_func_modules,
                                                     map_types,
                                                     map_flattens)
    else
        generated_map_func = gen_map_function(map_func_name_sym,
                                              map_func_names,
                                              map_func_modules,
                                              map_types,
                                              map_flattens)
    end
    eval_expr_on_all(generated_map_func, :Main)
    return processed_dist_array
end

function Base.size(dist_array::DistArray)
    return tuple(dist_array.dims...)
end

function map_generic{T, N}(parent_dist_array::DistArray{T, N},
                           map_func::Function,
                           map_value_only::Bool,
                           new_keys::Bool,
                           flatten_results::Bool,
                           is_dense::Bool,
                           alter_parent_sparsity::Bool,
                           new_dims::Nullable{Vector{Int64}})::DistArray
    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1

    map_func_module = which(Base.function_name(map_func))
    map_func_name = string(Base.function_name(map_func))
    local map_type
    local ValueType, key_num_dims
    if !map_value_only && new_keys
        ValueType, key_num_dims = parse_map_new_keys_function(map_func,
                                                              (Tuple, T),
                                                              flatten_results)
        map_type = DistArrayMapType_map
    elseif map_value_only && new_keys
        ValueType, key_num_dims = parse_map_new_keys_function(map_func,
                                                              (T,),
                                                              flatten_results)
        map_type = DistArrayMapType_map_values_new_keys
    elseif !map_value_only && !new_keys
        ValueType = parse_map_fixed_keys_function(map_func,
                                                  (Tuple, T),
                                                  flatten_results)
        map_type = DistArrayMapType_map_fixed_keys
    else
        ValueType = parse_map_fixed_keys_function(map_func,
                                                  (T,),
                                                  flatten_results)
        map_type = DistArrayMapType_map_values
    end

    partition_info = parent_dist_array.partition_info
    if partition_info.partition_type == DistArrayPartitionType_hash_server
        partition_info = DistArrayPartitionInfo(DistArrayPartitionType_hash_executor,
                                                DistArrayIndexType_none)
    end

    if map_type == DistArrayMapType_map ||
        map_type == DistArrayMapType_map_values_new_keys
        partition_info = DistArrayPartitionInfo(DistArrayPartitionType_naive,
                                                DistArrayIndexType_none)
    end

    local dist_array, num_dims, dims
    if isnull(new_dims)
        num_dims = N
    else
        num_dims = length(get(new_dims))
    end
    @assert !new_keys || num_dims == key_num_dims

    if isa(parent_dist_array, ValueOnlyDistArray)
        if new_keys && is_dense
            dist_array = DenseDistArray{ValueType, num_dims}(id,
                                                             get(new_dims),
                                                             DistArrayParentType_dist_array,
                                                             map_type)
        elseif new_keys && !is_dense
            dist_array = SparseDistArray{ValueType, num_dims}(id,
                                                              get(new_dims),
                                                              DistArrayParentType_dist_array,
                                                              map_type)
        else
            dist_array = ValueOnlyDistArray{ValueType}(id,
                                                       DistArrayParentType_dist_array,
                                                       map_type)
        end
    elseif alter_parent_sparsity
        if isa(parent_dist_array, DenseDistArray)
            dist_array = SparseDistArray{ValueType, num_dims}(id,
                                                              isnull(new_dims) ? parent_dist_array.dims : get(new_dims),
                                                              DistArrayParentType_dist_array,
                                                              map_type)
        else
            dist_array = DenseDistArray{ValueType, num_dims}(id,
                                                             isnull(new_dims) ? parent_dist_array.dims : get(new_dims),
                                                             DistArrayParentType_dist_array,
                                                             map_type)
        end
    else
        if isa(parent_dist_array, DenseDistArray)
            dist_array = DenseDistArray{ValueType, num_dims}(id,
                                                             isnull(new_dims) ? parent_dist_array.dims : get(new_dims),
                                                             DistArrayParentType_dist_array,
                                                             map_type)
        else
            dist_array = SparseDistArray{ValueType, num_dims}(id,
                                                             isnull(new_dims) ? parent_dist_array.dims : get(new_dims),
                                                              DistArrayParentType_dist_array,
                                                              map_type)
        end
    end

    dist_array.map_info = Nullable{DistArrayMapInfo}(
        DistArrayMapInfo(
            parent_dist_array.id,
            flatten_results,
            map_func_module,
            map_func_name
        ))

    dist_array.partition_info = partition_info

    dist_array.iterate_dims = Nullable{Vector{Int64}}(copy(dist_array.dims))
    dist_arrays[id] = dist_array
    return dist_array
end

function map(parent_dist_array::DistArray, map_func::Function;
             map_values::Bool = false,
             new_keys::Bool = false,
             flatten_results::Bool = false,
             is_dense::Bool = false,
             alter_parent_sparsity::Bool = false,
             new_dims::Nullable{Vector{Int64}} = Nullable{Vector{Int64}}())::DistArray

    return map_generic(parent_dist_array, map_func, map_values, new_keys, flatten_results,
                       is_dense, alter_parent_sparsity, new_dims)
end

function check_and_repartition(dist_array::DistArray,
                               partition_info::DistArrayPartitionInfo)
    println("check_and_repartition ", dist_array.id, " ", partition_info.partition_type)
    curr_partition_info = dist_array.partition_info
    dist_array.target_partition_info = Nullable{DistArrayPartitionInfo}()
    repartition = false
    if !isnull(partition_info.partition_func_name) &&
        !isnull(curr_partition_info.partition_func_name) &&
        get(partition_info.partition_func_name) == get(curr_partition_info.partition_func_name)
        repartition = false
    elseif partition_info.partition_type == DistArrayPartitionType_2d_unimodular
        repartition = true
    elseif is_partition_equal(partition_info, curr_partition_info)
        repartition = false
    else
        repartition = true
    end
    local partition_func_name = ""
    if !isnull(partition_info.partition_func_name)
        partition_func_name = get(partition_info.partition_func_name)
    end
    if repartition
        dist_array.partition_info = partition_info
        ccall((:orion_repartition_dist_array, lib_path),
        Void, (Int32, Cstring, Int32, Int32),
        dist_array.id,
        partition_func_name,
        dist_array_partition_type_to_int32(partition_info.partition_type),
        dist_array_index_type_to_int32(partition_info.index_type))
    end
end

function dist_array_set_symbol{T, N}(dist_array::AbstractDistArray{T, N},
                                     symbol)
    buff = IOBuffer()
    serialize(buff, T)
    buff_array = takebuf_array(buff)
    dist_array.symbol = symbol
end

# DistArray Access
# A DistArray supports both iteration (range-based for loop) and indexed query

# Range-based for loop:
# syntax: for iteration_var in dist_array
# If dist_array.iterate_dims == dist_array.dims
# iteration_var is a tuple that contains the DistArray element
# iteration_var[1] is the key and iteration_var[2] is the value
#
# else iteration_dims contains the first K elements of dims
# iteration_var is a list of elements whose key share the same first-K prefix
# iteration_var[1] is a list of keys
# iteration_var[2] is a list of values

# Point Query:
# syntax: a[1, 2, 3]
# returns a single value

# Range Query:
# syntax (for example): a[:, 1]
# returns var

function set_iterate_dims(dist_array::DistArray,
                          dims::Vector{Int64})
    dist_array.iterate_dims = Nullable{Vector{Int64}}(copy(dims))
end

function from_int64_to_keys(key::Int64, dims::Vector{Int64})::Vector{Int64}
    dim_keys = []
    for dim in dims
        key_this_dim = key % dim + 1
        push!(dim_keys, key_this_dim)
        key = fld(key, dim)
    end
    return dim_keys
end

function from_keys_to_int64(key, dims::Vector{Int64})::Int64
    key_int = 0
    for i = length(dims):-1:2
        key_int += key[i] - 1
        key_int *= dims[i - 1]
    end
    key_int += key[1] - 1
    return key_int
end

function create_dist_array_accessor{T, N}(
    dist_array::DenseDistArray{T, N},
    key_begin::Int64,
    values::Vector{T})
    dist_array.accessor = Nullable{DenseDistArrayAccessor{T, N}}(
        DenseDistArrayAccessor{T, N}(key_begin, values,
                                     dist_array.dims))
end

function create_dist_array_accessor{T, N}(
    dist_array::SparseDistArray{T, N},
    keys::Vector{Int64},
    values::Vector)
    dist_array.accessor = Nullable{SparseDistArrayAccesor{T, N}}(
        SparseDistArrayAccessor{T, N}(keys, values,
                                      dist_array.dims))
end

function create_dist_array_cache_accessor{T, N}(
    dist_array::DistArray{T, N},
    keys::Vector{Int64},
    values::Vector{T})
    dist_array.cache_accessor = Nullable{DistArrayCacheAccessor{T, N}}(
        DistArrayCacheAccessor{T, N}(dist_array.id,
                                     keys, values,
                                     dist_array.dims))

end

function delete_dist_array_accessor{T, N}(dist_array::DistArray{T, N})
    dist_array.accessor = Nullable{DistArrayAccessor{T, N}}()
    dist_array.cache_accessor = Nullable{DistArrayAccessor{T, N}}()
end

function dist_array_get_accessor_keys_vec(dist_array::DistArray)::Vector{Int64}
    accessor = isnull(dist_array.cache_accessor) ?
        get(dist_array.accessor) :
        get(dist_array.cache_accessor)
    dist_array_accessor_get_keys_vec(accessor)
end

function dist_array_get_accessor_values_vec(dist_array::DistArray)::Vector
    accessor = isnull(dist_array.cache_accessor) ?
        get(dist_array.accessor) :
        get(dist_array.cache_accessor)
    dist_array_accessor_get_values_vec(accessor)
end

function Base.size(dist_array::AbstractDistArray)
    return tuple(dist_array.dims...)
end

function Base.getindex(dist_array::DistArray,
                       I...)
    if !isnull(dist_array.cache_accessor)
        accessor = get(dist_array.cache_accessor)
        return getindex(accessor, I...)
    else
        accessor = get(dist_array.accessor)
        return getindex(accessor, I...)
    end

end

function Base.setindex!(dist_array::DistArray,
                        v, I...)
    accessor = get(dist_array.accessor)
    setindex!(accessor, v, I...)
end

function dist_array_create_and_append_partition{T, N}(dist_array::AbstractDistArray{T, N})::DistArrayPartition{T}
    values = Vector{T}()
    partition = DistArrayPartition{T}(values)
    push!(dist_array.partitions, partition)
    return partition
end

function dist_array_delete_partitions(dist_array::DistArray,
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

function dist_array_clear_partition(partition::DistArrayPartition)
    resize!(partition.values, 0)
end

immutable DistArrayAccessSetRecorder{N} <: AbstractArray{Int32, N}
    keys_set::Set{Int64}
    dims::NTuple{N, Int64}
    DistArrayAccessSetRecorder(dims::Vector{Int64}) = new(Set{Int64}(),
                                                          tuple(dims...))

    DistArrayAccessSetRecorder(dims::NTuple{N, Int64}) = new(Set{Int64}(),
                                                             dims)
end

function Base.linearindexing{T<:DistArrayAccessSetRecorder}(::Type{T})
    return Base.LinearFast()
end

function Base.size(access_recorder::DistArrayAccessSetRecorder)
    return access_recorder.dims
end

function Base.getindex(access_recorder::DistArrayAccessSetRecorder,
                       i::Int)
    push!(access_recorder.keys_set, i)
    return 0
end

function Base.setindex!(access_reorder::DistArrayAccessSetRecorder,
                        v,
                        i::Int)
    @assert false "no writes supported"
end

function Base.similar{T, N}(access_recorder::DistArrayAccessSetRecorder{N},
                            ::Type{T}, dims::NTuple{N, Int64})
    return DistArrayAccessSetRecorder{N}(dims)
end

immutable DistArrayAccessCountRecorder{N} <: AbstractArray{Int32, N}
    keys_dict::Dict{Int64, UInt64}
    dims::NTuple{N, Int64}
    DistArrayAccessCountRecorder(dims::Vector{Int64}) = new(Dict{Int64, UInt64}(),
                                                            tuple(dims...))

    DistArrayAccessCountRecorder(dims::NTuple{N, Int64}) = new(Dict{Int64, UInt64}(),
                                                               dims)
end

function Base.linearindexing{T<:DistArrayAccessCountRecorder}(::Type{T})
    return Base.LinearFast()
end

function Base.size(access_recorder::DistArrayAccessCountRecorder)
    return access_recorder.dims
end

function Base.getindex(access_recorder::DistArrayAccessCountRecorder,
                       i::Int)
    if i in keys(access_recorder.keys_dict)
        access_recorder.keys_dict[i] += 1
    else
        access_recorder.keys_dict[i] = 1
    end
    return 0
end

function Base.setindex!(access_reorder::DistArrayAccessCountRecorder,
                        v,
                        i::Int)
    @assert false "no writes supported"
end

function Base.similar{T, N}(access_recorder::DistArrayAccessCountRecorder{N},
                            ::Type{T}, dims::NTuple{N, Int64})
    return DistArrayAccessCountRecorder{N}(dims)
end
