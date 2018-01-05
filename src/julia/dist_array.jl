import Base
import Base: copy, ==
import Main: size, getindex, setindex!
export DistArray, size, getindex, setindex!

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
    5 DistArrayPartitionType_hash =
    6

@enum DistArrayIndexType DistArrayIndexType_none =
    1 DistArrayIndexType_global =
    2 DistArrayIndexType_local =
    3

@enum ForLoopParallelScheme ForLoopParallelScheme_naive =
    1 ForLoopParallelScheme_1d =
    2 ForLoopParallelScheme_2d =
    3 ForLoopParallelScheme_unimodular =
    4 ForLoopParallelScheme_none =
    5

type DistArrayPartitionInfo
    partition_type::DistArrayPartitionType
    partition_func_name::String
    partition_dims::Union{Tuple, Void} # tuple or nothing
    tile_sizes::Union{Tuple, Void} # tuple or nothing
    index_type::DistArrayIndexType
end

type DistArrayPartitionDistinguisher
    partition_type::DistArrayPartitionType
    partition_dims
    tile_sizes
    index_type::DistArrayIndexType
end

function get_dist_array_partition_distinguisher(partition_info::DistArrayPartitionInfo)
    return DistArrayPartitionDistinguisher(
    partition_info.partition_type,
    partition_info.partition_dims,
    partition_info.tile_sizes,
    partition_info.index_type)
end

function ==(partition_a::DistArrayPartitionDistinguisher,
            partition_b::DistArrayPartitionDistinguisher)
    return partition_a.partition_type == partition_b.partition_type &&
        partition_a.partition_dims == partition_b.partition_dims &&
        partition_a.tile_sizes == partition_b.tile_sizes &&
        partition_a.index_type == partition_b.index_type
end

abstract AbstractDistArray{T} <: AbstractArray{T}

type DistArray{T} <: AbstractDistArray{T}
    id::Int32
    parent_type::DistArrayParentType
    flatten_results::Bool
    map_type::DistArrayMapType
    num_dims::UInt64
    ValueType::DataType
    file_path::String
    parent_id::Int32
    init_type::DistArrayInitType
    map_func_module::Module
    map_func_name::String
    is_materialized::Bool
    dims::Vector{Int64}
    random_init_type::DataType
    is_dense::Bool
    partition_info::DistArrayPartitionInfo
    symbol
    access_ptr
    iterate_dims::Vector{Int64}
    partitions::Vector{Vector{T}}
    num_partitions_per_dim::Int64
    DistArray(id::Integer,
              parent_type::DistArrayParentType,
              flatten_results::Bool,
              map_type::DistArrayMapType,
              num_dims::Integer,
              file_path::String,
              parent_id::Integer,
              init_type::DistArrayInitType,
              map_func_module::Module,
              map_func_name::String,
              is_materialized::Bool,
              random_init_type::DataType,
              is_dense::Bool,
              partition_info::DistArrayPartitionInfo) = new(
                  id,
                  parent_type,
                  flatten_results,
                  map_type,
                  num_dims,
                  T,
                  file_path,
                  parent_id,
                  init_type,
                  map_func_module,
                  map_func_name,
                  is_materialized,
                  zeros(Int64, num_dims),
                  random_init_type,
                  is_dense,
                  partition_info,
                  nothing,
                  nothing,
                  zeros(Int64, num_dims),
                  Vector{Vector{T}}(),
                  num_executors * 2)

    DistArray() = new(-1,
                      DistArrayParentType_init,
                      false,
                      DistArrayMapType_no_map,
                      0,
                      T,
                      "",
                      -1,
                      DistArrayInitType_empty,
                      Main,
                      "",
                      false,
                      zeros(Int64, 0),
                      Void,
                      false,
                      DistArrayPartitionInfo(DistArrayPartitionType_naive, "",
                                             nothing, nothing, DistArrayIndexType_none),
                      nothing,
                      nothing,
                      zeros(Int64, 0),
                      Vector{Vector{T}}(),
                      1)
end

const dist_arrays = Dict{Int32, AbstractDistArray}()
dist_array_id_counter = 0

function text_file(
    file_path::AbstractString,
    map_func::Function;
    is_dense::Bool = false,
    with_line_number::Bool = false,
    new_keys::Bool = true,
    flatten::Bool = false,
    num_dims::Int64 = 0)::DistArray

    map_func_module = which(Base.function_name(map_func))
    map_func_name = string(Base.function_name(map_func))
    local map_type
    local ValueType, key_num_dims
    if with_line_number && new_keys
        ValueType, key_num_dims = parse_map_new_keys_function(map_func, (Int64, AbstractString), flatten)
        map_type = DistArrayMapType_map
    elseif !with_line_number && new_keys
        ValueType, key_num_dims = parse_map_new_keys_function(map_func, (AbstractString,), flatten)
        map_type = DistArrayMapType_map_values_new_keys
    elseif with_line_number && !new_keys
        ValueType = parse_map_fixed_keys_function(map_func, (Int64, AbstractString), flatten)
        map_type = DistArrayMapType_map_fixed_keys
    else
        ValueType = parse_map_fixed_keys_function(map_func, (AbstractString,), flatten)
        map_type = DistArrayMapType_map_values
    end

    if new_keys && num_dims == 0
        @assert key_num_dims > 0
        num_dims = key_num_dims
    end

    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1
    dist_array = DistArray{ValueType}(
        id,
        DistArrayParentType_text_file,
        flatten,
        map_type,
        num_dims,
        file_path,
        -1,
        DistArrayInitType_empty,
        map_func_module,
        map_func_name,
        false,
        Void,
        is_dense,
        DistArrayPartitionInfo(DistArrayPartitionType_naive, "",
                               nothing, nothing, DistArrayIndexType_none))
    dist_arrays[id] = dist_array
    return dist_array
end

function rand(ValueType::DataType, dims...)::DistArray
    @assert ValueType == Float32 || ValueType == Float64
    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1
    dist_array = DistArray{ValueType}(
        id,
        DistArrayParentType_init,
        false,
        DistArrayMapType_no_map,
        length(dims),
        "",
        -1,
        DistArrayInitType_uniform_random,
        Module(),
        "",
        false,
        ValueType,
        true,
        DistArrayPartitionInfo(DistArrayPartitionType_range,
                               "", nothing, nothing,
                               DistArrayIndexType_none))
    dist_array.dims = [dims...]
    dist_array.iterate_dims = [dims...]
    dist_arrays[id] = dist_array
    return dist_array
end

function rand(dims...)::DistArray
    rand(Float32, dims...)
end

function randn(ValueType::DataType, dims...)::DistArray
    @assert ValueType == Float32 || ValueType == Float64
    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1
    dist_array = DistArray{ValueType}(
        id,
        DistArrayParentType_init,
        false,
        DistArrayMapType_no_map,
        length(dims),
        "",
        -1,
        DistArrayInitType_normal_random,
        Module(),
        "",
        false,
        ValueType,
        true,
        DistArrayPartitionInfo(DistArrayPartitionType_range,
                               "", nothing, nothing,
                               DistArrayIndexType_none))
    dist_array.dims = [dims...]
    dist_array.iterate_dims = [dims...]
    dist_arrays[id] = dist_array
    return dist_array
end

function randn(dims...)::DistArray
    randn(Float32, dims...)
end

function materialize(dist_array::DistArray)
    if dist_array.is_materialized
        return
    end

    if dist_array.parent_type == DistArrayParentType_dist_array
        dist_array_to_create = process_dist_array_map(dist_array)
        #dist_array_to_create.map_func_name = "map_values_func"
        #dist_array_to_create.map_func_module = Main
    else
        dist_array_to_create = dist_array
    end

    buff = IOBuffer()
    serialize(buff, dist_array_to_create.ValueType)
    buff_array = takebuf_array(buff)

    println("map_type = ", dist_array_to_create.map_type)
    ccall((:orion_create_dist_array, lib_path),
          Void, (Int32, Int32, Int32, Int32, Bool, UInt64, Int32,
                 Cstring, Int32, Int32, Int32, Cstring, Ptr{Int64}, Int32,
                 Bool, Cstring, Ref{UInt8}, UInt64),
          dist_array_to_create.id,
          dist_array_parent_type_to_int32(dist_array_to_create.parent_type),
          dist_array_map_type_to_int32(dist_array_to_create.map_type),
          dist_array_partition_type_to_int32(dist_array_to_create.partition_info.partition_type),
          dist_array_to_create.flatten_results,
          dist_array_to_create.num_dims,
          data_type_to_int32(dist_array_to_create.ValueType),
          dist_array_to_create.file_path,
          dist_array_to_create.parent_id,
          dist_array_init_type_to_int32(dist_array_to_create.init_type),
          module_to_int32(Symbol(dist_array_to_create.map_func_module)),
          dist_array_to_create.map_func_name,
          dist_array_to_create.dims,
          data_type_to_int32(dist_array_to_create.random_init_type),
          dist_array_to_create.is_dense,
          dist_array_to_create.symbol,
          buff_array,
          length(buff_array))
    dist_array.is_materialized = true
    if dist_array.iterate_dims == zeros(Int64, length(dist_array.iterate_dims))
        dist_array.iterate_dims = copy(dist_array.dims)
    end
end

function copy(dist_array::DistArray)::DistArray
    new_dist_array = DistArray{dist_array.ValueType}()
    new_dist_array.id = dist_array.id
    new_dist_array.parent_type = dist_array.parent_type
    new_dist_array.map_type = dist_array.map_type
    new_dist_array.num_dims = dist_array.num_dims
    new_dist_array.ValueType = dist_array.ValueType
    new_dist_array.file_path = dist_array.file_path
    new_dist_array.parent_id = dist_array.parent_id
    new_dist_array.init_type = dist_array.init_type
    new_dist_array.map_func_module = dist_array.map_func_module
    new_dist_array.map_func_name = dist_array.map_func_name
    new_dist_array.is_materialized = dist_array.is_materialized
    new_dist_array.dims = copy(dist_array.dims)
    new_dist_array.random_init_type = dist_array.random_init_type
    new_dist_array.is_dense = dist_array.is_dense
    new_dist_array.partition_info = dist_array.partition_info
    new_dist_array.symbol = dist_array.symbol
    new_dist_array.iterate_dims = copy(dist_array.iterate_dims)
    return new_dist_array
end

function process_dist_array_map(dist_array::DistArray)::DistArray
    @assert dist_array.parent_type == DistArrayParentType_dist_array
    processed_dist_array = copy(dist_array)
    origin_dist_array = dist_array
    map_func_names = Vector{String}()
    map_func_modules = Vector{Module}()
    map_types = Vector{DistArrayMapType}()
    map_flattens = Vector{Bool}()

    while !origin_dist_array.is_materialized &&
        origin_dist_array.parent_type == DistArrayParentType_dist_array

        insert!(map_func_names, 1, origin_dist_array.map_func_name)
        insert!(map_func_modules, 1, origin_dist_array.map_func_module)
        insert!(map_types, 1, origin_dist_array.map_type)
        insert!(map_flattens, 1, origin_dist_array.flatten_results)
        parent_id = origin_dist_array.parent_id
        origin_dist_array = dist_arrays[parent_id]
    end

    map_func_name_sym = gen_unique_symbol()
    processed_dist_array.map_func_name = string(map_func_name_sym)
    processed_dist_array.map_func_module = Module(:Main)
    processed_dist_array.parent_type = origin_dist_array.parent_type

    if origin_dist_array.parent_type == DistArrayParentType_init
        processed_dist_array.random_init_type = origin_dist_array.random_init_type
    end

    if origin_dist_array.parent_type != DistArrayParentType_dist_array
        processed_dist_array.parent_id = -1
        processed_dist_array.init_type = origin_dist_array.init_type
    else
        processed_dist_array.parent_id = origin_dist_array.id
        processed_dist_array.init_type = DistArrayInitType_empty
    end

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
    flatten_results = reduce((x, y) -> x || y, false, map_flattens)

    if origin_dist_array.parent_type == DistArrayParentType_text_file
        @assert false "not yet supported"
    else
        processed_dist_array.flatten_results = false
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
    end
    return processed_dist_array
end

function Base.size(dist_array::DistArray)
    return tuple(dist_array.dims...)
end

function map_generic(parent_dist_array::DistArray,
                     map_func::Function,
                     map_value_only::Bool,
                     new_keys::Bool,
                     flatten::Bool,
                     num_dims::Int64)::DistArray
    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1

    map_func_module = which(Base.function_name(map_func))
    map_func_name = string(Base.function_name(map_func))
    local map_type
    local ValueType, key_num_dims
    if !map_value_only && new_keys
        ValueType, key_num_dims = parse_map_new_keys_function(map_func, (Tuple, parent_dist_array.ValueType), flatten)
        map_type = DistArrayMapType_map
    elseif map_value_only && new_keys
        ValueType, key_num_dims = parse_map_new_keys_function(map_func, (parent_dist_array.ValueType,), flatten)
        map_type = DistArrayMapType_map_values_new_keys
    elseif !map_value_only && !new_keys
        ValueType = parse_map_fixed_keys_function(map_func, (Tuple, parent_dist_array.ValueType), flatten)
        map_type = DistArrayMapType_map_fixed_keys
    else
        ValueType = parse_map_fixed_keys_function(map_func, (parent_dist_array.ValueType,), flatten)
        map_type = DistArrayMapType_map_values
    end

    partition_info = parent_dist_array.partition_info

    if map_type == DistArrayMapType_map ||
        map_type == DistArrayMapType_map_values_new_keys
        partition_info = DistArrayPartitionInfo(DistArrayPartitionType_naive,
                                                "", nothing, nothing,
                                                DistArrayIndexType_none)
    end
    if new_keys && num_dims == 0
        @assert key_num_dims > 0
        num_dims = key_num_dims
    end
    @assert !new_keys ||
        (isempty(parent_dist_array.dims) || length(parent_dist_array.dims) == num_dims)

    dist_array = DistArray{ValueType}(
        id,
        DistArrayParentType_dist_array,
        flatten,
        map_value_only ?
        (new_keys ? DistArrayMapType_map_values_new_keys : DistArrayMapType_map_values) :
        (new_keys ? DistArrayMapType_map : DistArrayMapType_map_fixed_keys),
        length(parent_dist_array.dims),
        "",
        parent_dist_array.id,
        DistArrayInitType_empty,
        map_func_module,
        map_func_name,
        false,
        parent_dist_array.random_init_type,
        parent_dist_array.is_dense,
        partition_info)
    if !new_keys || !isempty(parent_dist_array.dims)
        dist_array.dims = parent_dist_array.dims
    else
        dist_array.num_dims = num_dims
    end
    dist_array.iterate_dims = parent_dist_array.iterate_dims
    dist_arrays[id] = dist_array
    return dist_array
end

function map(parent_dist_array::DistArray, map_func::Function;
             map_values::Bool = false,
             new_keys::Bool = false,
             flatten::Bool = false,
             num_dims::Int64 = 0)::DistArray
    return map_generic(parent_dist_array, map_func, map_values, new_keys, flatten, num_dims)
end

function check_and_repartition(dist_array::DistArray,
                               partition_info::DistArrayPartitionInfo)
    println("check_and_repartition ", dist_array.id)
    curr_partition_info = dist_array.partition_info
    repartition = false
    if partition_info.index_type == DistArrayIndexType_global
        partition_info.partition_type = DistArrayPartitionType_hash
    end

    if partition_info.partition_func_name == curr_partition_info.partition_func_name
        repartition = false
    elseif partition_info.partition_type == DistArrayPartitionType_2d_unimodular
        repartition = true
    else
        partition_dist_new = get_dist_array_partition_distinguisher(partition_info)
        partition_dist_old = get_dist_array_partition_distinguisher(curr_partition_info)
        if partition_dist_new == partition_dist_old
            repartition = false
        else
            repartition = true
        end
    end
    if repartition
        dist_array.partition_info = partition_info
        ccall((:orion_repartition_dist_array, lib_path),
              Void, (Int32, Cstring, Int32, Int32),
              dist_array.id,
              partition_info.partition_func_name,
              dist_array_partition_type_to_int32(partition_info.partition_type),
              dist_array_index_type_to_int32(partition_info.index_type))
    end
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
# If a is dense, var is a list of values
# else var[1] is a list of keys and var[2] is a list of values

function set_iterate_dims(dist_array::DistArray,
                          dims::Vector{Int64})
    dist_array.iterate_dims = copy(dims)
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
