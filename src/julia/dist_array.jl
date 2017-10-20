import Base
import Base.copy
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
    partition_func_name
    partition_dims
    index_type::DistArrayIndexType
end

type DistArray{T} <: AbstractArray{T}
    id::Int32
    parent_type::DistArrayParentType
    flatten_results::Bool
    map_type::DistArrayMapType
    num_dims::UInt64
    ValueType::DataType
    file_path::String
    parent_id::Int32
    init_type::DistArrayInitType
    mapper_func_module::Module
    mapper_func_name::String
    is_materialized::Bool
    dims::Vector{Int64}
    random_init_type::DataType
    is_dense::Bool
    partition_info::DistArrayPartitionInfo
    symbol
    access_ptr
    DistArray(id::Integer,
              parent_type::DistArrayParentType,
              flatten_results::Bool,
              map_type::DistArrayMapType,
              num_dims::Integer,
              file_path::String,
              parent_id::Integer,
              init_type::DistArrayInitType,
              mapper_func_module::Module,
              mapper_func_name::String,
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
                  mapper_func_module,
                  mapper_func_name,
                  is_materialized,
                  zeros(Int64, num_dims),
                  random_init_type,
                  is_dense,
                  partition_info,
                  nothing,
                  nothing)

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
                      DistArrayPartitionInfo(DistArrayPartitionType_naive, nothing,
                                             nothing, DistArrayIndexType_none),
                      nothing,
                      nothing)
end

const dist_arrays = Dict{Int32, DistArray}()
dist_array_id_counter = 0

function text_file(
    file_path::AbstractString,
    parser_func::Function,
    is_dense::Bool = false)::DistArray

    parser_func_module = which(Base.function_name(parser_func))
    parser_func_name = string(Base.function_name(parser_func))

    ValueType, num_dims, flatten_results, preserving_keys =
        parse_map_function(parser_func, (AbstractString,))
    @assert !preserving_keys

    map_type = num_dims > 0 ?
        DistArrayMapType_map_values_new_keys :
        DistArrayMapType_map_values

    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1
    dist_array = DistArray{ValueType}(
        id,
        DistArrayParentType_text_file,
        flatten_results,
        map_type,
        num_dims,
        file_path,
        -1,
        DistArrayInitType_empty,
        parser_func_module,
        parser_func_name,
        false,
        Void,
        is_dense,
        DistArrayPartitionInfo(DistArrayPartitionType_naive, nothing, nothing,
                               DistArrayIndexType_none))
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
        DistArrayPartitionInfo(DistArrayPartitionType_range, nothing, nothing,
                           DistArrayIndexType_none))
    dist_array.dims = [dims...]
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
        DistArrayPartitionInfo(DistArrayPartitionType_range, nothing, nothing,
                           DistArrayIndexType_none))
    dist_array.dims = [dims...]
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
        #dist_array_to_create.mapper_func_name = "map_values_func"
        #dist_array_to_create.mapper_func_module = Main
    else
        dist_array_to_create = dist_array
    end

    println("map_type = ", dist_array_to_create.map_type)
    ccall((:orion_create_dist_array, lib_path),
          Void, (Int32, Int32, Int32, Bool, UInt64, Int32,
                 Cstring, Int32, Int32, Int32, Cstring, Ptr{Int64}, Int32,
                 Bool, Cstring),
          dist_array_to_create.id,
          dist_array_parent_type_to_int32(dist_array_to_create.parent_type),
          dist_array_map_type_to_int32(dist_array_to_create.map_type),
          dist_array_to_create.flatten_results,
          dist_array_to_create.num_dims,
          data_type_to_int32(dist_array_to_create.ValueType),
          dist_array_to_create.file_path,
          dist_array_to_create.parent_id,
          dist_array_init_type_to_int32(dist_array_to_create.init_type),
          module_to_int32(Symbol(dist_array_to_create.mapper_func_module)),
          dist_array_to_create.mapper_func_name,
          dist_array_to_create.dims,
          data_type_to_int32(dist_array_to_create.random_init_type),
          dist_array_to_create.is_dense,
          dist_array_to_create.symbol)
    dist_array.is_materialized = true
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
    new_dist_array.mapper_func_module = dist_array.mapper_func_module
    new_dist_array.mapper_func_name = dist_array.mapper_func_name
    new_dist_array.is_materialized = dist_array.is_materialized
    new_dist_array.dims = copy(dist_array.dims)
    new_dist_array.random_init_type = dist_array.random_init_type
    new_dist_array.is_dense = dist_array.is_dense
    new_dist_array.partition_info = dist_array.partition_info
    new_dist_array.symbol = dist_array.symbol
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

        insert!(map_func_names, 1, origin_dist_array.mapper_func_name)
        insert!(map_func_modules, 1, origin_dist_array.mapper_func_module)
        insert!(map_types, 1, origin_dist_array.map_type)
        insert!(map_flattens, 1, origin_dist_array.flatten_results)
        parent_id = origin_dist_array.parent_id
        origin_dist_array = dist_arrays[parent_id]
    end

    map_func_name_sym = gen_unique_symbol()
    processed_dist_array.mapper_func_name = string(map_func_name_sym)
    processed_dist_array.mapper_func_module = Module(:Main)
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

function space_time_repartition(dist_array::DistArray,
                                partition_func_name::AbstractString)

end

function repartition_1d(dist_array::DistArray,
                        partition_func_name::AbstractString)
    ccall((:orion_repartition_1d_dist_array, lib_path),
          Void, (Int32, Cstring),
          dist_array.id,
          partition_func_name)
end

function Base.size(dist_array::DistArray)
    return tuple(dist_array.dims...)
end

function map_generic(parent_dist_array::DistArray,
                     map_func::Function,
                     map_value_only::Bool)::DistArray
    global dist_array_id_counter
    id = dist_array_id_counter
    dist_array_id_counter += 1

    map_func_module = which(Base.function_name(map_func))
    map_func_name = string(Base.function_name(map_func))

    if !map_value_only
        ValueType, num_dims, flatten_results, preserving_keys =
            parse_map_function(map_func, (Tuple, parent_dist_array.ValueType))
    else
        ValueType, num_dims, flatten_results, preserving_keys =
            parse_map_function(map_func, (parent_dist_array.ValueType,))
    end

    dist_array = DistArray{ValueType}(
        id,
        DistArrayParentType_dist_array,
        flatten_results,
        map_value_only ?
        (preserving_keys ? DistArrayMapType_map_values : DistArrayMapType_map_values_new_keys) :
        (preserving_keys ? DistArrayMapType_map_fixed_keys : DistArrayMapType_map),
        length(parent_dist_array.dims),
        "",
        parent_dist_array.id,
        DistArrayInitType_empty,
        map_func_module,
        map_func_name,
        false,
        parent_dist_array.random_init_type,
        parent_dist_array.is_dense,
        DistArrayPartitionInfo(DistArrayPartitionType_naive, nothing, nothing, DistArrayIndexType_none))
    dist_array.dims = parent_dist_array.dims
    dist_arrays[id] = dist_array
    return dist_array
end

function map(parent_dist_array::DistArray, map_func::Function)::DistArray
    return map_generic(parent_dist_array, map_func, false)
end

function map_value(parent_dist_array::DistArray, map_func::Function)::DistArray
    return map_generic(parent_dist_array, map_func, true)
end

function check_and_repartition(dist_array::DistArray,
                               partition_info::DistArrayPartitionInfo)
    println("check_and_repartition ", dist_array.id)
    curr_partition_info = dist_array.partition_info
    repartition = false
    if partition_info.index_type == DistArrayIndexType_global
        if dist_array.is_dense
            partition_info.partition_type = DistArrayPartitionType_range
        else
            partition_info.partition_type = DistArrayPartitionType_hash
        end
    end
    if curr_partition_info.partition_type ==
        partition_info.partition_type
        if (partition_info.partition_type == DistArrayPartitionType_1d ||
            partition_info.partition_type == DistArrayPartitionType_2d) &&
            partition_info.partition_dims != curr_partition_info.partition_dims
            repartition = true
        elseif partition_info.partition_type == DistArrayPartitionType_2d_unimodular &&
            partition_info.partition_func_name != curr_partition_info.partition_func_name
            repartition = true
        end
    else
        repartition = true
    end
    dist_array.partition_info = partition_info
    if repartition
        ccall((:orion_repartition_dist_array, lib_path),
              Void, (Int32, Cstring, Int32, Int32),
              dist_array.id,
              partition_info.partition_func_name,
              dist_array_partition_type_to_int32(partition_info.partition_type),
              dist_array_index_type_to_int32(partition_info.index_type))
    end
end

function dist_array_set_symbol(dist_array::DistArray,
                               symbol)
    dist_array.symbol = symbol
    println(dist_array.dims, " symbol type = ", typeof(symbol), " ", symbol)
end

function getindex(dist_array::DistArray, index::Integer)
end

function setindex!(dist_array::DistArray, value, index::Integer)
end

function dist_array_get_value_type(dist_array::DistArray)::DataType
    value_type_int32 = ccall((:orion_dist_array_get_value_type, lib_path),
                             Int32, (Ptr{Void},), partition_ptr)
    return int32_to_data_type(value_type_int32)
end

function getindex(dist_array::DistArray, index...)
    num_dist_array_dims = length(index)
    dims = dist_array.dims
    ValueType = dist_array.ValueType
    access_ptr = dist_array.access_ptr
    if length(index) == 1
        value_array = Vector{ValueType}(1)
        ccall((:orion_dist_array_read, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
              access_ptr, index[1], 1, value_array)
        return value_array
    end

    result_dims = Vector{Int64}()
    @assert length(index) == length(dims)
    for i in eachindex(index)
        dim_i = index[i]
        if isa(dim_i, Integer)
        elseif isa(dim_i, Vector)
            push!(result_dims, length(dim_i))
        elseif dim_i == Colon()
            push!(result_dims, dims[i])
        elseif isa(dim_i, UnitRange)
            push!(result_dims, max(0, dim_i[end] - dim_i[1] + 1))
        else
            @assert false
        end
    end

    per_read_size = 1
    num_dims_per_read = 0
    # column major!!
    for i in eachindex(index)
        dim_i = index[i]
        if dim_i == Colon()
            per_read_size *= dims[i]
        else
            break
        end
        num_dims_per_read += 1
    end

    read_array = Vector{ValueType}(per_read_size)
    read_dims_begin = num_dims_per_read + 1

    num_reads_by_dim = Vector{Int64}()
    for i = read_dims_begin:length(dims)
        dim_i = index[i]
        if dim_i == Colon()
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

    array_size = reduce((x, y) -> x * y, 1, result_dims)
    read_offset = 1
    result_array = Vector{ValueType}(array_size)

    num_reads = fld(array_size, per_read_size)
    key_begin = Vector{Int64}(length(dims))
    fill!(key_begin, 1)
    key_begin_index = Vector{Int64}(length(dims))
    fill!(key_begin_index, 1)
    for i = 1:num_reads
        for j = read_dims_begin:length(dims)
            index_this_dim = key_begin_index[j]
            if index[j] == Colon()
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
        key_begin_int64 = from_keys_to_int64(key_begin, dims)
        ccall((:orion_dist_array_read, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
              access_ptr, key_begin_int64, per_read_size, read_array)
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

function setindex!(dist_array::DistArray,
                  values,
                  index...)
    num_dist_array_dims = length(index)
    array_size = length(values)
    write_dims = Vector{Int64}()
    dims = dist_array.dims
    ValueType = dist_array.ValueType
    value_array = reshape(values, (array_size,))
    access_ptr = dist_array.access_ptr

    if length(index) == 1
        value_array = Vector{ValueType}(1)
        ccall((:orion_dist_array_write, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
              access_ptr, index[1], 1, values)
    end

    per_write_size = 1
    num_dims_per_write = 0
    for i in eachindex(index)
        dim_i = index[i]
        if dim_i != :
            break
        end
        per_write_size *= dims[i]
        num_dims_per_write += 1
    end

    write_dims_begin = num_dims_per_write + 1
    write_offset = 1

    num_writes_by_dim = Vector{Int64}()
    for i = write_dims_begin:length(dims)
        dim_i = index[i]
        if dim_i == :
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
            if index[j] == Colon()
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
        key_begin_int64 = from_keys_to_int64(key_begin, dims)
        ccall((:orion_dist_array_write, lib_path),
              Void, (Ptr{Void}, Int64, UInt64, Ptr{Void}),
              access_ptr, key_begin_int64, per_write_size, write_array)
        if write_dims_begin <= length(dims)
            key_begin_index[write_dims_begin] += 1
            j = write_dims_begin
            while j < length(dims)
                if key_begin_index[j] == (num_writes_by_dim[j - num_dims_per_write] + 1)
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

function create_dist_array_for_access(ValueType::DataType,
                                      symbol::AbstractString,
                                      dims::Vector,
                                      is_dense::Bool,
                                      access_ptr)
    dist_array = DistArray{ValueType}()
    dist_array.symbol = symbol
    dist_array.dims = dims
    dist_array.num_dims = length(dims)
    dist_array.is_dense = is_dense
    dist_array.access_ptr = access_ptr
    return dist_array
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

function from_keys_to_int64(key, dims::Vector{Int64})
    key_int = 0
    for i = length(dims):-1:2
        key_int += key[i] - 1
        key_int *= dims[i - 1]
    end
    key_int += key[1] - 1
    return key_int
end
