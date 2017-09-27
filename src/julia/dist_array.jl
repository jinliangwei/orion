import Base
import Base.copy

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
    dims::Array{Int64, 1}
    random_init_type::DataType
    is_dense::Bool

    DistArray(id::Integer,
              parent_type::DistArrayParentType,
              flatten_results::Bool,
              map_type::DistArrayMapType,
              num_dims::Integer,
              ValueType::DataType,
              file_path::String,
              parent_id::Integer,
              init_type::DistArrayInitType,
              mapper_func_module::Module,
              mapper_func_name::String,
              is_materialized::Bool,
              random_init_type::DataType,
              is_dense::Bool) = new(id,
                                    parent_type,
                                    flatten_results,
                                    map_type,
                                    num_dims,
                                    ValueType,
                                    file_path,
                                    parent_id,
                                    init_type,
                                    mapper_func_module,
                                    mapper_func_name,
                                    is_materialized,
                                    zeros(Int64, num_dims),
                                    random_init_type,
                                    is_dense)

    DistArray() = new(-1,
                      DistArrayParentType_init,
                      false,
                      DistArrayMapType_no_map,
                      0,
                     Float32,
                      "",
                      -1,
                      DistArrayInitType_empty,
                      Main,
                      "",
                      false,
                      zeros(Int64, 0),
                      Void,
                      false)
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
        ValueType,
        file_path,
        -1,
        DistArrayInitType_empty,
        parser_func_module,
        parser_func_name,
        false,
        Void,
        is_dense)
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
        ValueType,
        "",
        -1,
        DistArrayInitType_uniform_random,
        Module(),
        "",
        false,
        ValueType,
        true)
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
        ValueType,
        "",
        -1,
        DistArrayInitType_normal_random,
        Module(),
        "",
        false,
        ValueType,
        true)
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
                 Cstring, Int32, Int32, Int32, Cstring, Ptr{Int64}, Int32),
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
          data_type_to_int32(dist_array_to_create.random_init_type))
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
        println("here ", typeof(origin_dist_array.mapper_func_name))

        insert!(map_func_names, 1, origin_dist_array.mapper_func_name)
        insert!(map_func_modules, 1, origin_dist_array.mapper_func_module)
        insert!(map_types, 1, origin_dist_array.map_type)
        insert!(map_flattens, 1, origin_dist_array.flatten_results)
        parent_id = origin_dist_array.parent_id
        origin_dist_array = dist_arrays[parent_id]
    end

    map_func_name_sym = gen_unique_symbol()
    processed_dist_array.mapper_func_name = string(map_func_name_sym)
    processed_dist_array.mapper_func_module = Module(:OrionGen)
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
        eval_expr_on_all(generated_map_func, :OrionGen)
    end
    return processed_dist_array
end

function space_time_repartition(dist_array::DistArray,
                     partition_func_name::AbstractString)
    ccall((:orion_space_time_repartition_dist_array, lib_path),
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
        parent_dist_array.ValueType,
        "",
        parent_dist_array.id,
        DistArrayInitType_empty,
        map_func_module,
        map_func_name,
        false,
        parent_dist_array.random_init_type,
        parent_dist_array.is_dense)
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
