function set_lib_path(path::AbstractString)
    global const lib_path = path
    #ccall((:orion_helloworld, "/home/ubuntu/orion/lib/liborion.so"), Void, ())
end

function load_constants()
    load_module_int32()
    load_type_int32()
    load_dist_array_parent_type_int32()
    load_dist_array_init_type_int32()
    load_dist_array_map_type_int32()
    load_dist_array_partition_scheme_int32()
    load_dist_array_index_type_int32()
    load_for_loop_parallel_scheme_int32()
end

function load_module_int32()
    ptr_val = cglobal((:ORION_JULIA_MODULE_CORE, lib_path), Int32)
    global const module_core_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_JULIA_MODULE_BASE, lib_path), Int32)
    global const module_base_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_JULIA_MODULE_MAIN, lib_path), Int32)
    global const module_main_int32 = unsafe_load(ptr_val)
end

function load_type_int32()
    ptr_val = cglobal((:ORION_TYPE_VOID, lib_path), Int32)
    global const type_void_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_TYPE_INT8, lib_path), Int32)
    global const type_int8_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_TYPE_UINT8, lib_path), Int32)
    global const type_uint8_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_TYPE_INT16, lib_path), Int32)
    global const type_int16_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_TYPE_UINT16, lib_path), Int32)
    global const type_uint16_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_TYPE_INT32, lib_path), Int32)
    global const type_int32_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_TYPE_UINT32, lib_path), Int32)
    global const type_uint32_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_TYPE_INT64, lib_path), Int32)
    global const type_int64_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_TYPE_UINT64, lib_path), Int32)
    global const type_uint64_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_TYPE_FLOAT32, lib_path), Int32)
    global const type_float32_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_TYPE_FLOAT64, lib_path), Int32)
    global const type_float64_int32 = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_TYPE_STRING, lib_path), Int32)
    global const type_string_int32 = unsafe_load(ptr_val)
end

function load_dist_array_parent_type_int32()
    ptr_val = cglobal((:ORION_DIST_ARRAY_PARENT_TYPE_TEXT_FILE, lib_path), Int32)
    global const dist_array_parent_type_text_file = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_PARENT_TYPE_DIST_ARRAY, lib_path), Int32)
    global const dist_array_parent_type_dist_array = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_PARENT_TYPE_INIT, lib_path), Int32)
    global const dist_array_parent_type_init = unsafe_load(ptr_val)
end

function load_dist_array_init_type_int32()
    ptr_val = cglobal((:ORION_DIST_ARRAY_INIT_TYPE_EMPTY, lib_path), Int32)
    global const dist_array_init_type_empty = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_INIT_TYPE_UNIFORM_RANDOM, lib_path), Int32)
    global const dist_array_init_type_uniform_random = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_INIT_TYPE_NORMAL_RANDOM, lib_path), Int32)
    global const dist_array_init_type_normal_random = unsafe_load(ptr_val)
end

function load_dist_array_map_type_int32()
    ptr_val = cglobal((:ORION_DIST_ARRAY_MAP_TYPE_NO_MAP, lib_path), Int32)
    global const dist_array_map_type_no_map = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_MAP_TYPE_MAP, lib_path), Int32)
    global const dist_array_map_type_map = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_MAP_TYPE_MAP_FIXED_KEYS, lib_path), Int32)
    global const dist_array_map_type_fixed_keys = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_MAP_TYPE_MAP_VALUES, lib_path), Int32)
    global const dist_array_map_type_map_values = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_MAP_TYPE_MAP_VALUES_NEW_KEYS, lib_path), Int32)
    global const dist_array_map_type_map_values_new_keys = unsafe_load(ptr_val)
end

function load_dist_array_partition_scheme_int32()
    ptr_val = cglobal((:ORION_DIST_ARRAY_PARTITION_SCHEME_NAIVE, lib_path), Int32)
    global const dist_array_partition_scheme_naive = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_PARTITION_SCHEME_SPACE_TIME, lib_path), Int32)
    global const dist_array_partition_scheme_space_time = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_PARTITION_SCHEME_1D, lib_path), Int32)
    global const dist_array_partition_scheme_1d = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_PARTITION_SCHEME_HASH_SERVER, lib_path), Int32)
    global const dist_array_partition_scheme_hash_server = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_PARTITION_SCHEME_HASH_EXECUTOR, lib_path), Int32)
    global const dist_array_partition_scheme_hash_executor = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_PARTITION_SCHEME_RANGE, lib_path), Int32)
    global const dist_array_partition_scheme_range = unsafe_load(ptr_val)
end

function load_dist_array_index_type_int32()
    ptr_val = cglobal((:ORION_DIST_ARRAY_INDEX_TYPE_NONE, lib_path), Int32)
    global const dist_array_index_type_none = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_DIST_ARRAY_INDEX_TYPE_RANGE, lib_path), Int32)
    global const dist_array_index_type_range = unsafe_load(ptr_val)
end

function load_for_loop_parallel_scheme_int32()
    ptr_val = cglobal((:ORION_FOR_LOOP_PARALLEL_SCHEME_1D, lib_path), Int32)
    global const for_loop_parallel_scheme_1d = unsafe_load(ptr_val)
    ptr_val = cglobal((:ORION_FOR_LOOP_PARALLEL_SCHEME_SPACE_TIME, lib_path), Int32)
    global const for_loop_parallel_scheme_space_time = unsafe_load(ptr_val)
end

function dist_array_parent_type_to_int32(parent_type::DistArrayParentType)::Int32
    if parent_type == DistArrayParentType_text_file
        return dist_array_parent_type_text_file
    elseif parent_type == DistArrayParentType_dist_array
        return dist_array_parent_type_dist_array
    elseif parent_type == DistArrayParentType_init
        return dist_array_parent_type_init
    else
        error("Unknown ", parent_type)
    end
    return -1
end

function dist_array_init_type_to_int32(init_type::DistArrayInitType)::Int32
    if init_type == DistArrayInitType_empty
        return dist_array_init_type_empty
    elseif init_type == DistArrayInitType_uniform_random
        return dist_array_init_type_uniform_random
    elseif init_type == DistArrayInitType_normal_random
        return dist_array_init_type_normal_random
    else
        error("Unknown ", init_type)
    end
    return -1
end

function dist_array_map_type_to_int32(map_type::DistArrayMapType)::Int32
    if map_type == DistArrayMapType_no_map
        return dist_array_map_type_no_map
    elseif map_type == DistArrayMapType_map
        return dist_array_map_type_map
    elseif map_type == DistArrayMapType_map_fixed_keys
        return dist_array_map_type_fixed_keys
    elseif map_type == DistArrayMapType_map_values
        return dist_array_map_type_map_values
    elseif map_type == DistArrayMapType_map_values_new_keys
        return dist_array_map_type_map_values_new_keys
    else
        error("unknown ", map_type)
    end
    return -1
end

function dist_array_partition_type_to_int32(partition_type::DistArrayPartitionType)
    if partition_type == DistArrayPartitionType_naive
        return dist_array_partition_scheme_naive
    elseif partition_type == DistArrayPartitionType_1d
        return dist_array_partition_scheme_1d
    elseif partition_type == DistArrayPartitionType_2d ||
        partition_type == DistArrayPartitionType_2d_unimodular
        return dist_array_partition_scheme_space_time
    elseif partition_type == DistArrayPartitionType_range
        return dist_array_partition_scheme_range
    elseif partition_type == DistArrayPartitionType_hash_server
        return dist_array_partition_scheme_hash_server
    elseif partition_type == DistArrayPartitionType_hash_executor
        return dist_array_partition_scheme_hash_executor
    else
        error("unknown ", partition_type)
    end
    return -1
end

function dist_array_index_type_to_int32(index_type::DistArrayIndexType)
    if index_type == DistArrayIndexType_none
        return dist_array_index_type_none
    elseif index_type == DistArrayIndexType_range
        return dist_array_index_type_range
    else
        error("unknown ", index_type)
    end
    return -1
end

function for_loop_parallel_scheme_to_int32(parallel_scheme::ForLoopParallelScheme)
    if parallel_scheme == ForLoopParallelScheme_naive ||
        parallel_scheme == ForLoopParallelScheme_1d
        return for_loop_parallel_scheme_1d
    elseif parallel_scheme == ForLoopParallelScheme_2d ||
        parallel_scheme == ForLoopParallelScheme_unimodular
        return for_loop_parallel_scheme_space_time
    else
        error("unknown ", parallel_scheme)
    end
    return -1
end

function module_to_int32(m::Symbol)::Int32
    if m == :Core
        return module_core_int32
    elseif m == :Base
        return module_base_int32
    elseif m == :Main
        return module_main_int32
    end
    return -1
end

function data_type_to_int32(ResultType::DataType)::Int32
    if ResultType == Void
        return type_void_int32
    elseif ResultType == Int8
        return type_int8_int32
    elseif ResultType == UInt8
        return type_uint8_int32
    elseif ResultType == Int16
        return type_int16_int32
    elseif ResultType == UInt16
        return type_uint16_int32
    elseif ResultType == Int32
        return type_int32_int32
    elseif ResultType == UInt32
        return type_uint32_int32
    elseif ResultType == Int64
        return type_int64_int32
    elseif ResultType == UInt64
        return type_uint64_int32
    elseif ResultType == Float32
        return type_float32_int32
    elseif ResultType == Float64
        return type_float64_int32
    elseif ResultType == String
        return type_string_int32
    else
        return type_void_int32
    end
end

function int32_to_data_type(data_type::Int32)::DataType
    if data_type == type_void_int32
        return Void
    elseif data_type == type_int8_int32
        return Int8
    elseif data_type == type_uint8_int32
        return UInt8
    elseif data_type == type_int16_int32
        return Int16
    elseif data_type == type_uint16_int32
        return UInt16
    elseif data_type == type_int32_int32
        return Int32
    elseif data_type == type_uint32_int32
        return UInt32
    elseif data_type == type_int64_int32
        return Int64
    elseif data_type == type_uint64_int32
        return UInt64
    elseif data_type == type_float32_int32
        return Float32
    elseif data_type == type_float64_int32
        return Float64
    elseif data_type == type_string_int32
        return String
    end
    return Void
end
