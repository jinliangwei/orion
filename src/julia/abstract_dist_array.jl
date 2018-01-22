type DistArray{T} <: AbstractDistArray{T}







    access_ptr
    iterate_dims::Vector{Int64}


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
