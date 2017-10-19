module OrionWorker

include("src/julia/dist_array.jl")
include("src/julia/constants.jl")

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
end

end

using OrionWorker

function orionres_define_dist_array(ValueType::DataType,
                                    symbol::AbstractString,
                                    dims::Vector{Int64},
                                    is_dense::Bool,
                                    access_ptr)
    dist_array = OrionWorker.create_dist_array_for_access(ValueType,
                                                          symbol,
                                                          dims,
                                                          is_dense,
                                                          access_ptr)
    dist_array_symbol = Symbol(symbol)
    eval(:(global $dist_array_symbol = $dist_array))
end
