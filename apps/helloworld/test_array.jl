struct MyArray{T, N} <: AbstractArray{T, N}
    values::Vector{T}
    dims::NTuple{N, Int64}
    MyArray{T, N}(dims::NTuple{N, Int64}) where {T, N} = length(dims) > 0 ?
        new(randn(reduce(*, [dims...])), dims) :
        new(Vector{T}(), dims)
end

Base.IndexStyle(::MyArray) = IndexLinear()

function Base.size(array::MyArray)
    return array.dims
end

function Base.getindex(array::MyArray, i::Int)
    println(i)
    return array.values[i]
end

function Base.setindex!(array::MyArray, v, i::Int)
    array.values[i] = v
end

function Base.similar{T}(array::MyArray, ::Type{T}, dims::Dims)
    return MyArray{T, length(dims)}(dims)
end

my_array_A = MyArray{Float64, 2}((100, 10000))
my_array_B = MyArray{Float64, 2}((100, 10000))

function test_my_array(test_arr_A,
                       test_arr_B)
    sum = zeros(100)
    @time for i = 1:10000
        row = @view test_arr[:, i]
        sum .= sum + row
    end
    println(sum)
end


@code_warntype test_my_array(my_array)
