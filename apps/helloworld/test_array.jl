module Test
import Main: size, getindex
export MyArray, size, getindex
type MyArray{T} <: AbstractArray{T}
    value::Int32
    ValueType::DataType
    MyArray(value) = new(value,
                         T)
end

function size(a::MyArray)
    return 10
end

function getindex(a::MyArray, i::Int)
    println("i = ", i)
    return a.value
end

function getindex(a::MyArray, I...)
    println("I = ", typeof(I))
    return a.value
end

end

using Test

a = MyArray{Int32}(2)
c = a[10]
println(a.ValueType)

println(c)

d = a[1, 2]
println(d)

t = Vector{Int32}(0)
push!(t, 10)
println(t[1])

function print_sym(s, t)
    println(typeof(s), " ", s, " value = ", t)
end

macro print_sym(expr)
    sym_expr = Expr(expr)
    println(sym_expr)
    return :(print_sym($expr, :($expr)))
    #return :(println("good"))
end

g = 10

@print_sym g
