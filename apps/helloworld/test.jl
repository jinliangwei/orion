macro test(ast)
    println("macro called, a = ", a)
    return ast
end

a = 10

module MyTest

function my_func()
    Main.@test println("good!")
end
end


println("call my func")
MyTest.my_func()
