include("/home/ubuntu/orion/src/julia/orion.jl")

function visit(ast::Any,
               cbdata,
               top_level_num,
               is_top_level,
               read)
    println("hello world")
    return Orion.AstWalk.AST_WALK_RECURSE
end

macro ast_walk(ast)
    Orion.AstWalk.ast_walk(ast, visit, 1)
end

@ast_walk a = 1
