module AstWalk

import ..DebugMsg
DebugMsg.init()
# debug messages set below this level will be printed
set_debug_level(3)

abstract AST_WALK_DIDNT_MODIFY

immutable AST_WALK_RECURSE <: AST_WALK_DIDNT_MODIFY
end

immutable AST_WALK_RECURSE_DUPLICATE <: AST_WALK_DIDNT_MODIFY
end

# Return this node to indicate that AstWalk should remove the current node from
# the container in which it resides.
immutable AST_WALK_REMOVE
end

export AstWalk, AST_WALK_RECURSE, AST_WALK_REMOVE, AST_WALK_RECURSE_DUPLICATE

"""
!! Mimicing Intel's CompilerTools
Entry point into the code to perform an AST walk.
This function will cause every node in the AST to be visited and a callback invoked on each one.
The callback can record information about nodes it sees through the opaque cbdata parameter.
The callback can also cause the current node to be replaced in the AST by returning a new AST
node.  If the callback doesn't wish to change the node then it returns AST_WALK_RECURSE which causes
AstWalk to recursively process the sub-trees under the current AST node.  If you want to modify the
current AST node and want the sub-trees of that AST node to be processed first then you manually have
to recursively call AstWalk on each one.  There are some cases where you don't want to recursively
process the sub-trees first and so this recursive process has to be left up to the user.
You generally pass a lambda expression as the first argument although any AST node is acceptable.
The third argument is an object that is opaque to AstWalk but that is passed to every callback.
You can use this object to collect data about the AST as it is walked or to hold information on
how to change the AST as you are walking over it.
The second argument is a callback function.  For each AST node, AstWalk will invoke this callback.
The signature of the callback must be (Any, Any, Int64, Bool, Bool).  The arguments to the callback
are as follows:
    1) The current node of the AST being walked.
    2) The callback data object that you originally passed as the first argument to AstWalk.
    3) Specifies the index of the body's statement that is currently being processed.
    4) True if the current AST node being walked is the root of a top-level statement, false if the AST node is a sub-tree of a top-level statement.
    5) True if the AST node is being read, false if it is being written.
"""

function ast_walk(ast::Any, callback::Function, cbdata::Any)
    @dprintln(0, "ast_walk called")
    from_expr(ast, 1, callback, cbdata, 0, false, true)
end

function from_expr(ast::Any,
                   depth::Integer,
                   callback::Function,
                   cbdata::Any,
                   top_level_number,
                   is_top_level::Bool,
                   read::Bool)
    @dprintln(2, "from_expr depth = ", depth, ",  AST: ", ast)
    # For each AST node, we first call the user-provided callback to see if they
    # want to do something with the node.
    # this is the only place where callback is called
    ret = callback(ast, cbdata, top_level_number, is_top_level, read)
    @dprintln(2, "callback ret = ", ret)
    if ret != AST_WALK_RECURSE && ret != AST_WALK_RECURSE_DUPLICATE
        return ret
    end

    # The user callback didn't replace the AST node so recursively process it.
    # We have a different from_expr_helper that is accurately typed for each
    # possible AST node type.
    ast = from_expr_helper(ast, depth, callback, cbdata,
                           top_level_number, is_top_level, read)
    if ret == AST_WALK_RECURSE_DUPLICATE
        ast = copy(ast)
    end

    @dprintln(3, "Before return for ", ast)
    return ast
end

function from_expr_helper(ast::Expr,
                          depth::Integer,
                          callback::Function,
                          cbdata::Any,
                          top_level_number,
                          is_top_level::Bool,
                          read::Bool)
    # If you have an assignment with an Expr as its left-hand side
    # then you get here with "read = false"
    # but that doesn't  mean the whole Expr is written.  In fact, none of it
    # is written so we set read
    # back to true and then restore in the incoming read value at the end.
    read_save = read
    read = true
    #@dprintln(2, "Expr ")
    head = ast.head
    args = ast.args
    typ  = ast.typ
    #@dprintln(2, " ", args)
    if head == :macrocall
        for i = 1:length(args)
            args[i] = from_expr(args[i], depth, callback, cbdata,
                                top_level_number, false, read)
        end
    elseif head == :for
        for i = 1:length(args)
            args[i] = from_expr(args[i], depth, callback, cbdata,
                                top_level_number, false, read)
        end
    elseif head == :(=) ||
        head == :(.=) ||
        head == :(+=) ||
        head == :(-=) ||
        head == :(*=) ||
        head == :(/=)
        args[1] = from_expr(args[1], depth, callback, cbdata, top_level_number,
                            false, false)
        args[2] = from_expr(args[2], depth, callback, cbdata, top_level_number,
                            false, read)
    elseif head == :line
    end
end
end
