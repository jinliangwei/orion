module AstWalk

import ..DebugMsg
DebugMsg.init()
# debug messages set below this level will be printed
set_debug_level(3)

abstract type AST_WALK_DIDNT_MODIFY end

struct AST_WALK_RECURSE <: AST_WALK_DIDNT_MODIFY
end

struct AST_WALK_RECURSE_DUPLICATE <: AST_WALK_DIDNT_MODIFY
end

# Return this node to indicate that AstWalk should remove the current node from
# the container in which it resides.
struct AST_WALK_REMOVE
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
The signature of the callback must be (Any, Any).  The arguments to the callback
are as follows:
    1) The current node of the AST being walked.
    2) The callback data object that you originally passed as the first argument to AstWalk.
"""

function ast_walk(ast::Any, callback::Function, cbdata::Any)
    @dprintln(0, "ast_walk called")
    from_expr(ast, callback, cbdata)
end

function from_expr(ast::Any,
                   callback::Function,
                   cbdata::Any)
    @dprintln(2, "AST: ", ast)
    # For each AST node, we first call the user-provided callback to see if they
    # want to do something with the node.
    # this is the only place where callback is called
    ret = callback(ast, cbdata)
    @dprintln(2, "callback ret = ", ret)
    if ret != AST_WALK_RECURSE && ret != AST_WALK_RECURSE_DUPLICATE
        return ret
    end

    # The user callback didn't replace the AST node so recursively process it.
    # We have a different from_expr_helper that is accurately typed for each
    # possible AST node type.
    ast = from_expr_helper(ast, callback, cbdata)
    if ret == AST_WALK_RECURSE_DUPLICATE
        ast = copy(ast)
    end

    @dprintln(3, "Before return for ", ast)
    return ast
end

function from_expr_helper(ast::Expr,
                          callback::Function,
                          cbdata::Any)
    # If you have an assignment with an Expr as its left-hand side
    # then you get here with "read = false"
    # but that doesn't  mean the whole Expr is written.  In fact, none of it
    # is written so we set read
    # back to true and then restore in the incoming read value at the end.
    #@dprintln(2, "Expr ")
    head = ast.head
    args = ast.args
    typ  = ast.typ
    #@dprintln(2, " ", args)
    if head == :macrocall
        for i = 1:length(args)
            args[i] = from_expr(args[i], callback, cbdata)
        end
    elseif head == :for
        for i = 1:length(args)
            args[i] = from_expr(args[i], callback, cbdata)
        end
    elseif head == :if
        for i = 1:length(args)
            args[i] = from_expr(args[i], callback, cbdata)
        end
    elseif head in Set([:(=), :(.=), :(+=), :(/=), :(*=), :(-=)])
        args[1] = from_expr(args[1], callback, cbdata)
        args[2] = from_expr(args[2], callback, cbdata)
    elseif head == :block
        args = from_exprs(args, callback, cbdata)
    elseif head == :return
        args = from_exprs(args, callback, cbdata)
    elseif head == :invoke
        args = from_call(args, callback, cbdata)
    elseif head == :call
        args = from_call(args, callback, cbdata)
    elseif head == :call1
        args = from_call(args, callback, cbdata)
    elseif head == :foreigncall
        args = from_call(args, callback, cbdata)
    elseif head == :line
        # skip
    elseif head == :ref
        for i = 1:length(args)
            args[i] = from_expr(args[i], callback, cbdata)
        end
    elseif head == :(:)
        # args are either Expr or Symbol
        for i = 1:length(args)
            args[i] = from_expr(args[i], callback, cbdata)
        end
    elseif head == :(.)
        args = from_exprs(args, callback, cbdata)
    end
    ast.head = head
    ast.args = args
    ast.typ = typ
    return ast
end

function from_exprs(ast, callback, cbdata::Any)
    len = length(ast)
    top_level = false
    body = Vector{Any}()
    for i = 1:len
        new_expr = from_expr(ast[i], callback, cbdata)
        if new_expr != AST_WALK_REMOVE
            push!(body, new_expr)
        end
    end
    return body
end

function from_call(ast::Array{Any,1}, callback, cbdata)
  assert(length(ast) >= 1)
  # A call is a function followed by its arguments.  Extract these parts below.
  fun  = ast[1]
  args = ast[2:end]
  @dprintln(2,"from_call fun = ", fun, " typeof fun = ", typeof(fun))
  if length(args) > 0
    @dprintln(2,"first arg = ",args[1], " type = ", typeof(args[1]))
  end
  # Symbols don't need to be translated.
  # if typeof(fun) != Symbol
      # I suppose this "if" could be wrong.  If you wanted to replace all "x" functions with "y" then you'd need this wouldn't you?
  #    fun = from_expr(fun, callback, cbdata, top_level_number, false, read)
  # end
  # Process the arguments to the function recursively.
  args = from_exprs(ast, callback, cbdata)

  return args
end

# The following are for non-Expr AST nodes are generally leaf nodes of the AST where no
# recursive processing is possible.
function from_expr_helper(ast::Symbol,
                          callback,
                          cbdata::Any)
    @dprintln(2, typeof(ast), " type")
    # Intentionally do nothing.
    return ast
end

function from_expr_helper(ast::NewvarNode,
                          callback,
                          cbdata::Any)
    return NewvarNode(from_expr(ast.slot, callback, cbdata))
end

function from_expr_helper(ast::Union{LineNumberNode,LabelNode,GotoNode,DataType,AbstractString,Void,Function,Module},
                          callback,
                          cbdata::Any)
    # Intentionally do nothing.
    return ast
end

function from_expr_helper(ast::Tuple,
                          callback,
                          cbdata::Any)
    # N.B. This also handles the empty tuple correctly.

    new_tt = Expr(:tuple)
    for i = 1:length(ast)
        push!(new_tt.args, from_expr(ast[i], callback, cbdata))
    end
    new_tt.typ = typeof(ast)
    ast = eval(new_tt)

    return ast
end

function from_expr_helper(ast::QuoteNode,
                          callback,
                          cbdata::Any)
    value = ast.value
    #TODO: fields: value
    @dprintln(2,"QuoteNode type ",typeof(value))

    return ast
end

function from_expr_helper(ast::SimpleVector,
                          callback,
                          cbdata::Any)

    new_values = [from_expr(ast[i], callback, cbdata) for i = 1:length(ast)]
    return Core.svec(new_values...)
end

"""
The catchall function to process other kinds of AST nodes.
"""
function from_expr_helper(ast::Any,
                          callback,
                          cbdata::Any)
    asttyp = typeof(ast)

    if isdefined(:GetfieldNode) && asttyp == GetfieldNode  # GetfieldNode = value + name
        @dprintln(2,"GetfieldNode type ",typeof(ast.value), " ", ast)
    elseif isdefined(:GlobalRef) && asttyp == GlobalRef
        @dprintln(2,"GlobalRef type ",typeof(ast.mod), " ", ast)  # GlobalRef = mod + name
    elseif isbits(asttyp)
        #skip
    elseif isa(asttyp, CodeInfo)
        #skip
    else
        println(ast, "ast = ", ast, " type = ", typeof(ast))
        throw(string("from_expr: unknown AST (", typeof(ast), ",", ast, ")"))
    end

    return ast
end

end
