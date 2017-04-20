symbol_counter = 0
function generate_symbol()::String
    symbol_counter += 1
    return "orion_sym_" * string(symbol_counter)
end

function share_generated(ex::Expr)
    assert(ex.head == :function)
    eval_expr_on_all(ex, Void, OrionGenerated)
end

type VarInfo
    ValueType::DataType
    value
    shared::Bool
    is_accumulator::Bool

    VarInfo(ValueType::DataType,
            value) = new(ValueType,
                         value,
                         false,
                         false)
end

type Context
    # the list global variables that Orion depends on
    # writes by the non-parallel part of the loop will be propogated
    # outside of the scope; writes by the parallel part will not
    global_var::Dict{Symbol, VarInfo}
    local_var::Dict{Symbol, VarInfo}
    Context() = new(Dict{Symbol, VarInfo}(),
                    Dict{Symbol, VarInfo}())
end

context = Context()

macro share(ex::Expr)
    if ex.head == :function
        eval_expr_on_all(ex, Void, Main)
    else
        error("Do not support sharing Expr of this kind")
    end
    esc(ex)
end


macro iterative(expr::Expr)
    if expr.head == :for
        println("analyzing for-loop")
        parallelize_iterative_loop(expr)
    elseif expr.head == :function
        parallelize_iterative_func(expr)
    else
        error("Expression ", expr.head, " cannot be parallelized (yet)")
    end
end

function parallelize_iterative_loop(expr::Expr)
    println("here")
    curr_module = current_module()
    iterative_loop_args = expr.args
    @assert expr.args[1].head == :(=)
    iteration_index = expr.args[1].args[1]
    iterative_body = quote
    end
    push!(iterative_body.args, Expr(:call, :println, esc(iteration_index)))
    ret = quote
    end

    for stmt in expr.args[2].args
        dump(stmt)
        translated = translate_iterative_stmt(stmt, context)
        println(translated)
#        if translated == nothing
#            continue
#        else
#            push!(iterative_body.args, translated)
#        end
    end

    println(ret)

    push!(ret.args,
          Expr(expr.head,
               esc(expr.args[1]),
               iterative_body
               )
          )
    return ret
end

function parallelize_iterative_func(expr::Expr)
    return :(assert(false))
end

macro accumulator(expr::Expr)
end

macro parallel_for(expr::Expr)
end
