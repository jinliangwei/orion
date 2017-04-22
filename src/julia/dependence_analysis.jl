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
    value::Any
    assigned_to::Bool
    modified::Bool
    is_accumulator::Bool
    marked_local::Bool

    VarInfo(ValueType::DataType,
            value) = new(ValueType,
                         value,
                         false,
                         false,
                         false,
                         false)

    VarInfo() = new(Any,
                    nothing,
                    false,
                    false,
                    false,
                    false)
end

type ParForContext
    iteration_var::Symbol
    iteration_space::Symbol
    iteration_index::Array{Symbol}
    ParForContext(iteration_var::Symbol,
                  iteration_space::Symbol) = new(iteration_var,
                                         iteration_space,
                                         Array{Symbol, 1}())
end

type ScopeContext
    inherited_var::Dict{Symbol, VarInfo}
    local_var::Dict{Symbol, VarInfo}
    par_for_scope::Array{ScopeContext}
    child_scope::Array{ScopeContext}

    par_for_context::Array{ParForContext}
    ScopeContext() = new(Dict{Symbol, VarInfo}(),
                         Dict{Symbol, VarInfo}(),
                         Array{ScopeContext, 1}(),
                         Array{ScopeContext, 1}(),
                         Array{ParForContext, 1}())
end

function pretty_print(scope_context::ScopeContext, indent = 0)
    indent_str = " " ^ indent
    println(indent_str + "inherited:")
    in_indent_str = " " ^ (indent + 1)
    for (var, info) in scope_context.inherited_var
        print(in_indent_str, var, " ", info)
    end
    println(indent_str + "local:")
    for (var, info) in scope_context.local_var
        print(in_indent_str, var, " ", info)
    end
    println(indent_str + "child scope:")
    for scope in scope_context.child_scope
        pretty_print(scope, indent + 2)
    end

    println(indent_str + "par_for scope:")
    for scope in scope_context.par_for_scope
        pretty_print(scope, indent + 2)
    end
end

function add_inherited_var!(scope_context::ScopeContext,
                            var::Symbol,
                            info::VarInfo)
    if var in keys(scope_context.local_var)
        if info.modified
            scope_context.local_var[var].modified = true
        end
    elseif var in keys(scope_context.inherited_var)
        if info.modified
            scope_context.inherited_var[var].modified = true
        end
    else
        scope_context.inherited_var[var] = info
    end
end

function add_local_var!(scope_context::ScopeContext,
                        var::Symbol,
                        info::VarInfo)
    if var in keys(scope_context.inherited_var)
        if scope_context.inherited_var[var].modified
            info.modified = true
        end
        delete!(scope_context.inherited_var, var)
        scope_context.local_var[var] = info
    elseif var in keys(scope_context.local_var)
        scope_context.local_var[var].assigned_to = info.assigned_to
        scope_context.local_var[var].modified = info.modified
        scope_context.local_var[var].is_accumulator = info.is_accumulator
        scope_context.local_var[var].marked_local = info.marked_local
    else
        scope_context.local_var[var] = info
    end
end

function merge_scope!(dst::ScopeContext,
                      src::ScopeContext)
    for (var, info) in src.inherited_var
        @assert !info.assigned_to &&
            !info.is_accumulator &&
            !info.marked_local
        add_inherited_var!(dst, var, info)
    end
    for (var, info) in src.local_var
        add_local_var!(dst, var, info)
    end
    for scope in src.par_for_scope
        push!(dst.par_for_scope, scope)
    end
    for scope in src.child_scope
        push!(dst.child_scope, scope)
    end
    for context in src.par_for_context
        push!(dst.par_for_context, context)
    end
end

macro share(ex::Expr)
    if ex.head == :function
        eval_expr_on_all(ex, Void, Main)
    else
        error("Do not support sharing Expr of this kind")
    end
    esc(ex)
end


macro transform(expr::Expr)
    if expr.head == :for
        context = ScopeContext()
        transform_loop(expr, context)
    elseif expr.head == :function
        transform_func(expr)
    else
        error("Expression ", expr.head, " cannot be parallelized (yet)")
    end
end

function transform_loop(expr::Expr, context::ScopeContext)
    curr_module = current_module()
    iterative_loop_args = expr.args
    @assert expr.args[1].head == :(=)
    iteration_index = expr.args[1].args[1]
    iterative_body = quote
    end
    push!(iterative_body.args, Expr(:call, :println, esc(iteration_index)))
    ret = quote
    end

    scope_context = ScopeContext()
    for stmt in expr.args[2].args
        println(stmt)
        get_vars!(scope_context, stmt)
#        dump(stmt)
#        translated = translate_stmt(stmt, context)
#        println(translated)
#        if translated == nothing
#            continue
#        else
#            push!(iterative_body.args, translated)
#        end
    end
    pretty_print(scope_context)
    println(ret)

    push!(ret.args,
          Expr(expr.head,
               esc(expr.args[1]),
               iterative_body
               )
          )
    return ret
end

function transform_func(expr::Expr)
    return :(assert(false))
end

macro accumulator(expr::Expr)
end

macro parallel_for(expr::Expr)
end
