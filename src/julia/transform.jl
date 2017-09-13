import Base.print, Base.string

function share_generated(ex::Expr)
    assert(ex.head == :function)
    eval_expr_on_all(ex, Void, OrionGenerated)
end

# describe the information of a variable for code within a scope
# the only exception is is_accumulator, which is defined by the code
# within the code and the code in its ancestors' scope before it
type VarInfo
    is_assigned_to::Bool
    is_modified::Bool
    is_accumulator::Bool
    is_marked_local::Bool
    is_marked_global::Bool

    VarInfo() = new(false,
                    false,
                    false,
                    false,
                    false)
end

function string(info::VarInfo)
    str = "VarInfo{"
    str *= "[is_assigned_to=" * string(info.is_assigned_to) * "], "
    str *= "[is_modified=" * string(info.is_modified) * "], "
    str *= "[is_accumulator=" * string(info.is_accumulator) * "],"
    str *= "[is_marked_local=" * string(info.is_marked_local) * "]}"
    str *= "[is_marked_global=" * string(info.is_marked_global) * "]}"
#    println(str)
end

type ParForContext
    iteration_var::Symbol
    iteration_space::Symbol
    iteration_index::Array{Symbol}
    loop_stmt::Expr
    ParForContext(iteration_var::Symbol,
                  iteration_space::Symbol,
                  loop_stmt) = new(iteration_var,
                                   iteration_space,
                                   Array{Symbol, 1}(),
                                   loop_stmt)
end

type ScopeContext
    parent_scope
    is_hard_scope::Bool
    inherited_var::Dict{Symbol, VarInfo}
    local_var::Dict{Symbol, VarInfo}
    par_for_scope::Array{ScopeContext}
    child_scope::Array{ScopeContext}
    par_for_context::Array{ParForContext}

    ScopeContext() = new(nothing,
                         false,
                         Dict{Symbol, VarInfo}(),
                         Dict{Symbol, VarInfo}(),
                         Array{ScopeContext, 1}(),
                         Array{ScopeContext, 1}(),
                         Array{ParForContext, 1}())

    ScopeContext(parent_scope::ScopeContext) = new(parent_scope,
                                                   false,
                                                   Dict{Symbol, VarInfo}(),
                                                   Dict{Symbol, VarInfo}(),
                                                   Array{ScopeContext, 1}(),
                                                   Array{ScopeContext, 1}(),
                                                   Array{ParForContext, 1}())
end

function print(scope_context::ScopeContext, indent = 0)
    indent_str = " " ^ indent
    println(indent_str * "inherited:")
    in_indent_str = " " ^ (indent + 1)
    for (var, info) in scope_context.inherited_var
        println(in_indent_str, var, "  ", string(info))
    end
    println(indent_str * "local:")
    for (var, info) in scope_context.local_var
        println(in_indent_str, var, "  ", string(info))
    end
    println(indent_str * "child scope:")
    for scope in scope_context.child_scope
       print(scope, indent + 2)
    end

    println(indent_str * "par_for scope:")
    for scope in scope_context.par_for_scope
        print(scope, indent + 2)
    end
end

# there would be a run time error if users defines a function named "a"
# and try to use "a" as variable in @transform
function is_var_defined(var::Symbol)::Bool
    if isdefined(current_module(), var) && which(var) == current_module()
        return true
    else
        return false
    end
end

function is_var_defined_in_parent(
    scope_context::ScopeContext,
    var::Symbol,
    info::VarInfo)
    if scope_context.parent_scope == nothing
        return is_var_defined(var)
    else
        parent_scope = scope_context.parent_scope
        if var in keys(parent_scope.local_var) ||
            var in keys(parent_scope.inherited_var)
            return true
        elseif parent_scope.is_hard_scope &&
            info.is_assigned_to
            return false
        else
            return is_var_defined_in_parent(parent_scope, var, info)
        end
    end
end

function add_global_var!(scope_context::ScopeContext,
                         var::Symbol,
                         info::VarInfo)
    if var in keys(scope_context.local_var)
        error("conflicting classification of symbol ", var, " local or global?")
    else
        scope_context.inherited_var[var].is_assigned_to |= info.is_assigned_to
        scope_context.inherited_var[var].is_modified |= info.is_modified
        scope_context.inherited_var[var].is_accumulator |= info.is_accumulator
        scope_context.inherited_var[var].is_marked_global |= info.is_marked_global
    end
end

function prepare_par_for_scope(par_for_scope::ScopeContext)
    for (var, info) in par_for_scope.inherited_var
        if info.is_marked_global
            error("cannot use global within @parallel_for")
        end
    end
end

function add_uncertain_var!(scope_context::ScopeContext,
                            var::Symbol,
                            info::VarInfo)
    if var in keys(scope_context.local_var)
        scope_context.local_var[var].is_assigned_to |= info.is_assigned_to
        scope_context.local_var[var].is_modified |= info.is_modified
        scope_context.local_var[var].is_marked_local |= info.is_marked_local
        scope_context.local_var[var].is_accumulator |= info.is_accumulator
    elseif var in keys(scope_context.inherited_var)
        scope_context.inherited_var[var].is_assigned_to |= info.is_assigned_to
        scope_context.inherited_var[var].is_modified |= info.is_modified
        scope_context.inherited_var[var].is_accumulator |= info.is_accumulator
    else
        scope_context.inherited_var[var] = info
    end
end

function add_local_var!(scope_context::ScopeContext,
                        var::Symbol,
                        info::VarInfo)
    if var in keys(scope_context.inherited_var)
        if scope_context.inherited_var[var].is_marked_global
            error("conflicting classification for symbol ", var, " local or global?")
        end
        info.is_assigned_to |= scope_context.inherited_var[var].is_assigned_to
        info.is_modified |= scope_context.inherited_var[var].is_modified
        info.is_marked_local |= scope_context.inherited_var[var].is_marked_local
        info.is_accumulator |= scope_context.inherited_var[var].is_accumulator
        delete!(scope_context.inherited_var, var)
        scope_context.local_var[var] = info
    elseif var in keys(scope_context.local_var)
        @assert !scope_context.local_var[var].is_marked_global &&
            !info.is_marked_global
        scope_context.local_var[var].is_assigned_to |= info.is_assigned_to
        scope_context.local_var[var].is_modified |= info.is_modified
        scope_context.local_var[var].is_accumulator |= info.is_accumulator
        scope_context.local_var[var].is_marked_local |= info.is_marked_local
    else
        scope_context.local_var[var] = info
    end
end

# a variable may be added as many times as it occurs in the scope; basically a variable
# is always added to inherited unless there's at least one occurance that makes it local.

# it is added to local only if the occurance makes it certain (matches one or some
# of the rules) that it is local, it will be added to local and removes itself
# from inherited if it's there and aborts it can't do so

# a variable is added to global only if it is certain that it is global

# otherwise a variable is added as uncertain
function add_var!(scope_context::ScopeContext,
                  var::Symbol,
                  info::VarInfo)
    # if I am sure var is local
    if info.is_marked_local ||
        # if this introduces a new variable that is not defined in the parent scope
        # it must be local
        (!is_var_defined_in_parent(scope_context, var, info) &&
         info.is_assigned_to) ||
        (scope_context.is_hard_scope && info.is_assigned_to)
        add_local_var!(scope_context, var, info)
    elseif info.is_marked_global
        add_global_var!(scope_context, var, info)
    else
        add_uncertain_var!(scope_context, var, info)
    end
end

function is_var_accumulator(scope_context::ScopeContext,
                            var::Symbol)::Bool
    if var in keys(scope_context.inherited_var)
        return scope_context.inherited_var[var].is_accumulator
    elseif var in keys(scope_context.local_var)
        return scope_context.local_var[var].is_accumulator
    elseif scope_context.parent_scope == nothing
        return false
    else
        return is_var_accumulator(scope_context.parent_scope, var)
    end
end

function add_child_scope!(parent::ScopeContext,
                          child::ScopeContext,
                          par_for::Bool = false)
    for (var, info) in child.inherited_var
        if info.is_marked_global
            add_global_var!(parent, var, info)
        elseif var in keys(parent.local_var) ||
            (parent.is_hard_scope && info.is_assigned_to)
            add_local_var!(parent, var, info)
        else
            add_uncertain_var!(parent, var, info)
        end
        info.is_accumulator |= is_var_accumulator(parent, var)
    end
    if par_for
        push!(parent.par_for_scope, child)
    else
        push!(parent.child_scope, child)
    end
end

macro share(ex::Expr)
    if ex.head == :function
        eval_expr_on_all(ex, :Main)
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

function get_vars_to_broadcast(scope_context::ScopeContext)
    static_broadcast_var = Set{Symbol}()
    dynamic_broadcast_var_array = Array{Set{Symbol}, 1}()
    accumulator_var_array = Array{Set{Symbol}, 1}()
    for par_for_scope in scope_context.par_for_scope
        dynamic_broadcast_var = Set{Symbol}()
        accumulator_var = Set{Symbol}()
        for (var, info) in par_for_scope.inherited_var
            if isa(eval(current_module(), var), DistArray)
                continue
            end
            # we do not broadcast variables that are defined in other modules
            if var in keys(scope_context.inherited_var) &&
                which(var) != current_module()
                println(var, " not in current module")
                continue
            elseif var in keys(scope_context.inherited_var) &&
                !scope_context.inherited_var[var].is_modified &&
                !scope_context.inherited_var[var].is_assigned_to
                println("static broadcast ", var)
                push!(static_broadcast_var, var)
            elseif var in keys(scope_context.local_var) &&
                info.is_accumulator
                println("create accumulator ", var)
                push!(accumulator_var,var)
            else
                println("dynamic broadcast ", var)
                push!(dynamic_broadcast_var, var)
            end
        end
        push!(dynamic_broadcast_var_array, dynamic_broadcast_var)
        push!(accumulator_var_array, accumulator_var)
    end
    return (static_broadcast_var, dynamic_broadcast_var_array,
            accumulator_var_array)
end

function transform_loop(expr::Expr, context::ScopeContext)
    scope_context = get_vars!(nothing, expr)
    print(scope_context)

    iterative_body = quote
    end
    push!(iterative_body.args, Expr(:call, :println, "ran one iteration"))
    ret = quote
    end

    static_bc_var, dynamic_bc_var_array,
    accumulator_var_array = get_vars_to_broadcast(scope_context)
    println("broadcat list ", static_bc_var)
    define_var(static_bc_var)

    for curr_par_for_context in scope_context.par_for_context
        iteration_space = curr_par_for_context.iteration_space
        iteration_space_dist_array = eval(current_module(), iteration_space)
        println(typeof(iteration_space_dist_array))
        partition_func_name = gen_unique_symbol()
        partition_func = gen_space_time_partition_function(partition_func_name,
                                                [(1,), (2,)], [(1,), (1,)],
                                                100, 100)
        eval_expr_on_all(partition_func, :OrionGen)
        space_time_repartition(iteration_space_dist_array,
                               string(partition_func_name))
    end

    #bc_expr_array = Array{Array{Expr, 1}, 1}()
#    for dynamic_bc_var in dynamic_bc_var_array
#        bc_expr_array = gen_stmt_broadcast_var(dynamic_bc_var)
        #push!(bc_expr_array, expr_array)
#        for bc_expr in bc_expr_array
#            push!(iterative_body.args, bc_expr)
#        end
#    end

 #   push!(ret.args,
 #         Expr(expr.head,
 #              esc(expr.args[1]),
 #              iterative_body
 #              )
 #         )
    return ret
end

function transform_func(expr::Expr)
    return :(assert(false))
end

macro accumulator(expr::Expr)
end

macro parallel_for(expr::Expr)
end
