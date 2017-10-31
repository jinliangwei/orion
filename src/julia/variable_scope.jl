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

function is_var_accumulator(scope_context::ScopeContext,
                            var::Symbol)::Bool
    if var in keys(scope_context.inherited_var) &&
        var in keys(accumulator_info_dict)
        return true
    else
        return false
    end
end

function add_global_var!(scope_context::ScopeContext,
                         var::Symbol,
                         info::VarInfo)
    if var in keys(scope_context.local_var)
        error("conflicting classification of symbol ", var, " local or global?")
    else
        scope_context.inherited_var[var].is_assigned_to |= info.is_assigned_to
        scope_context.inherited_var[var].is_mutated |= info.is_mutated
        scope_context.inherited_var[var].is_marked_global |= info.is_marked_global
    end
end

function add_uncertain_var!(scope_context::ScopeContext,
                            var::Symbol,
                            info::VarInfo)
    if var in keys(scope_context.local_var)
        scope_context.local_var[var].is_assigned_to |= info.is_assigned_to
        scope_context.local_var[var].is_mutated |= info.is_mutated
        scope_context.local_var[var].is_marked_local |= info.is_marked_local
    elseif var in keys(scope_context.inherited_var)
        scope_context.inherited_var[var].is_assigned_to |= info.is_assigned_to
        scope_context.inherited_var[var].is_mutated |= info.is_mutated
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
        info.is_mutated |= scope_context.inherited_var[var].is_mutated
        info.is_marked_local |= scope_context.inherited_var[var].is_marked_local
        delete!(scope_context.inherited_var, var)
        scope_context.local_var[var] = info
    elseif var in keys(scope_context.local_var)
        @assert !scope_context.local_var[var].is_marked_global &&
            !info.is_marked_global
        scope_context.local_var[var].is_assigned_to |= info.is_assigned_to
        scope_context.local_var[var].is_mutated |= info.is_mutated
        scope_context.local_var[var].is_marked_local |= info.is_marked_local
    else
        scope_context.local_var[var] = info
    end
end

function get_vars_to_broadcast(scope_context::ScopeContext)
    bc_vars = Set{Symbol}()
    accumulator_vars = Set{Symbol}()
    for (var, info) in scope_context.inherited_var
        if !isdefined(current_module(), var) ||
            isa(eval(current_module(), var), DistArray) ||
            var in keys(accumulator_info_dict)
            continue
        end

        if var in keys(scope_context.inherited_var)
            push!(bc_vars, var)
        end
    end
    return bc_vars
end

function add_child_scope!(parent::ScopeContext,
                          child::ScopeContext)
    for (var, info) in child.inherited_var
        if info.is_marked_global
            add_global_var!(parent, var, info)
        elseif var in keys(parent.local_var) ||
            (parent.is_hard_scope && info.is_assigned_to)
            add_local_var!(parent, var, info)
        else
            add_uncertain_var!(parent, var, info)
        end
    end
    push!(parent.child_scope, child)
end
