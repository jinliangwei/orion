import Base.print, Base.string

type VarInfo
    is_assigned_to::Bool
    is_mutated::Bool
    is_marked_local::Bool
    is_marked_global::Bool

    VarInfo() = new(false,
                    false,
                    false,
                    false)
end

function string(info::VarInfo)
    str = "VarInfo{"
    str *= "[is_assigned_to=" * string(info.is_assigned_to) * "], "
    str *= "[is_mutated=" * string(info.is_mutated) * "], "
    str *= "[is_marked_local=" * string(info.is_marked_local) * "]}"
    str *= "[is_marked_global=" * string(info.is_marked_global) * "]}"
#    println(str)
end

@enum DistArrayAccessSubscript_value DistArrayAccessSubscript_value_any =
    1 DistArrayAccessSubscript_value_static =
    2 DistArrayAccessSubscript_value_unknown =
    3

type DistArrayAccessSubscript
    expr
    value_type
    loop_index_dim
    offset
end

type DistArrayAccess
    dist_array::Symbol
    subscripts::Vector{DistArrayAccessSubscript}
    is_read::Bool
    DistArrayAccess(dist_array, is_read) =
        new(dist_array,
            Vector{DistArrayAccessSubscript}(),
            is_read)
end

type ParForContext
    iteration_var::Symbol
    iteration_space::Symbol
    loop_stmt::Expr
    dist_array_access_dict::Dict{Symbol, Vector{DistArrayAccess}}
    is_ordered::Bool
    ParForContext(iteration_var::Symbol,
                  iteration_space::Symbol,
                  loop_stmt,
                  is_ordered::Bool) = new(iteration_var,
                                          iteration_space,
                                          loop_stmt,
                                          Dict{Symbol, Vector{DistArrayAccess}}(),
                                          is_ordered)
end

type AccumulatorInfo
    sym::Symbol
    initializer
    combiner_func::Symbol
end

accumulator_info_dict = Dict{Symbol, AccumulatorInfo}()

type ScopeContext
    parent_scope
    is_hard_scope::Bool
    inherited_var::Dict{Symbol, VarInfo}
    local_var::Dict{Symbol, VarInfo}
    child_scope::Vector{ScopeContext}

    ScopeContext() = new(nothing,
                         false,
                         Dict{Symbol, VarInfo}(),
                         Dict{Symbol, VarInfo}(),
                         Vector{ScopeContext}())

    ScopeContext(parent_scope::ScopeContext) = new(parent_scope,
                                                   false,
                                                   Dict{Symbol, VarInfo}(),
                                                   Dict{Symbol, VarInfo}(),
                                                   Vector{ScopeContext}())
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
end

function print(par_for_context::ParForContext, indent = 0)
    indent_str = " " ^ indent
    println(indent_str, "iteration_space: ", par_for_context.iteration_space)
    println(indent_str, "iteration_var: ", par_for_context.iteration_var)
    for da_access_vec in values(par_for_context.dist_array_access_dict)
        for da_access in da_access_vec
            println(indent_str, "dist_array_access: (", da_access.dist_array, " subscripts: ",
                    da_access.subscripts, " is_read: ", da_access.is_read, ")")
        end
    end
end
