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

type AccumulatorInfo
    sym::Symbol
    init_value
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
    if length(scope_context.child_scope) > 0
        println(indent_str * "child scope:")
        for scope in scope_context.child_scope
            println(indent_str * (" " ^ 2), "...")
            print(scope, indent + 2)
            println(indent_str * (" " ^ 2), "...")
        end
    end
end
