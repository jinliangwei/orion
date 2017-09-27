import Base.print, Base.string

# The transform scope itself contains no hard scope.
# The SymbolTable.sym_def contains symbols that appear in the transform scope.
# The symbols that are defined or redefined have exactly one definition (SSA) otherwise it
# has no definition (mapping to nothing). The definition contains only SSA variables.

type SymbolTable
    sym_def::Dict{Symbol, Any}
    sym_remap::Dict{Symbol, Symbol}
    SymbolTable() = new(Dict{Symbol, Any}(),
                        Dict{Symbol, Any}())
end

type VarInfo
    is_assigned_to::Bool
    is_mutated::Bool
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
    str *= "[is_mutated=" * string(info.is_mutated) * "], "
    str *= "[is_accumulator=" * string(info.is_accumulator) * "],"
    str *= "[is_marked_local=" * string(info.is_marked_local) * "]}"
    str *= "[is_marked_global=" * string(info.is_marked_global) * "]}"
#    println(str)
end

type DistArrayAccessSubscript
    expr
    offset::Int64
    loop_index_dim
    DistArrayAccessSubscript() = new(nothing,
                                     0,
                                     nothing)
end

type DistArrayAccess
    dist_array::Symbol
    subscripts::Vector{DistArrayAccessSubscript}
    is_read::Bool
    DistArrayAccess() = new(Symbol(""),
                            Vector{DistArrayAccessSubscript}(),
                            false)

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


type ScopeContext
    parent_scope
    is_hard_scope::Bool
    inherited_var::Dict{Symbol, VarInfo}
    local_var::Dict{Symbol, VarInfo}
    par_for_scope::Array{ScopeContext}
    child_scope::Array{ScopeContext}
    par_for_context::Array{ParForContext}
    symbol_table::SymbolTable

    ScopeContext() = new(nothing,
                         false,
                         Dict{Symbol, VarInfo}(),
                         Dict{Symbol, VarInfo}(),
                         Array{ScopeContext, 1}(),
                         Array{ScopeContext, 1}(),
                         Array{ParForContext, 1}(),
                         SymbolTable())

    ScopeContext(parent_scope::ScopeContext) = new(parent_scope,
                                                   false,
                                                   Dict{Symbol, VarInfo}(),
                                                   Dict{Symbol, VarInfo}(),
                                                   Array{ScopeContext, 1}(),
                                                   Array{ScopeContext, 1}(),
                                                   Array{ParForContext, 1}(),
                                                   SymbolTable())
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

    println(indent_str * "par_for context:")
    for context in scope_context.par_for_context
        print(context, indent + 2)
    end

    println(indent_str * "symbol defs:")
    println(indent_str, scope_context.symbol_table.sym_def)

    println(indent_str * "symbol remaps:")
    println(indent_str, scope_context.symbol_table.sym_remap)
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
