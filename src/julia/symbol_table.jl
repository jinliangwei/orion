# symbol must be SSA
function get_symbol_def(scope_context::ScopeContext,
                        sym::Symbol)
    curr_scope_context = scope_context
    while curr_scope_context != nothing
        if sym in keys(curr_scope_context.symbol_table.sym_def)
            return curr_scope_context.symbol_table.sym_def[sym]
        else
            curr_scope_context = curr_scope_context.parent_scope
        end
    end
    return nothing
end

# since a variable can be remapped to different SSA variables,
# the result of this function depends on where the current processing is at
function get_ssa_symbol(scope_context::ScopeContext, sym::Symbol)
    curr_scope_context = scope_context
    sym_to_find = sym

    while curr_scope_context != nothing
        if sym_to_find in keys(curr_scope_context.symbol_table.sym_remap)
            return curr_scope_context.symbol_table.sym_remap[sym_to_find]
        else
            curr_scope_context = curr_scope_context.parent_scope
        end
    end
    return sym
end

function is_symbol_defined(scope_context::ScopeContext, sym::Symbol)::Bool
    curr_scope_context = scope_context
    sym_to_find = sym
    while curr_scope_context != nothing
        if sym_to_find in keys(curr_scope_context.symbol_table.sym_def)
            return true
        else
            curr_scope_context = curr_scope_context.parent_scope
        end
    end
    return false
end

function add_symbol_def!(scope_context::ScopeContext, sym::Symbol, def)
    scope_context.symbol_table.sym_def[sym] = def
end

function add_symbol_remap!(scope_context::ScopeContext, from::Symbol, to::Symbol)
    scope_context.symbol_table.sym_remap[from] = to
end
