import Base: linearindexing, size, getindex, setindex!

mutable struct GetDistArrayAccessContext
    iteration_var::Symbol
    ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}}
    access_dict::Dict{Symbol, Vector{DistArrayAccess}}
    buffer_set::Set{Symbol}
    accessed_buffer::Bool
    stmt_access_dict::Dict{Symbol, Vector{DistArrayAccess}}
    da_access_context::DistArrayAccessContext
    GetDistArrayAccessContext(iteration_var,
                              ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}}) =
                                  new(iteration_var,
                                      ssa_defs,
                                      Dict{Symbol, Vector{DistArrayAccess}}(),
                                      Set{Symbol}(),
                                      false,
                                      Dict{Symbol, Vector{DistArrayAccess}}(),
                                      DistArrayAccessContext())
end


function get_dist_array_access_visit(expr,
                                     context::GetDistArrayAccessContext)
    ssa_defs = context.ssa_defs
    if isa(expr, Expr)
        head = expr.head
        if head in Set([:(=), :(.=), :(+=), :(-=), :(.*=), :(./=)])
            assigned_to = assignment_get_assigned_to(expr)
            if is_ref(assigned_to)
                referenced_var = ref_get_referenced_var(assigned_to)
                if isa(referenced_var, Symbol)
                    if referenced_var in keys(ssa_defs)
                        referenced_var = ssa_defs[referenced_var][1]
                    end
                    if isdefined(current_module(), referenced_var) &&
                        isa(eval(current_module(), referenced_var), DistArray)
                        da_access = DistArrayAccess(referenced_var, false)
                        for sub in ref_get_subscripts(assigned_to)
                            evaled_sub = eval_subscript_expr(sub,
                                                             context.iteration_var,
                                                             ssa_defs)
                            subscript = DistArrayAccessSubscript(sub, evaled_sub...)
                            push!(da_access.subscripts, subscript)
                        end
                        if !(referenced_var in keys(context.access_dict))
                            context.access_dict[referenced_var] = Vector{DistArrayAccess}()
                        end
                        push!(context.access_dict[referenced_var], da_access)
                        if !(referenced_var in keys(context.stmt_access_dict))
                            context.stmt_access_dict[referenced_var] = Vector{DistArrayAccess}()
                        end
                        push!(context.stmt_access_dict[referenced_var], da_access)
                        if head != :(=) &&
                            head != :(.=)
                            da_access = copy(da_access)
                            da_access.is_read = true
                            push!(context.access_dict[referenced_var], da_access)
                            push!(context.stmt_access_dict[referenced_var], da_access)
                        end
                    elseif isdefined(current_module(), referenced_var) &&
                        isa(eval(current_module(), referenced_var), DistArrayBuffer)
                        push!(context.buffer_set, referenced_var)
                        context.accessed_buffer = true
                    end
                    subscripts = ref_get_subscripts(assigned_to)
                    for sub in subscripts
                        AstWalk.ast_walk(sub, get_dist_array_access_visit, context)
                    end
                    assigned_from = assignment_get_assigned_from(expr)
                    AstWalk.ast_walk(assigned_from, get_dist_array_access_visit, context)
                    return expr
                else
                    return AstWalk.AST_WALK_RECURSE
                end
            else
                return AstWalk.AST_WALK_RECURSE
            end
        elseif is_ref(expr)
            referenced_var = ref_get_referenced_var(expr)
            if isa(referenced_var, Symbol)
                if referenced_var in keys(ssa_defs)
                    referenced_var = ssa_defs[referenced_var][1]
                end
                if isdefined(current_module(), referenced_var) &&
                    isa(eval(current_module(), referenced_var), DistArray)
                    da_access = DistArrayAccess(referenced_var, true)
                    for sub in ref_get_subscripts(expr)
                        evaled_sub = eval_subscript_expr(sub,
                                                         context.iteration_var,
                                                         ssa_defs)
                        subscript = DistArrayAccessSubscript(sub, evaled_sub...)
                        push!(da_access.subscripts, subscript)
                    end
                    if !(referenced_var in keys(context.access_dict))
                        context.access_dict[referenced_var] = Vector{DistArrayAccess}()
                    end
                    push!(context.access_dict[referenced_var], da_access)
                    if !(referenced_var in keys(context.stmt_access_dict))
                        context.stmt_access_dict[referenced_var] = Vector{DistArrayAccess}()
                    end
                    push!(context.stmt_access_dict[referenced_var], da_access)
                elseif isdefined(current_module(), referenced_var) &&
                    isa(eval(current_module(), referenced_var), DistArrayBuffer)
                    push!(context.buffer_set, referenced_var)
                    context.accessed_buffer = true
                end
                subscripts = ref_get_subscripts(expr)
                for sub in subscripts
                    AstWalk.ast_walk(sub, get_dist_array_access_visit, context)
                end
                return expr
            else
                return AstWalk.AST_WALK_RECURSE
            end
        else
            return AstWalk.AST_WALK_RECURSE
        end
    else
        return expr
    end
end

function get_dist_array_access_bb(bb::BasicBlock,
                                  context::GetDistArrayAccessContext)
    #println("access bb ", bb.id)
    stmt_access_dict = Dict{Int64, Dict{Symbol, Vector{DistArrayAccess}}}()
    bb_dist_array_access_dict = context.da_access_context.bb_dist_array_access_dict
    bb_dist_array_buffer_access_stmts_dict = context.da_access_context.bb_dist_array_buffer_access_stmts_dict

    for idx in eachindex(bb.stmts)
        stmt = bb.stmts[idx]
        AstWalk.ast_walk(stmt, get_dist_array_access_visit, context)
        if !isempty(context.stmt_access_dict)
            stmt_access_dict[idx] = context.stmt_access_dict
            context.stmt_access_dict = Dict{Symbol, Vector{DistArrayAccess}}()
        end
        if context.accessed_buffer
            context.accessed_buffer = false
            if !(bb.id in keys(bb_dist_array_buffer_access_stmts_dict))
                bb_dist_array_buffer_access_stmts_dict[bb.id] = Set{Int64}()
            end
            push!(bb_dist_array_buffer_access_stmts_dict[bb.id], idx)
        end
    end
    if !isempty(stmt_access_dict)
        bb_dist_array_access_dict[bb.id] = stmt_access_dict
    end
end

function get_dist_array_access(par_for_loop_entry::BasicBlock,
                               iteration_var::Symbol,
                               ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}})
    get_da_access_context = GetDistArrayAccessContext(iteration_var, ssa_defs)
    traverse_flow_graph(par_for_loop_entry,
                        get_dist_array_access_bb,
                        get_da_access_context)
    return get_da_access_context.access_dict, get_da_access_context.buffer_set, get_da_access_context.da_access_context
end
