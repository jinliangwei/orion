type GetDistArrayAccessContext
    iteration_var::Symbol
    ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}}
    access_dict::Dict{Symbol, Vector{DistArrayAccess}}
    stmt_access_dict::Dict{Symbol, Vector{DistArrayAccess}}

    GetDistArrayAccessContext(iteration_var,
                              ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}}) =
                                  new(iteration_var,
                                      ssa_defs,
                                      Dict{Symbol, Vector{DistArrayAccess}}(),
                                      Dict{Symbol, Vector{DistArrayAccess}}())
end

# bb_id -> (stmt_index -> stmt_dist_array_access)
bb_dist_array_access_dict = Dict{Int64, Dict{Int64, Dict{Symbol, Vector{DistArrayAccess}}}}()

function get_dist_array_access_visit(expr,
                                     context::GetDistArrayAccessContext,
                                     top_level::Integer,
                                     is_top_level::Bool,
                                     read::Bool)
    ssa_defs = context.ssa_defs
    if isa(expr, Expr)
        head = expr.head
        if head in Set([:(=), :(+=), :(-=), :(.*=), :(./=)])
            assigned_to = assignment_get_assigned_to(expr)
            if is_ref(assigned_to)
                referenced_var = ref_get_referenced_var(assigned_to)
                if isa(referenced_var, Symbol)
                    if referenced_var in keys(ssa_defs)
                        referenced_var = ssa_defs[referenced_var][1]
                    end
                    println("referencing ", referenced_var)
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
                        if head != :(=)
                            da_access = copy(da_access)
                            da_access.is_read = true
                            push!(context.access_dict[referenced_var], da_access)
                            push!(context.stmt_access_dict[referenced_var], da_access)
                        end
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
    stmt_access_dict = Dict{Int64, Dict{Symbol, Vector{DistArrayAccess}}}()
    for idx in eachindex(bb.stmts)
        stmt = bb.stmts[idx]
        AstWalk.ast_walk(stmt, get_dist_array_access_visit, context)
        if !isempty(context.stmt_access_dict)
            stmt_access_dict[idx] = context.stmt_access_dict
            context.stmt_access_dict = Dict{Symbol, Vector{DistArrayAccess}}()
        end
    end
    if !isempty(stmt_access_dict)
        bb_dist_array_access_dict[bb.id] = stmt_access_dict
    end
end

function get_dist_array_access(par_for_loop_entry::BasicBlock,
                               iteration_var::Symbol,
                               ssa_context::SsaContext)
    context = GetDistArrayAccessContext(iteration_var,
                                        ssa_context.ssa_defs)
    traverse_for_loop(par_for_loop_entry, get_dist_array_access_bb,
                      context)
    return context.access_dict
end
