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
                        println("found! ", da_access)
                        push!(context.access_dict[referenced_var], da_access)
                        if head != :(=)
                            da_access = copy(da_access)
                            da_access.is_read = true
                            push!(context.access_dict[referenced_var], da_access)
                            println("found! ", da_access)
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
                    println("found! ", da_access)
                    push!(context.access_dict[referenced_var], da_access)
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
    for stmt in bb.stmts
        AstWalk.ast_walk(stmt, get_dist_array_access_visit, context)
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

function rewrite_dist_array_access_visit(expr,
                                         cbdata,
                                         top_level::Integer,
                                         is_top_level::Bool,
                                         read::Bool)
    if isa(expr, Expr)
        head = expr.head
        if head in Set([:(=), :(+=), :(-=), :(.*=), :(./=)])
            assigned_to = assignment_get_assigned_to(expr)
            assigned_from = assignment_get_assigned_from(expr)
            if is_ref(assigned_to)
                referenced_var = ref_get_referenced_var(assigned_to)
                if isa(referenced_var, Symbol) &&
                    isdefined(current_module(), referenced_var) &&
                    isa(eval(current_module(), referenced_var), DistArray)
                    dist_array = eval(current_module(), referenced_var)
                    dist_array_id = dist_array.id
                    if head != :(=)
                        error("not supported!")
                    end

                    subscripts = ref_get_subscripts(assigned_to)
                    rewritten_subscripts = Vector{Any}(length(subscripts))
                    for sub_idx in eachindex(subscripts)
                        sub = subscripts[sub_idx]
                        rewritten_subscripts[sub_idx] = rewrite_dist_array_access(sub)
                    end
                    assigned_from = rewrite_dist_array_access(assigned_from)
                    return gen_dist_array_write_func_call(dist_array_id, tuple(subscripts...), assigned_from)
                else
                    return AstWalk.AST_WALK_RECURSE
                end
            else
                return AstWalk.AST_WALK_RECURSE
            end
        elseif is_ref(expr)
            referenced_var = ref_get_referenced_var(expr)
            if isa(referenced_var, Symbol) &&
                isdefined(current_module(), referenced_var) &&
                isa(eval(current_module(), referenced_var), DistArray)
                dist_array = eval(current_module(), referenced_var)
                dist_array_id = dist_array.id

                subscripts = ref_get_subscripts(expr)
                rewritten_subscripts = Vector{Any}(length(subscripts))
                for sub_idx in eachindex(subscripts)
                    sub = subscripts[sub_idx]
                    rewritten_subscripts[sub_idx] = rewrite_dist_array_access(sub)
                end
                return gen_dist_array_read_func_call(dist_array_id, tuple(subscripts...))
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

function rewrite_dist_array_access(expr)
    AstWalk.ast_walk(expr, rewrite_dist_array_access_visit, nothing)
end
