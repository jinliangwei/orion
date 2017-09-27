type ScopeContextProcessInfo
    scope_context
    helper_context
    ScopeContextProcessInfo(scope_context) = new(scope_context, nothing)
end
function get_scope_context_visit(expr::Any,
                                 scope_context_info::ScopeContextProcessInfo,
                                 top_level::Integer,
                                 is_top_level::Bool,
                                 read::Bool)
    scope_context = scope_context_info.scope_context
    is_par_for = isa(scope_context_info.helper_context, ParForContext)

    if isa(expr, Symbol)
        println(expr)
        if expr == Symbol("@accumulator") ||
            expr == Symbol("@parallel_for") ||
            expr == :(:) ||
            (isdefined(current_module(), expr) &&
             isa(eval(current_module(), expr), Function))
            return expr
        end

        info = VarInfo()
        if !read
            info.is_mutated = true
        end
        add_var!(scope_context, expr, info)
        return expr
    end
    if isa(expr, Number) || isa(expr, String)
        return expr
    end
    @assert isa(expr, Expr) "must be an Expr"
    head = expr.head
    args = expr.args
    if head == :macrocall
        if macrocall_get_symbol(expr) == Symbol("@accumulator")
            @assert is_variable_definition(args[2])
            var = assignment_get_assigned_to(args[2])
            @assert isa(var, Symbol)
            info = VarInfo()
            info.is_accumulator = true
            add_var!(scope_context, var, info)
            return AstWalk.AST_WALK_RECURSE
        elseif macrocall_get_symbol(expr) == Symbol("@parallel_for") ||
            macrocall_get_symbol(expr) == Symbol("@ordered_parallel_for")
            @assert scope_context.parent_scope == nothing
            @assert length(args) == 2
            @assert is_for_loop(args[2]) println(args[2])
            loop_stmt = args[2]
            iteration_var = for_get_iteration_var(loop_stmt)
            iteration_space = for_get_iteration_space(loop_stmt)
            @assert isa(iteration_space, Symbol)
            @assert isdefined(iteration_space)
            @assert isa(eval(current_module(), iteration_space), DistArray)

            par_for_context = ParForContext(iteration_var,
                                            iteration_space,
                                            loop_stmt,
                                            macrocall_get_symbol(expr) == Symbol("@ordered_parallel_for"))
            push!(scope_context.par_for_context, par_for_context)
            par_for_scope = get_scope_context!(scope_context, loop_stmt, par_for_context)
            add_child_scope!(scope_context, par_for_scope, true)
            return expr
        else
            error("unsupported macro call in Orion transform scope")
        end
    elseif head == :for
        loop_stmt = args[2]
        child_scope = get_scope_context!(scope_context, loop_stmt)
        return expr
    elseif head == :(=) ||
        head == :(+=) ||
        head == :(-=) ||
        head == :(*=) ||
        head == :(/=) ||
        head == :(.*=) ||
        head == :(./=)
        assigned_to = assignment_get_assigned_to(expr)
        assigned_from = assignment_get_assigned_from(expr)
       if isa(assigned_to, Symbol)
           var = assigned_to
           info = VarInfo()
           info.is_assigned_to = true
           add_var!(scope_context, var, info)

           ssa_var = get_ssa_symbol(scope_context, var)

           if head == :(+=)
               def_expr = :($ssa_var + $(deepcopy(assigned_from)))
           elseif head == :(-=)
               def_expr = :($ssa_var - $(deepcopy(assigned_from)))
           elseif head == :(*=)
               def_expr = :($ssa_var * $(deepcopy(assigned_from)))
           elseif head == :(/=)
               def_expr = :($ssa_var / $(deepcopy(assigned_from)))
           elseif head == :(.*=)
               def_expr = :($ssa_var .* $(deepcopy(assigned_from)))
           elseif head == :(./=)
               def_expr = :($ssa_var ./ $(deepcopy(assigned_from)))
           elseif head == :(=)
               def_expr = deepcopy(assigned_from)
           else
               error("syntax not yet supported ", expr)
           end
           def_expr = AstWalk.ast_walk(def_expr, substitute_ssa_symbol_visit,
                                       scope_context)

           if is_symbol_defined(scope_context, var)
               ssa_sym = gen_unique_symbol()
               add_symbol_remap!(scope_context, var, ssa_sym)
               add_symbol_def!(scope_context, ssa_sym, def_expr)
           else
               add_symbol_def!(scope_context, var, def_expr)
           end
       elseif is_ref(assigned_to)
           var = ref_get_root_var(expr)
           info = VarInfo()
           info.is_mutated = true
           add_var!(scope_context, var, info)
       else
           error("unsupported syntax", expr)
       end
        return AstWalk.AST_WALK_RECURSE
    elseif expr.head == :call
        #println("get :call ")
        #dump(expr)
        return AstWalk.AST_WALK_RECURSE
    elseif expr.head == :ref
        if is_par_for
            par_for_context = scope_context_info.helper_context
            expr_referenced = ref_get_referenced_var(expr)
            if !isa(expr_referenced, Symbol) ||
                !is_dist_array(expr_referenced)
                return AstWalk.AST_WALK_RECURSE
            end
            da_access = DistArrayAccess()
            da_access.dist_array = expr_referenced
            for sub in expr.args[2:end]
                push!(da_access.subscripts, DistArrayAccessSubscript())
                if sub == :(:) || isa(sub, Number)
                    da_access.subscripts[end].expr = sub
                else
                    sub_expr = deepcopy(sub)
                    sub_expr = AstWalk.ast_walk(sub_expr,
                                                substitute_ssa_symbol_visit,
                                                scope_context)
                    da_access.subscripts[end].expr = sub_expr
                end
            end
            da_access.is_read = read
            dist_array_access_dict = par_for_context.dist_array_access_dict
            if !haskey(dist_array_access_dict, expr_referenced)
                dist_array_access_dict[expr_referenced] = Vector{DistArrayAccess}()
            end
            push!(dist_array_access_dict[expr_referenced], da_access)
        end
        return AstWalk.AST_WALK_RECURSE
    elseif head == :block
        return AstWalk.AST_WALK_RECURSE
    else
        return expr
    end
end

function substitute_ssa_symbol_visit(expr::Any,
                                     scope_context::ScopeContext,
                                     top_level::Integer,
                                     is_top_level::Bool,
                                     read::Bool)
    if isa(expr, Symbol)
        ssa_symbol = get_ssa_symbol(scope_context, expr)
        if ssa_symbol == nothing
            return expr
        else
            return ssa_symbol
        end
    elseif isa(expr, Expr)
        return AstWalk.AST_WALK_RECURSE
    else
        return expr
    end
end

# expr is contained in the scope denoted by scope_context
function get_scope_context!(scope_context::Any,
                            expr,
                            helper_context = false)
    if isa(expr, Expr)
        head = expr.head
        if head == :for ||
            head == :while ||
            # for list comprehension
            head == :generator
            child_scope_context = ScopeContext()
            child_scope_context.parent_scope = scope_context
            child_scope_context_info = ScopeContextProcessInfo(child_scope_context)
            child_scope_context_info.helper_context = helper_context
            for arg in expr.args
                AstWalk.ast_walk(arg, get_scope_context_visit, child_scope_context_info)
            end
            return child_scope_context
        else
            scope_context_info = ScopeContextProcessInfo(scope_context)
            AstWalk.ast_walk(expr, get_scope_context_visit, scope_context_info)
            return nothing
        end
    else
        scope_context_info = ScopeContextProcessInfo(scope_context)
        AstWalk.ast_walk(expr, get_scope_context_visit, scope_context_info)
        return nothing
    end
end
