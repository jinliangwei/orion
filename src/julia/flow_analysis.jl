type VarDef
    assignment
    mutations::Vector{Expr}
    VarDef(assigned) = new(assigned, Vector{Expr}())
end

type BasicBlock
    id::Int64
    predecessors::Vector{Any}
    successors::Vector{Tuple{Any, BasicBlock}}
    stmts::Vector{Any}
    control_flow
    uses::Set{Symbol}
    defsout::Dict{Symbol, VarDef}
    killed::Set{Symbol}
    mutates::Dict{Symbol, Vector{Expr}}
    dominators::Set{BasicBlock}
    im_doms::Set{BasicBlock}
    dominatees::Set{BasicBlock}
    df::Set{BasicBlock}
    ssa_defsout::Set{Symbol}
    sym_to_ssa_var_map::Dict{Symbol, Symbol}
    reaches::Set{Symbol}
    ssa_reaches::Set{Symbol}
    ssa_reaches_alive::Set{Symbol}
    ssa_reaches_dict::Dict{Symbol, Vector{Symbol}}
    BasicBlock(id) = new(id,
                         Vector{BasicBlock}(),
                         Vector{Tuple{Any, BasicBlock}}(),
                         Vector{Any}(),
                         nothing,
                         Set{Symbol}(),
                         Dict{Symbol, VarDef}(),
                         Set{Symbol}(),
                         Dict{Symbol, Vector{Expr}}(),
                         Set{BasicBlock}(),
                         Set{BasicBlock}(),
                         Set{BasicBlock}(),
                         Set{BasicBlock}(),
                         Set{Symbol}(),
                         Dict{Symbol, Symbol}(),
                         Set{Symbol}(),
                         Set{Symbol}(),
                         Set{Symbol}(),
                         Dict{Symbol, Vector{Symbol}}())
end

type FlowGraphBuildContext
    bb_counter::Int64
    par_for_bbs::Vector{BasicBlock}
    FlowGraphBuildContext() = new(0,
                                  Vector{BasicBlock}())
end

function create_basic_block(build_context::FlowGraphBuildContext)::BasicBlock
    bb = BasicBlock(build_context.bb_counter)
    build_context.bb_counter += 1
    return bb
end

function build_flow_graph(expr::Expr)
    build_context = FlowGraphBuildContext()
    graph_entry = create_basic_block(build_context)
    push!(graph_entry.predecessors, nothing)
    return graph_entry, build_flow_graph(expr, graph_entry, build_context)
end

function print_basic_block(bb::BasicBlock)
    println("BasicBlock, id = ", bb.id)
    println("predecessors = [", Base.map(x -> (x != nothing ? x.id : "entry"), bb.predecessors), "]")
    println("successors = [", Base.map(x -> (x[1], x[2].id), bb.successors), "]")
    println("uses = [", bb.uses, "]")
    println("defsout = [", bb.defsout, "]")
    println("killed = [", bb.killed, "]")
    println("dominators = [", Base.map(x -> x.id, bb.dominators), "]")
    println("im_dominators = [", Base.map(x -> x.id, bb.im_doms), "]")
    println("dominance frontier = ", Base.map(x -> x.id, bb.df), "]")
    println("ssa_defsout = [", bb.ssa_defsout, "]")
    println("ssa_reaches = [", bb.ssa_reaches, "]")
    println("stmts = [")
    for stmt in bb.stmts
        println("  ", stmt)
    end
    println("]")
end

function traverse_flow_graph(entry::BasicBlock,
                             callback,
                             cbdata)
    bb_list = Vector{BasicBlock}()
    push!(bb_list, entry)
    visited = Set{Int64}()
    while !isempty(bb_list)
        bb = shift!(bb_list)
        if bb.id in visited
            continue
        end
        push!(visited, bb.id)
        callback(bb, cbdata)
        for suc in bb.successors
            push!(bb_list, suc[2])
        end
    end
end

function print_flow_graph_visit(bb::BasicBlock, cbdata)
    print_basic_block(bb)
end

function print_flow_graph(entry::BasicBlock)
    traverse_flow_graph(entry, print_flow_graph_visit, nothing)
end

function compute_use_def_flow_graph_visit(bb::BasicBlock, cbdata)
    compute_use_def(bb)
end

function compute_use_def_flow_graph(entry::BasicBlock)
    traverse_flow_graph(entry, compute_use_def_flow_graph_visit, nothing)
end

function build_flow_graph(expr::Expr, bb::BasicBlock,
                          build_context::FlowGraphBuildContext)::Vector{BasicBlock}
    exit_bbs = Vector{BasicBlock}()
    if !isa(expr, Expr)
        push!(bb.stmts, expr)
        push!(exit_bbs, bb)
        return exit_bbs
    end
    if expr.head == :if
        condition = if_get_condition(expr)
        push!(bb.stmts, condition)
        true_bb_entry = create_basic_block(build_context)

        true_bb_exits = build_flow_graph(if_get_true_branch(expr), true_bb_entry,
                                         build_context)
        push!(bb.successors, (true, true_bb_entry))
        push!(true_bb_entry.predecessors, bb)
        false_branch = if_get_false_branch(expr)
        if false_branch != nothing
            false_bb_entry = create_basic_block(build_context)
            false_bb_exits = build_flow_graph(false_branch, false_bb_entry,
                                              build_context)
            push!(bb.successors, (false, false_bb_entry))
            push!(false_bb_entry.predecessors, bb)
            append!(exit_bbs, true_bb_exits)
            append!(exit_bbs, false_bb_exits)
            bb.control_flow = (:if, :else)
        else
            exit_bb = create_basic_block(build_context)
            for true_bb_exit in true_bb_exits
                push!(true_bb_exit.successors, (nothing, exit_bb))
                push!(exit_bb.predecessors, true_bb_exit)
            end
            push!(bb.successors, (false, exit_bb))
            push!(exit_bb.predecessors, bb)
            push!(exit_bbs, exit_bb)
            bb.control_flow = :if
        end
    elseif expr.head == :for
        loop_condition = for_get_loop_condition(expr)
        push!(bb.stmts, loop_condition)
        true_bb_entry = create_basic_block(build_context)

        true_bb_exits = build_flow_graph(for_get_loop_body(expr),
                                         true_bb_entry,
                                         build_context)
        push!(bb.successors, (true, true_bb_entry))
        push!(true_bb_entry.predecessors, bb)

        for exit_bb in true_bb_exits
            push!(bb.predecessors, exit_bb)
            push!(exit_bb.successors, (nothing, bb))
        end

        loop_exit_bb = create_basic_block(build_context)
        push!(bb.successors, (false, loop_exit_bb))
        push!(loop_exit_bb.predecessors, bb)

        push!(exit_bbs, loop_exit_bb)
        bb.control_flow = :for
    elseif expr.head == :while
        loop_condition = while_get_loop_condition(expr)
        push!(bb.stmts, loop_condition)
        true_bb_entry = create_basic_block(build_context)

        true_bb_exits = build_flow_graph(for_get_loop_body(expr),
                                         true_bb_entry,
                                         build_context)
        push!(bb.successors, (true, true_bb_entry))
        push!(true_bb_entry.predecessors, bb)
        for exit_bb in true_bb_exits
            push!(bb.predecessors, exit_bb)
            push!(exit_bb.successors, (nothing, bb))
        end
        loop_exit_bb = create_basic_block(build_context)
        push!(bb.successors, (false, loop_exit_bb))
        push!(exit_bbs, loop_exit_bb)
        bb.control_flow = :while
    elseif expr.head == :block
        curr_bb = bb
        for stmt in block_get_stmts(expr)
            exit_bbs = build_flow_graph(stmt, curr_bb, build_context)
            if length(exit_bbs) == 1
                curr_bb = exit_bbs[1]
            elseif length(exit_bbs) > 1
                curr_bb = create_basic_block(build_context)
                for exit_bb in exit_bbs
                    push!(exit_bb.successors, (nothing, curr_bb))
                    push!(curr_bb.predecessors, exit_bb)
                end
            end
        end
    elseif expr.head == :macrocall &&
        (macrocall_get_symbol(expr) == Symbol("@parallel_for") ||
         macrocall_get_symbol(expr) == Symbol("@ordered_parallel_for"))
        exit_bbs = build_flow_graph(expr.args[2], bb, build_context)
        bb.control_flow == macrocall_get_symbol(expr)
        push!(build_context.par_for_bbs, bb)
    else
        push!(bb.stmts, expr)
        push!(exit_bbs, bb)
    end
    return exit_bbs
end

function compute_use_def_expr(stmt, bb::BasicBlock)
    uses = bb.uses
    defsout = bb.defsout
    killed = bb.killed
    mutates = bb.mutates

    if isa(stmt, Symbol)
        if !(stmt in keys(defsout))
            push!(uses, stmt)
        end
    elseif isa(stmt, Expr)
        if stmt.head in Set([:(=), :(+=), :(-=), :(*=), :(/=), :(.*=), :(./=)])
            compute_use_def_expr(assignment_get_assigned_from(stmt), bb)
            assigned_to = assignment_get_assigned_to(stmt)
            if isa(assigned_to, Symbol)
                assigned_expr = assignment_get_assigned_from(stmt)
                if stmt.head == :(+=)
                    assigned_expr = Expr(:call, :+, assigned_to, assigned_expr)
                elseif stmt.head == :(-=)
                    assigned_expr = Expr(:call, :-, assigned_to, assigned_expr)
                elseif stmt.head == :(*=)
                    assigned_expr = Expr(:call, :*, assigned_to, assigned_expr)
                elseif stmt.head == :(/=)
                    assigned_expr = Expr(:call, :/, assigned_to, assigned_expr)
                elseif stmt.head == :(.*=)
                    assigned_expr = Expr(:call, :(.*), assigned_to, assigned_expr)
                elseif stmt.head == :(./=)
                    assigned_expr = Expr(:call, :(./), assigned_to, assigned_expr)
                end
                defsout[assigned_to] = VarDef(assigned_expr)
                push!(killed, assigned_to)
                if stmt.head != :(=) &&
                    !(assigned_to in keys(defsout))
                    push!(uses, assigned_to)
                end
            else
                @assert isa(assigned_to, Expr)
                @assert is_ref(assigned_to) || is_dot(assigned_to)
                var_mutated = ref_dot_get_mutated_var(assigned_to)
                if var_mutated != nothing
                    if var_mutated in keys(defsout)
                        push!(defsout[var_mutated].mutations, stmt)
                    else
                        if !(var_mutated in keys(mutates))
                            mutates[var_mutated] = Vector{Expr}()
                        end
                        push!(mutates[var_mutated], stmt)
                        push!(uses, var_mutated)
                        push!(killed, var_mutated)
                    end
                end
                compute_use_def_expr(assigned_to, bb)
            end
        elseif stmt.head in Set([:call, :invoke, :call1, :foreigncall])

            if call_get_func_name(stmt) in Set([:+, :-, :*, :/])
                for arg in call_get_arguments(stmt)
                    compute_use_def_expr(arg, bb)
                end
            else
                for arg in call_get_arguments(stmt)
                    var_mutated = nothing
                    if isa(arg, Symbol)
                        var_mutated = arg
                    elseif isa(arg, Expr) &&
                        (is_ref(arg) || is_dot(arg))
                        var_mutated = ref_dot_get_mutated_var(arg)
                    end
                    if var_mutated != nothing
                        if var_mutated in keys(defsout)
                            push!(defsout[var_mutated].mutations, stmt)
                        else
                            if !(var_mutated in keys(mutates))
                                mutates[var_mutated] = Vector{Expr}()
                            end
                            push!(mutates[var_mutated], stmt)
                            push!(killed, var_mutated)
                        end
                    end
                    compute_use_def_expr(arg, bb)
                end
            end
        else
            var_set = Set{Symbol}()
            AstWalk.ast_walk(stmt, get_symbols_visit, var_set)
            for var_sym in var_set
                if !(var_sym in keys(defsout))
                    push!(uses, var_sym)
                end
            end
        end
     end
end

function compute_use_def(bb::BasicBlock)
    for stmt in bb.stmts
        compute_use_def_expr(stmt, bb)
    end
end

function get_symbols_visit(expr::Any,
                           symbol_set::Set{Symbol},
                           top_level::Integer,
                           is_top_level::Bool,
                           read::Bool)
    if isa(expr, Symbol)
        push!(symbol_set, expr)
        return expr
    elseif isa(expr, Expr)
        if expr.head == :line
            return expr
        end
        return AstWalk.AST_WALK_RECURSE
    end
    return expr
end

function append_basic_block_visit(bb::BasicBlock, vec::Vector{BasicBlock})
    push!(vec, bb)
end

function flow_graph_to_list(entry::BasicBlock)
    vec = Vector{BasicBlock}()
    traverse_flow_graph(entry, append_basic_block_visit, vec)
    return vec
end

function compute_dominators(entry::BasicBlock)
    bb_list = flow_graph_to_list(entry)
    for bb in bb_list
        if isempty(bb.predecessors)
            bb.dominators = Set([bb])
        else
            bb.dominators = Set(bb_list)
        end
    end

    changed = true
    while changed
        changed = false
        for bb in bb_list
            new_doms = bb.dominators
            for p in bb.predecessors
                if p == nothing
                    new_doms = Set{BasicBlock}()
                else
                    new_doms = intersect(p.dominators, new_doms)
                end
            end
            push!(new_doms, bb)
            if new_doms != bb.dominators
                bb.dominators = new_doms
                changed = true
            end
        end
    end
end

function compute_im_doms(entry::BasicBlock)
    bb_list = flow_graph_to_list(entry)
    for bb in bb_list
        bb.im_doms = Set{BasicBlock}()
        strict_doms = setdiff(bb.dominators, Set([bb]))
        strict_doms_vec = [x for x in strict_doms]
        sort!(strict_doms_vec,
              lt = ((x, y) -> (length(x.dominators) < length(y.dominators))),
              rev = true)
        num_doms = 0
        for dom in strict_doms_vec
            if num_doms == 0
                num_doms = length(dom.dominators)
                push!(bb.im_doms, dom)
            elseif num_doms == length(dom.dominators)
                push!(bb.im_doms, dom)
            else
                break
            end
        end
    end
end

function compute_dominatees(entry::BasicBlock)
    bb_list = flow_graph_to_list(entry)
    for bb in bb_list
        for dom in bb.dominators
            push!(dom.dominatees, bb)
        end
    end
end

function construct_dominance_frontier(entry::BasicBlock)
    compute_im_doms(entry)
    compute_dominatees(entry)
    bb_list = flow_graph_to_list(entry)
    # topological sorting based on dominance
    dom_reverse_list = Vector{BasicBlock}()
    while length(dom_reverse_list) < length(bb_list)
        for bb in bb_list
            if length(bb.dominatees) == 1
                push!(dom_reverse_list, bb)
                for bb_iter in bb_list
                    delete!(bb_iter.dominatees, bb)
                end
            end
        end
    end
    while !isempty(dom_reverse_list)
        bb = shift!(dom_reverse_list)
        for suc in bb.successors
            if !(bb in suc[2].im_doms)
                push!(bb.df, suc[2])
            end
        end
        for bb_iter in bb_list
            if bb in bb_iter.im_doms
                for df in bb_iter.df
                    if !(bb in df.im_doms)
                        push!(bb.df, df)
                    end
                end
            end
        end
    end
end

type SccContext
    counter::Int64
    index::Dict{BasicBlock, Int64}
    low_link::Dict{BasicBlock, Int64}
    stack::Vector{BasicBlock}
    on_stack::Dict{BasicBlock, Bool}
    connected::Set{Set{BasicBlock}}
    SccContext() = new(0,
                       Dict{BasicBlock, Int64}(),
                       Dict{BasicBlock, Int64}(),
                       Vector{BasicBlock}(),
                       Dict{BasicBlock, Bool}(),
                       Set{Set{BasicBlock}}())
end

function strongly_connected_components(entry::BasicBlock,
                                       get_successors_func)
    scc_context = SccContext()
    bb_list = flow_graph_to_list(entry)

    for bb in bb_list
        if !(bb in keys(scc_context.index))
            strongly_connected_components_helper(bb, scc_context, get_successors_func)
        end
    end
    return scc_context.connected
end

function get_successors_phi_insertion(bb::BasicBlock)
    return bb.df
end

function strongly_connected_components_helper(entry::BasicBlock,
                                             context::SccContext,
                                             get_successors_func)
    context.index[entry] = context.counter
    context.low_link[entry] = context.counter
    context.counter += 1
    push!(context.stack, entry)
    context.on_stack[entry] = true

    for suc in get_successors_func(entry)
        if !(suc in keys(context.index))
            strongly_connected_components_helper(suc, context, get_successors_func)
            context.low_link[entry] = min(context.low_link[suc], context.low_link[entry])
        elseif context.on_stack[suc]
            context.low_link[entry] = min(context.low_link[entry], context.index[suc])
        end
    end

    if context.low_link[entry] == context.index[entry]
        connected = Set{BasicBlock}()
        bb_iter = pop!(context.stack)
        context.on_stack[bb_iter] = false
        push!(connected, bb_iter)
        while bb_iter != entry
            bb_iter = pop!(context.stack)
            context.on_stack[bb_iter] = false
            push!(connected, bb_iter)
        end
        push!(context.connected, connected)
    end
end

type DfNode
    successors::Vector{DfNode}
    bb_set::Set{BasicBlock}
    DfNode() = new(
        Vector{DfNode}(),
        Set{BasicBlock}())

end

function locate_phi(fg_entry::BasicBlock)
    connected = strongly_connected_components(fg_entry, get_successors_phi_insertion)
    df_nodes = Vector{DfNode}()
    bb_to_df_map = Dict{BasicBlock, DfNode}()
    for connected_set in connected
        df_node = DfNode()
        df_node.bb_set = connected_set
        push!(df_nodes, df_node)
        for bb in connected_set
            bb_to_df_map[bb] = df_node
        end
    end

    bb_list = flow_graph_to_list(fg_entry)
    for bb in bb_list
        for df in bb.df
            if bb_to_df_map[bb] != bb_to_df_map[df]
                push!(bb_to_df_map[bb].successors, bb_to_df_map[df])
            end
        end
    end

    df_node_sucs = Dict{DfNode, Set{DfNode}}()
    for df_node in df_nodes
        df_node_sucs[df_node] = Set(df_node.successors)
    end

    df_node_set = Set(df_nodes)
    df_node_list = Vector{DfNode}()
    while !isempty(df_node_set)
        new_added = Set{DfNode}()
        for df in df_node_set
            if length(df_node_sucs[df]) == 0 ||
                length(df_node_sucs[df]) == 1
                push!(df_node_list, df)
                push!(new_added, df)
            end
            for to_delete in new_added
                delete!(df_node_set, to_delete)
                for df_iter in df_node_set
                    delete!(df_node_sucs[df_iter], to_delete)
                end
            end
        end
    end

    put_phi_map = Dict{DfNode, Set{Symbol}}()

    defs = Dict{DfNode, Set{Symbol}}()
    for df_node in df_node_list
        defs_this_df_node = Set{Symbol}()
        for bb in df_node.bb_set
            defs_this_df_node = union(defs_this_df_node, Set(keys(bb.defsout)))
        end
        defs[df_node] = defs_this_df_node
        put_phi_map[df_node] = Set{Symbol}()
    end

    for df_node in df_node_list
        for suc in df_node.successors
            put_phi_map[suc] = union(put_phi_map[suc], defs[df_node])
        end
    end

    put_phi_bb = Dict{BasicBlock, Set{Symbol}}()
    for df_node in df_node_list
        if length(df_node.bb_set) >= 1 ||
            Base.reduce(((x, y) -> x && y), true,
                        Base.map((x -> x in x.predecessors), df_node.bb_set))
            put_phi_map[df_node] = union(put_phi_map[df_node], defs[df_node])
        end
        for bb in df_node.bb_set
            if Base.reduce(((x, y) -> x || y), false,
                           Base.map((x -> !(x in df_node.bb_set)), bb.predecessors))
                put_phi_bb[bb] = put_phi_map[df_node]
            end
        end
    end
    return put_phi_bb
end

function insert_phi(entry::BasicBlock,
                    put_phi_bb::Dict{BasicBlock, Set{Symbol}})
    bb_list = flow_graph_to_list(entry)
    for bb in bb_list
        phis = put_phi_bb[bb]
        for phi in phis
            if phi in bb.uses
                println("bb = ", bb.id, " insert ", phi)
                insert!(bb.stmts, 1, (phi, Vector{Symbol}()))
            end
        end
    end
end

type SsaContext
    ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}}
    SsaContext() = new(
        Dict{Symbol, Tuple{Symbol, VarDef}}())
end

function print_ssa_defs(ssa_context::SsaContext)
    ssa_defs = ssa_context.ssa_defs
    for (key, def) in ssa_defs
        println("ssa_var = ", string(key),
                " def = [", string(def[1]),
                " ", def[2].assignment, " ",
                def[2].mutations)
    end
end

function compute_ssa_defs(entry::BasicBlock)
    bb_list = flow_graph_to_list(entry)
    ssa_context = SsaContext()
    for bb in bb_list
        compute_ssa_defs_basic_block(bb, ssa_context)
    end
    for (key, val) in ssa_context.ssa_defs
        println(key, " ", val[1], " ", val[2].assignment, " ", val[2].mutations)
    end
    return ssa_context
end

function compute_ssa_defs_stmt(stmt,
                               context::SsaContext,
                               sym_to_ssa_var_map::Dict{Symbol, Symbol})
    if isa(stmt, Tuple)
        sym = stmt[1]
        def = stmt[2]
        ssa_var = get_unique_sp_symbol()
        context.ssa_defs[ssa_var] = (sym, VarDef(def))
        sym_to_ssa_var_map[sym] = ssa_var
        println(stmt, " def ", sym, " to ", ssa_var)
        return stmt
    elseif isa(stmt, Symbol)
        if stmt in keys(sym_to_ssa_var_map)
            return sym_to_ssa_var_map[stmt]
        end
        return stmt
    elseif isa(stmt, Number) || isa(stmt, String)
        return stmt
    elseif isa(stmt, Expr)
        if stmt.head in Set([:(=), :(+=), :(-=), :(*=), :(/=), :(.*=), :(./=)])
            old_assigned_expr = compute_ssa_defs_stmt(assignment_get_assigned_from(stmt),
                                                      context, sym_to_ssa_var_map)
            assigned_to = assignment_get_assigned_to(stmt)
            ssa_var = get_unique_sp_symbol()
            assigned_expr = old_assigned_expr
            if isa(assigned_to, Symbol)
                if stmt.head == :(+=)
                    assigned_expr = Expr(:call, :+, assigned_to, assigned_expr)
                elseif stmt.head == :(-=)
                    assigned_expr = Expr(:call, :-, assigned_to, assigned_expr)
                elseif stmt.head == :(*=)
                    assigned_expr = Expr(:call, :*, assigned_to, assigned_expr)
                elseif stmt.head == :(/=)
                    assigned_expr = Expr(:call, :/, assigned_to, assigned_expr)
                elseif stmt.head == :(.*=)
                    assigned_expr = Expr(:call, :(.*), assigned_to, assigned_expr)
                elseif stmt.head == :(./=)
                    assigned_expr = Expr(:call, :(./), assigned_to, assigned_expr)
                end
                context.ssa_defs[ssa_var] = (assigned_to, VarDef(assigned_expr))
                sym_to_ssa_var_map[assigned_to] = ssa_var
                println(stmt, " def ", assigned_to, " to ", ssa_var)
                return Expr(stmt.head, ssa_var, assigned_expr)
            else
                @assert isa(assigned_to, Expr)
                @assert is_ref(assigned_to) || is_dot(assigned_to)
                var_mutated = ref_dot_get_mutated_var(assigned_to)

                if var_mutated != nothing
                    new_ssa_var = get_unique_sp_symbol()
                    if var_mutated in keys(sym_to_ssa_var_map)
                        mutated_ssa_var = sym_to_ssa_var_map[var_mutated]
                        @assert mutated_ssa_var in keys(context.ssa_defs)
                        context.ssa_defs[new_ssa_var] = context.ssa_defs[mutated_ssa_var]
                        context.ssa_defs[new_ssa_var][2].mutations = copy(context.ssa_defs[mutated_ssa_var][2].mutations)
                        push!(context.ssa_defs[new_ssa_var][2].mutations, stmt)
                    else
                        context.ssa_defs[new_ssa_var] = (var_mutated, VarDef(nothing))
                        push!(context.ssa_defs[new_ssa_var][2].mutations, stmt)
                    end
                end
                assigned_to = compute_ssa_defs_stmt(assigned_to, context, sym_to_ssa_var_map)

                if var_mutated != nothing
                    sym_to_ssa_var_map[var_mutated] = new_ssa_var
                    println(stmt, " def ", var_mutated, " to ", new_ssa_var)
                end
                return stmt
            end
        elseif stmt.head in Set([:call, :invoke, :call1, :foreigncall])
            arguments = call_get_arguments(stmt)
            if call_get_func_name(stmt) in Set([:+, :-, :*, :/, :(.*), :(./), :dot])
                for idx in eachindex(arguments)
                    arg = arguments[idx]
                    stmt.args[idx + 1] = compute_ssa_defs_stmt(arg, context,
                                                               sym_to_ssa_var_map)
                end
            else
                for idx in eachindex(arguments)
                    arg = arguments[idx]
                    var_mutated = nothing
                    if isa(arg, Symbol)
                        var_mutated = arg
                    elseif isa(arg, Expr) &&
                        (is_ref(arg) || is_dot(arg))
                        var_mutated = ref_dot_get_mutated_var(arg)
                    end
                    stmt.args[idx + 1] = compute_ssa_defs_stmt(arg, context,
                                                               sym_to_ssa_var_map)
                    if var_mutated != nothing
                        new_ssa_var = get_unique_sp_symbol()
                        if var_mutated in keys(sym_to_ssa_var_map)
                            mutated_ssa_var = sym_to_ssa_var_map[var_mutated]
                            @assert mutated_ssa_var in keys(context.ssa_defs)
                            mutated_ssa_var = sym_to_ssa_var_map[var_mutated]
                            context.ssa_defs[new_ssa_var] = context.ssa_defs[mutated_ssa_var]
                            context.ssa_defs[new_ssa_var][2].mutations = copy(context.ssa_defs[mutated_ssa_var][2].mutations)
                            push!(context.ssa_defs[new_ssa_var][2].mutations, stmt)
                        else
                            context.ssa_defs[new_ssa_var] = (var_mutated, VarDef(nothing))
                            push!(context.ssa_defs[new_ssa_var][2].mutations, stmt)
                        end
                        sym_to_ssa_var_map[var_mutated] = new_ssa_var
                        println(stmt, " def ", var_mutated, " to ", new_ssa_var)
                    end
                end
            end
            return stmt
        else
            stmt = AstWalk.ast_walk(stmt, remap_symbols_visit, sym_to_ssa_var_map)
            return stmt
        end
    end
end

function remap_symbols_visit(expr::Any,
                             symbol_map::Dict{Symbol, Symbol},
                             top_level::Integer,
                             is_top_level::Bool,
                             read::Bool)

    if isa(expr, Symbol)
        if expr in keys(symbol_map)
            return symbol_map[expr]
        end
    elseif isa(expr, Expr)
        if expr.head == :line
            return expr
        end
        return AstWalk.AST_WALK_RECURSE
    end
    return expr
end

function compute_ssa_defs_basic_block(bb::BasicBlock,
                                      context::SsaContext)
    sym_to_ssa_var_map = bb.sym_to_ssa_var_map
    for idx in eachindex(bb.stmts)
        stmt = bb.stmts[idx]
        bb.stmts[idx] = compute_ssa_defs_stmt(stmt, context,
                                              sym_to_ssa_var_map)
    end
    for (sym, def) in bb.defsout
        @assert sym in keys(sym_to_ssa_var_map)
        push!(bb.ssa_defsout, sym_to_ssa_var_map[sym])
    end
    for stmt in bb.stmts
        if isa(stmt, Tuple)
            sym = stmt[1]
            @assert sym in keys(sym_to_ssa_var_map)
            push!(bb.ssa_defsout, sym_to_ssa_var_map[sym])
            push!(bb.killed, sym)
        end
    end
end

function compute_ssa_reaches(entry::BasicBlock,
                             context::SsaContext)
    bb_list = flow_graph_to_list(entry)
    changed = true
    while changed
        changed = false
        for bb in bb_list
            new_ssa_reaches = bb.ssa_reaches
            for pred in bb.predecessors
                if pred != nothing
                    new_ssa_reaches = union(new_ssa_reaches,
                                            pred.ssa_defsout,
                                            pred.ssa_reaches_alive)
                end

            end
            if new_ssa_reaches != bb.ssa_reaches
                changed = true
                bb.ssa_reaches = new_ssa_reaches
                bb.ssa_reaches_alive = Set{Symbol}()
                for ssa_sym in bb.ssa_reaches
                    sym = context.ssa_defs[ssa_sym][1]
                    if !(sym in bb.killed)
                        push!(bb.ssa_reaches_alive, ssa_sym)
                    end
                end
            end
        end
    end

    print_ssa_defs(context)
    for bb in bb_list
        propagate_ssa_reaches(bb, context)
    end
    print_flow_graph(entry)
end

function propagate_ssa_reaches_stmt(stmt,
                                    sym::Symbol,
                                    ssa_syms::Vector{Symbol})
    if isa(stmt, Tuple)
        if stmt[1] == sym
            append!(stmt[2], ssa_syms)
        end
        return stmt
    elseif isa(stmt, Symbol)
        if stmt == sym
            return ssa_syms[1]
        end
        return stmt
    elseif isa(stmt, Number) || isa(stmt, String)
        return stmt
    elseif isa(stmt, Expr)
        remap_dict = Dict(sym => ssa_syms[1])
        stmt = AstWalk.ast_walk(stmt, remap_symbols_visit, remap_dict)
        return stmt
    else
        remap_dict = Dict(sym => ssa_syms[1])
        stmt = AstWalk.ast_walk(stmt, remap_symbols_visit, remap_dict)
        return stmt
    end
end

function propagate_ssa_reaches(bb::BasicBlock,
                               context::SsaContext)
    println("propagate for bb ", bb.id)
    ssa_reaches_dict = bb.ssa_reaches_dict
    ssa_defs = context.ssa_defs
    for ssa_sym in bb.ssa_reaches
        sym = ssa_defs[ssa_sym][1]
        if !(sym in keys(bb.ssa_reaches_dict))
            ssa_reaches_dict[sym] = Vector{Symbol}()
        end
        push!(ssa_reaches_dict[sym], ssa_sym)
    end

    for (sym, ssa_syms) in ssa_reaches_dict
        for idx in eachindex(bb.stmts)
            stmt = bb.stmts[idx]
            bb.stmts[idx] = propagate_ssa_reaches_stmt(stmt, sym, ssa_syms)
        end
    end
end
