import Base: copy

type VarDef
    assignment
    mutation
    VarDef(assigned) = new(assigned, nothing)
end

type BasicBlock
    id::Int64
    predecessors::Vector{Any}
    successors::Vector{Tuple{Any, BasicBlock}}
    num_backward_edges::Int64
    stmts::Vector{Any}
    control_flow
    uses::Set{Symbol}
    defsout::Dict{Symbol, VarDef}
    killed::Set{Symbol}
    dominators::Set{BasicBlock}
    im_doms::Set{BasicBlock}
    dominatees::Set{BasicBlock}
    df::Set{BasicBlock}
    ssa_defsout::Set{Symbol}
    ssa_defs::Set{Symbol}
    sym_to_ssa_var_map::Dict{Symbol, Symbol}
    reaches::Set{Symbol}
    ssa_reaches::Set{Symbol}
    ssa_reaches_alive::Set{Symbol}
    ssa_reaches_dict::Dict{Symbol, Vector{Symbol}}
    # mapping stmt index to symbols defined or used
    stmt_ssa_defs::Dict{Int64, Set{Symbol}}
    stmt_ssa_uses::Dict{Int64, Set{Symbol}}
    BasicBlock(id) = new(id,
                         Vector{BasicBlock}(),
                         Vector{Tuple{Any, BasicBlock}}(),
                         0,
                         Vector{Any}(),
                         nothing,
                         Set{Symbol}(),
                         Dict{Symbol, VarDef}(),
                         Set{Symbol}(),
                         Set{BasicBlock}(),
                         Set{BasicBlock}(),
                         Set{BasicBlock}(),
                         Set{BasicBlock}(),
                         Set{Symbol}(),
                         Set{Symbol}(),
                         Dict{Symbol, Symbol}(),
                         Set{Symbol}(),
                         Set{Symbol}(),
                         Set{Symbol}(),
                         Dict{Symbol, Vector{Symbol}}(),
                         Dict{Int64, Set{Symbol}}(),
                         Dict{Int64, Set{Symbol}}())
end

function copy(bb::BasicBlock)::BasicBlock
    new_bb = BasicBlock(bb.id)
    new_bb.predecessors = copy(bb.predecessors)
    new_bb.successors = copy(bb.sucessors)
    new_bb.num_backward_edges = bb.num_backward_edges
    new_bb.stmts = copy(bb.stmts)
    control_flow = bb.control_flow
    new_bb.uses = copy(bb.uses)
    new_bb.defsout = copy(bb.defsout)
    new_bb.killed = copy(bb.killed)
    new_bb.dominators = copy(bb.dominators)
    new_bb.im_doms = copy(bb.im_doms)
    new_bb.dominatees = copy(bb.dominatees)
    new_bb.df = copy(bb.df)
    new_bb.ssa_defsout = copy(bb.ssa_defsout)
    new_bb.ssa_defs = copy(bb.ssa_defs)
    new_bb.sym_to_ssa_var_map = copy(bb.sym_to_ssa_var_map)
    new_bb.reaches = copy(bb.reaches)
    new_bb.ssa_reaches = copy(bb.ssa_reaches)
    new_bb.ssa_reaches_alive = copy(bb.ssa_reaches_alive)
    new_bb.ssa_reaches_dict = copy(bb.ssa_reaches_dict)
    new_bb.stmt_ssa_defs = copy(bb.stmt_ssa_defs)
    new_bb.stmt_ssa_uses = copy(bb.stmt_ssa_uses)
end

type FlowGraphContext
    bb_counter::Int64
    FlowGraphContext() = new(0)
end

function create_basic_block(build_context::FlowGraphContext)::BasicBlock
    bb = BasicBlock(build_context.bb_counter)
    build_context.bb_counter += 1
    return bb
end

function build_flow_graph(expr::Expr)
    build_context = FlowGraphContext()
    graph_entry = create_basic_block(build_context)
    push!(graph_entry.predecessors, nothing)
    graph_exits = build_flow_graph(expr, graph_entry, build_context)
    return graph_entry, graph_exits, build_context
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
    println("control_flow = ", bb.control_flow)
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

function print_flow_graph(bb_list::Vector{BasicBlock})
    for bb in bb_list
        print_basic_block(bb)
    end
end

function compute_use_def(bb_list::Vector{BasicBlock})
    for bb in bb_list
        compute_use_def_bb(bb)
    end
end

function build_flow_graph(expr::Expr, bb::BasicBlock,
                          build_context::FlowGraphContext)::Vector{BasicBlock}
    exit_bbs = Vector{BasicBlock}()
    if !isa(expr, Expr)
        push!(bb.stmts, copy(expr))
        push!(exit_bbs, bb)
        return exit_bbs
    end
    if expr.head == :if
        condition = if_get_condition(expr)
        push!(bb.stmts, copy(condition))
        true_bb_entry = create_basic_block(build_context)

        true_bb_exits = build_flow_graph(if_get_true_branch(expr),
                                         true_bb_entry,
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
    elseif expr.head == :for ||
        expr.head == :while
        loop_condition = for_get_loop_condition(expr)
        if isempty(bb.stmts)
            loop_condition_bb = bb
        else
            loop_condition_bb = create_basic_block(build_context)
            push!(bb.successors, (nothing, loop_condition_bb))
            push!(loop_condition_bb.predecessors, bb)
        end
        loop_condition_bb.control_flow = expr.head
        push!(loop_condition_bb.stmts, copy(loop_condition))
        true_bb_entry = create_basic_block(build_context)
        true_bb_exits = build_flow_graph(for_get_loop_body(expr),
                                         true_bb_entry,
                                         build_context)
        push!(loop_condition_bb.successors, (true, true_bb_entry))
        push!(true_bb_entry.predecessors, loop_condition_bb)
        loop_condition_bb.num_backward_edges = length(true_bb_exits)
        for exit_bb in true_bb_exits
            push!(loop_condition_bb.predecessors, exit_bb)
            push!(exit_bb.successors, (nothing, loop_condition_bb))
        end

        loop_exit_bb = create_basic_block(build_context)
        push!(loop_condition_bb.successors, (false, loop_exit_bb))
        push!(loop_exit_bb.predecessors, loop_condition_bb)

        push!(exit_bbs, loop_exit_bb)
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
    else
        push!(bb.stmts, copy(expr))
        push!(exit_bbs, bb)
    end
    return exit_bbs
end

function compute_use_def_expr(stmt, bb::BasicBlock)
    uses = bb.uses
    defsout = bb.defsout
    killed = bb.killed

    if isa(stmt, Symbol)
        if !(stmt in keys(defsout))
            push!(uses, stmt)
        end
    elseif isa(stmt, Expr)
        if is_assignment(stmt)
            assigned_to = assignment_get_assigned_to(stmt)
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
            compute_use_def_expr(assigned_expr, bb)
            if isa(assigned_to, Symbol)
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
                        defsout[var_mutated] = VarDef(defsout[var_mutated])
                        defsout[var_mutated].mutation = stmt
                    else
                        defsout[var_mutated] = VarDef(nothing)
                        defsout[var_mutated].mutation = stmt
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
                            defsout[var_mutated] = VarDef(defsout[var_mutated])
                            defsout[var_mutated].mutation = stmt
                        else
                            defsout[var_mutated] = VarDef(nothing)
                            defsout[var_mutated].mutation = stmt
                            push!(uses, var_mutated)
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

function compute_use_def_bb(bb::BasicBlock)
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

function compute_dominators(bb_list::Vector{BasicBlock})
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

function compute_im_doms(bb_list::Vector{BasicBlock})
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

function compute_dominatees(bb_list::Vector{BasicBlock})
    for bb in bb_list
        for dom in bb.dominators
            push!(dom.dominatees, bb)
        end
    end
end

function construct_dominance_frontier(bb_list::Vector{BasicBlock})
    compute_dominatees(bb_list)
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

function strongly_connected_components(bb_list::Vector{BasicBlock},
                                       get_successors_func)
    scc_context = SccContext()

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

function strongly_connected_components_helper(bb::BasicBlock,
                                             context::SccContext,
                                             get_successors_func)
    context.index[bb] = context.counter
    context.low_link[bb] = context.counter
    context.counter += 1
    push!(context.stack, bb)
    context.on_stack[bb] = true

    for suc in get_successors_func(bb)
        if !(suc in keys(context.index))
            strongly_connected_components_helper(suc, context, get_successors_func)
            context.low_link[bb] = min(context.low_link[suc], context.low_link[bb])
        elseif context.on_stack[suc]
            context.low_link[bb] = min(context.low_link[bb], context.index[suc])
        end
    end

    if context.low_link[bb] == context.index[bb]
        connected = Set{BasicBlock}()
        bb_iter = pop!(context.stack)
        context.on_stack[bb_iter] = false
        push!(connected, bb_iter)
        while bb_iter != bb
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

function locate_phi(bb_list::Vector{BasicBlock})
    connected = strongly_connected_components(bb_list, get_successors_phi_insertion)
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

function insert_phi(bb_list::Vector{BasicBlock},
                    put_phi_bb::Dict{BasicBlock, Set{Symbol}})
    for bb in bb_list
        phis = put_phi_bb[bb]
        for phi in phis
            if phi in bb.uses
                insert!(bb.stmts, 1, (phi, Vector{Symbol}()))
            end
        end
    end
end

type SsaContext
    # ssa symbol to (variable, variable definition)
    ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}}
    SsaContext() = new(
        Dict{Symbol, Tuple{Symbol, VarDef}}())
end

function print_ssa_defs(ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}})
    for (key, def) in ssa_defs
        println("ssa_var = ", string(key),
                " def = [", string(def[1]),
                " ", def[2].assignment, " ",
                def[2].mutation, "]")
    end
end

function compute_ssa_defs(bb_list::Vector{BasicBlock})
    ssa_context = SsaContext()
    for bb in bb_list
        compute_ssa_defs_basic_block(bb, ssa_context)
    end
    return ssa_context
end

function compute_ssa_defs_stmt(stmt,
                               context::SsaContext,
                               sym_to_ssa_var_map::Dict{Symbol, Symbol},
                               bb_ssa_defs::Set{Symbol},
                               stmt_ssa_defs::Set{Symbol})
    if isa(stmt, Tuple)
        sym = stmt[1]
        def = stmt[2]
        ssa_var = gen_unique_sp_symbol()
        context.ssa_defs[ssa_var] = (sym, VarDef(def))
        push!(bb_ssa_defs, ssa_var)
        push!(stmt_ssa_defs, ssa_var)
        sym_to_ssa_var_map[sym] = ssa_var
        return (sym, def, ssa_var)
    elseif isa(stmt, Symbol)
        if stmt in keys(sym_to_ssa_var_map)
            return sym_to_ssa_var_map[stmt]
        end
        return stmt
    elseif isa(stmt, Number) || isa(stmt, String)
        return stmt
    elseif isa(stmt, Expr)
        if is_assignment(stmt)
            assigned_to = assignment_get_assigned_to(stmt)
            assigned_expr = assignment_get_assigned_from(stmt)
            assigned_to_copy = isa(assigned_to, Expr) ? copy(assigned_to) : assigned_to
            if stmt.head == :(+=)
                assigned_expr = Expr(:call, :+, assigned_to_copy, assigned_expr)
            elseif stmt.head == :(-=)
                assigned_expr = Expr(:call, :-, assigned_to_copy, assigned_expr)
            elseif stmt.head == :(*=)
                assigned_expr = Expr(:call, :*, assigned_to_copy, assigned_expr)
            elseif stmt.head == :(/=)
                assigned_expr = Expr(:call, :/, assigned_to_copy, assigned_expr)
            elseif stmt.head == :(.*=)
                assigned_expr = Expr(:call, :(.*), assigned_to_copy, assigned_expr)
            elseif stmt.head == :(./=)
                assigned_expr = Expr(:call, :(./), assigned_to_copy, assigned_expr)
            end

            assigned_expr = compute_ssa_defs_stmt(assigned_expr,
                                                  context, sym_to_ssa_var_map, bb_ssa_defs,
                                                  stmt_ssa_defs)
            if isa(assigned_to, Symbol)
                ssa_var = gen_unique_sp_symbol()
                context.ssa_defs[ssa_var] = (assigned_to, VarDef(assigned_expr))
                push!(bb_ssa_defs, ssa_var)
                push!(stmt_ssa_defs, ssa_var)
                sym_to_ssa_var_map[assigned_to] = ssa_var
                return :($ssa_var = $assigned_expr)
            else
                @assert isa(assigned_to, Expr)
                @assert is_ref(assigned_to) || is_dot(assigned_to)
                var_mutated = ref_dot_get_mutated_var(assigned_to)

                if var_mutated != nothing
                    new_ssa_var = gen_unique_sp_symbol()
                    if var_mutated in keys(sym_to_ssa_var_map)
                        mutated_ssa_var = sym_to_ssa_var_map[var_mutated]
                        @assert mutated_ssa_var in keys(context.ssa_defs)
                        context.ssa_defs[new_ssa_var] = (context.ssa_defs[mutated_ssa_var][1],
                                                         VarDef(context.ssa_defs[mutated_ssa_var][2]))
                        context.ssa_defs[new_ssa_var][2].mutation = stmt
                        push!(bb_ssa_defs, new_ssa_var)
                    else
                        context.ssa_defs[new_ssa_var] = (var_mutated, VarDef(nothing))
                        context.ssa_defs[new_ssa_var][2].mutation = stmt
                        push!(bb_ssa_defs, new_ssa_var)
                    end
                    push!(stmt_ssa_defs, new_ssa_var)
                    sym_to_ssa_var_map[var_mutated] = new_ssa_var
                end
                assigned_to = compute_ssa_defs_stmt(assigned_to, context, sym_to_ssa_var_map,
                                                    bb_ssa_defs, stmt_ssa_defs)

                return :($assigned_to = $assigned_expr)
            end
        elseif stmt.head in Set([:call, :invoke, :call1, :foreigncall])
            arguments = call_get_arguments(stmt)
            if call_get_func_name(stmt) in Set([:+, :-, :*, :/, :(.*), :(./), :dot, :eachindex])
                for idx in eachindex(arguments)
                    arg = arguments[idx]
                    stmt.args[idx + 1] = compute_ssa_defs_stmt(arg, context,
                                                               sym_to_ssa_var_map,
                                                               bb_ssa_defs,
                                                               stmt_ssa_defs)
                end
            else
                for idx in eachindex(arguments)
                    arg = arguments[idx]
                    var_mutated = nothing
                    if isa(arg, Symbol)
                        var_mutated = arg
                    elseif isa(arg, Expr) && (is_ref(arg) || is_dot(arg))
                        var_mutated = ref_dot_get_mutated_var(arg)
                    end
                    #arg_copy = isa(arg, Expr) ? copy(arg) : arg
                    arg_copy = arg
                    stmt.args[idx + 1] = compute_ssa_defs_stmt(arg_copy, context,
                                                               sym_to_ssa_var_map,
                                                               bb_ssa_defs,
                                                               stmt_ssa_defs)
                    if var_mutated != nothing
                        new_ssa_var = gen_unique_sp_symbol()
                        if var_mutated in keys(sym_to_ssa_var_map)
                            mutated_ssa_var = sym_to_ssa_var_map[var_mutated]
                            @assert mutated_ssa_var in keys(context.ssa_defs)
                            mutated_ssa_var = sym_to_ssa_var_map[var_mutated]
                            context.ssa_defs[new_ssa_var] = (context.ssa_defs[mutated_ssa_var][1],
                                                             VarDef(context.ssa_defs[mutated_ssa_var][2]))
                            context.ssa_defs[new_ssa_var][2].mutation = stmt
                            push!(bb_ssa_defs, new_ssa_var)
                        else
                            context.ssa_defs[new_ssa_var] = (var_mutated, VarDef(nothing))
                            context.ssa_defs[new_ssa_var][2].mutation = stmt
                            push!(bb_ssa_defs, new_ssa_var)
                        end
                        push!(stmt_ssa_defs, new_ssa_var)
                        sym_to_ssa_var_map[var_mutated] = new_ssa_var
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
        stmt_ssa_defs = Set{Symbol}()
        bb.stmts[idx] = compute_ssa_defs_stmt(stmt, context,
                                              sym_to_ssa_var_map,
                                              bb.ssa_defs,
                                              stmt_ssa_defs)
        if !isempty(stmt_ssa_defs)
            bb.stmt_ssa_defs[idx] = stmt_ssa_defs
        end
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

function compute_ssa_reaches(bb_list::Vector{BasicBlock},
                             context::SsaContext)
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

    for bb in bb_list
        propagate_ssa_reaches(bb, context)
    end
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
        for ssa_var in bb.ssa_defs
            ssa_def = ssa_defs[ssa_var]
            if ssa_def[1] == sym &&
                ssa_def[2].assignment == nothing
                ssa_def[2].assignment = ssa_syms[1]
            end
        end
    end
end

function compute_stmt_ssa_uses(bb_list::Vector{BasicBlock})
    for bb in bb_list
        for idx in eachindex(bb.stmts)
            stmt = bb.stmts[idx]
            stmt_ssa_uses = Set{Symbol}()
            compute_stmt_ssa_uses_stmt(stmt,
                                       stmt_ssa_uses)

            if !isempty(stmt_ssa_uses)
                bb.stmt_ssa_uses[idx] = stmt_ssa_uses
            end
        end
    end
end

function get_stmt_ssa_defuses_visit(expr::Any,
                                    stmt_ssa_uses::Set{Symbol},
                                    top_level::Integer,
                                    is_top_level::Bool,
                                    read::Bool)

    if isa(expr, Symbol)
        push!(stmt_ssa_uses, expr)
    elseif isa(expr, Expr)
        if expr.head == :line
            return expr
        end
        return AstWalk.AST_WALK_RECURSE
    end
    return expr
end

function compute_stmt_ssa_uses_stmt(stmt,
                                    stmt_ssa_uses::Set{Symbol})
    if isa(stmt, Tuple)
        sym = stmt[1]
        def = stmt[2]
        ssa_sym = stmt[3]
        for use_sym in def
            push!(stmt_ssa_uses, use_sym)
        end
    elseif isa(stmt, Symbol)
        push!(stmt_ssa_uses, stmt)
    elseif isa(stmt, Number) || isa(stmt, String)
    elseif isa(stmt, Expr)
        if is_assignment(stmt)
            @assert stmt.head == :(=)
            compute_stmt_ssa_uses_stmt(assignment_get_assigned_from(stmt),
                                       stmt_ssa_uses)
            assigned_to = assignment_get_assigned_to(stmt)
            if isa(assigned_to, Expr)
                @assert isa(assigned_to, Expr)
                @assert is_ref(assigned_to) || is_dot(assigned_to)
                var_mutated = ref_dot_get_mutated_var(assigned_to)
                if var_mutated != nothing
                    @assert isa(var_mutated, Symbol)
                    push!(stmt_ssa_uses, var_mutated)
                end
                compute_stmt_ssa_uses_stmt(assigned_to, stmt_ssa_uses)
            end
        elseif stmt.head in Set([:call, :invoke, :call1, :foreigncall])
            arguments = call_get_arguments(stmt)
            for arg in arguments
                compute_stmt_ssa_uses_stmt(arg, stmt_ssa_uses)
            end
        else
            stmt = AstWalk.ast_walk(stmt, get_stmt_ssa_defuses_visit, stmt_ssa_uses)
        end
    end
end

function flow_analysis(expr::Expr)
    flow_graph_entry, flow_graph_exits, context = build_flow_graph(expr)
    bb_list = flow_graph_to_list(flow_graph_entry)
    compute_use_def(bb_list)
    compute_dominators(bb_list)
    compute_im_doms(bb_list)

    construct_dominance_frontier(bb_list)
    put_phi_here = locate_phi(bb_list)

    insert_phi(bb_list, put_phi_here)
    ssa_context = compute_ssa_defs(bb_list)
    compute_ssa_reaches(bb_list, ssa_context)
    compute_stmt_ssa_uses(bb_list)
    return flow_graph_entry, context, ssa_context
end

#function traverse_for_loop(loop_entry::BasicBlock,
#                           callback,
#                           cbdata)
#    bb_list = Vector{BasicBlock}()
#    push!(bb_list, loop_entry)
#    for suc in loop_entry.successors
#        if suc[1] == true
#            push!(bb_list, suc[2])
#        end
#    end

#    visited = Set{Int64}()
#    while !isempty(bb_list)
#        bb = shift!(bb_list)
#        if bb.id in visited
#            continue
#        end
#        push!(visited, bb.id)
#        callback(bb, cbdata)
#        for suc in bb.successors
#            push!(bb_list, suc[2])
#        end
#    end
#end

# returns a tuple (sub_value, loop_index_dim, offset)
function eval_subscript_expr(expr,
                             iteration_var::Symbol,
                             ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}})
    if isa(expr, Symbol)
        if expr == :(:)
            return (DistArrayAccessSubscript_value_any, nothing, nothing)
        elseif expr == iteration_var
            return (expr, nothing, nothing)
        elseif expr in keys(ssa_defs)
            def = ssa_defs[expr][2]
            if ssa_defs[expr][1] == iteration_var
                return (iteration_var, nothing, nothing)
            end
            if def.assignment != nothing &&
                def.mutation == nothing
                return eval_subscript_expr(def.assignment, iteration_var, ssa_defs)
            else
                return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
            end
        elseif isdefined(current_module(), expr)
            return eval_subscript_expr(eval(current_module(), expr),
                                       iteration_var, ssa_defs)
        else
            return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
            #error("accessing undefined var ", expr)
        end
    elseif isa(expr, Number)
        return (DistArrayAccessSubscript_value_static, nothing, expr)
    elseif isa(expr, Expr)
        head = expr.head
        if head == :ref
            referenced_var = ref_get_referenced_var(expr)
            subscripts = ref_get_subscripts(expr)
            evaled_referenced_var = eval_subscript_expr(referenced_var,
                                                        iteration_var,
                                                        ssa_defs)
            if length(subscripts) == 1 &&
                isa(subscripts[1], Number)
                sub_val = subscripts[1]
            else
                return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
            end

            if evaled_referenced_var[1] == iteration_var
                if sub_val == 1
                    return (:($iteration_var[1]), nothing, nothing)
                else
                    return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
                end
            elseif isa(evaled_referenced_var[1], Expr) &&
                evaled_referenced_var[1] == :($iteration_var[1])
                return (DistArrayAccessSubscript_value_static, sub_val, 0)
            else
                return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
            end
        elseif head == :call
            func_name = call_get_func_name(expr)
            if func_name in Set([:+, :-, :*, :/])
                arg1 = call_get_arguments(expr)[1]
                arg2 = call_get_arguments(expr)[2]
                arg1 = eval_subscript_expr(arg1, iteration_var, ssa_defs)
                arg2 = eval_subscript_expr(arg2, iteration_var, ssa_defs)
                if arg1[1] == DistArrayAccessSubscript_value_static &&
                    arg2[1] == DistArrayAccessSubscript_value_static
                    if arg1[2] == nothing && arg2[2] == nothing
                        return (DistArrayAccessSubscript_value_static, nothing,
                                eval(Expr(:call,
                                          func_name,
                                          arg1[3],
                                          arg2[3])))
                    else
                        if func_name in Set([:*, :/])
                            return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
                        end

                        if arg1[2] == nothing
                            return (DistArrayAccessSubscript_value_static, arg2[2],
                                    eval(Expr(:call,
                                              func_name,
                                              arg1[3],
                                              arg2[3])))
                        elseif arg2[2] == nothing
                            return (DistArrayAccessSubscript_value_static, arg1[2],
                                    eval(Expr(:call,
                                              func_name,
                                              arg1[3],
                                              arg2[3])))
                        else
                            return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
                        end
                    end
                elseif arg1[1] == DistArrayAccessSubscript_value_any &&
                    arg2[1] == DistArrayAccessSubscript_value_any
                    return (DistArrayAccessSubscript_value_any, nothing, nothing)
                else
                    return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
                end
            else
                return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
            end
        else
            return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
        end
    elseif isa(expr, Tuple)
        return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
    else
        return (DistArrayAccessSubscript_value_unknown, nothing, nothing)
    end
end

# Deleted symbols are SSA variables whose definition depends on
# reads from a DistArray other than the iteration space
# and SSA variables that recursively depend on them.

function get_deleted_syms(bb_list::Vector{BasicBlock})
    syms_deleted = Set{Symbol}()
    for bb in bb_list
        if !(bb.id in keys(bb_dist_array_access_dict))
            continue
        end
        stmt_access_dict = bb_dist_array_access_dict[bb.id]
        for (stmt_idx, access_dict) in stmt_access_dict
            read = false
            for (da_sym, access_vec) in access_dict
                for access in access_vec
                    if access.is_read
                        read = true
                        break
                    end
                end
            end
            if !read
                continue
            end
            #TODO: more accurate!
            if stmt_idx in keys(bb.stmt_ssa_defs)
                union!(syms_deleted, bb.stmt_ssa_defs[stmt_idx])
            end
        end
    end

    syms_are_deleted = !isempty(syms_deleted)
    while syms_are_deleted
        syms_are_deleted = false
        for bb in bb_list
            if bb.control_flow == (:if, :else) ||
                bb.control_flow == :if ||
                bb.control_flow == :for ||
                bb.control_flow == :while
                condition_stmt_idx = length(bb.stmts)
                uses_deleted_syms = false
                if condition_stmt_idx in keys(bb.stmt_ssa_uses)
                    uses_deleted_syms = !isempty(intersect(bb.stmt_ssa_uses[condition_stmt_idx],
                                                           syms_deleted))
                end
                if !uses_deleted_syms
                    continue
                end
                if condition_stmt_idx in keys(bb.stmt_ssa_defs)
                    new_syms_deleted = union(syms_deleted, bb.stmt_ssa_defs[condition_stmt_idx])
                end
                for suc in bb.successors
                    @assert isa(suc, Tuple)
                    if !isa(suc[1], Bool)
                        continue
                    end
                    suc_syms_deleted = suc[2].ssa_defs
                    new_syms_deleted = union(new_syms_deleted, suc_syms_deleted)
                end
                if new_syms_deleted != syms_deleted
                    syms_are_deleted = true
                    syms_deleted = new_syms_deleted
                end
            else
                for stmt_idx in eachindex(bb.stmts)
                    uses_deleted_syms = false
                    if stmt_idx in keys(bb.stmt_ssa_uses)
                        uses_deleted_syms = !isempty(intersect(bb.stmt_ssa_uses[stmt_idx], syms_deleted))
                    end
                    if !uses_deleted_syms
                        continue
                    end
                    if stmt_idx in keys(bb.stmt_ssa_defs)
                        new_syms_deleted = union(syms_deleted, bb.stmt_ssa_defs[stmt_idx])
                    end
                    if new_syms_deleted != syms_deleted
                        syms_are_deleted = true
                        syms_deleted = new_syms_deleted
                    end
                end
            end
        end
    end
    return syms_deleted
end


function remap_ssa_vars_visit(expr::Any,
                              ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}},
                              top_level::Integer,
                              is_top_level::Bool,
                              read::Bool)
    if isa(expr, Symbol) &&
        expr in keys(ssa_defs)
        return ssa_defs[expr][1]
    elseif isa(expr, Expr)
        if expr.head == :line
            return expr
        end
        return AstWalk.AST_WALK_RECURSE
    end
    return expr
end

function remap_ssa_vars(expr, ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}})
    return AstWalk.ast_walk(isa(expr, Expr) ? copy(expr) : expr, remap_ssa_vars_visit, ssa_defs)
end


function transform_dist_array_read_set(dist_array_read::Tuple,
                                       ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}})
    da_sym = dist_array_read[1]
    subscripts_vec = dist_array_read[2]
    record_access_stmt = :(oriongen_prefetch_point_dict[$da_sym.id][])

    for sub in subscripts_vec
        #remapped_sub = remap_ssa_vars(sub, ssa_defs)
        push!(record_access_stmt.args, sub)
    end
    temp_var = gen_unique_sp_symbol()
    stmt_block = quote
        $temp_var = $record_access_stmt
    end
    return stmt_block.args
end

function transform_dist_array_read_dict(dist_array_read::Tuple,
                                        ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}})
    da_sym = dist_array_read[1]
    subscripts_vec = dist_array_read[2]
    record_access_stmt = :(oriongen_access_count_dict[$da_sym.id][])

    for sub in subscripts_vec
        #remapped_sub = remap_ssa_vars(sub, ssa_defs)
        push!(record_access_stmt.args, sub)
    end
    temp_var = gen_unique_sp_symbol()
    return [:($temp_var = $record_access_stmt)]
end

function recreate_stmts_from_flow_graph(bb::BasicBlock,
                                        bb_stmts_dict::Dict{Int64, Dict{Int64, Any}},
                                        transform_dist_array_read_func::Function,
                                        stmt_vec::Vector{Any},
                                        appended_bbs::Set{Int64},
                                        ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}})
    println("recreate_stmts_from_flow_graph ", bb.id)
    if bb.id in keys(bb_stmts_dict) &&
        length(bb_stmts_dict[bb.id]) > 0
        stmt_dict = bb_stmts_dict[bb.id]
        for stmt_idx = eachindex(bb.stmts)
            if !(stmt_idx in keys(stmt_dict)) ||
                (bb.control_flow != nothing && stmt_idx == length(bb.stmts))
                continue
            end
            stmt = stmt_dict[stmt_idx]
            if isa(stmt, Vector)
                for dist_array_read in stmt
                    @assert isa(dist_array_read, Tuple)
                    transformed_read_stmt = transform_dist_array_read_func(dist_array_read, ssa_defs)
                    append!(stmt_vec, transformed_read_stmt)
                end
            elseif !isa(stmt, Tuple)
                push!(stmt_vec, stmt)
            end
        end
    end
    push!(appended_bbs, bb.id)
    unhandled_suc = nothing

    if bb.control_flow == :if
        if_stmt = :(if $(bb.stmts[end]) end)
        if_stmt_vec = if_stmt.args[2].args
        @assert length(bb.successors) == 2
        branch_unhandled_suc = nothing
        default_suc = nothing
        true_branch_stmt_vec = Vector{Any}()
        for suc in bb.successors
            @assert isa(suc, Tuple{Bool, BasicBlock})
            if suc[1]
                branch_unhandled_suc = recreate_stmts_from_flow_graph(suc[2],
                                                                      bb_stmts_dict,
                                                                      transform_dist_array_read_func,
                                                                      true_branch_stmt_vec,
                                                                      appended_bbs,
                                                                      ssa_defs)
            end
        end
        if !isempty(true_branch_stmt_vec)
            append!(if_stmt_vec, true_branch_stmt_vec)
            push!(stmt_vec, if_stmt_vec)
        end
        push!(stmt_vec, if_stmt)
        for suc in bb.successors
            if !suc[1]
                default_suc = suc[2]
                unhandled_suc = recreate_stmts_from_flow_graph(suc[2],
                                                               bb_stmts_dict,
                                                               transform_dist_array_read_func,
                                                               stmt_vec,
                                                               appended_bbs,
                                                               ssa_defs)
            end
        end
        @assert default_suc != nothing &&
            (unhandled_suc == nothing ||
             unhandled_suc == default_suc)
    elseif bb.control_flow == :(:if, :else)
        if_stmt = :(if $(bb.stmts[end]) else end)
        @assert length(bb.successors) == 2
        true_unhandled_suc = nothing
        false_unhandled_suc = nothing
        true_branch_stmt_vec = Vector{Any}()
        false_branch_stmt_vec = Vector{Any}()
        for suc in bb.successors
            @assert isa(suc, Tuple{Bool, BasicBlock})
            if suc[1]
                true_unhandled_suc = recreate_stmts_from_flow_graph(suc[2],
                                                                    bb_stmts_dict,
                                                                    transform_dist_array_read_func,
                                                                    true_branch_stmt_vec,
                                                                    appended_bbs,
                                                                    ssa_defs)
            else
                false_unhandled_suc = recreate_stmts_from_flow_graph(suc[2],
                                                                     bb_stmts_dict,
                                                                     transform_dist_array_read_func,
                                                                     false_branch_stmt_vec,
                                                                     appended_bbs,
                                                                     ssa_defs)

            end
        end
        @assert true_unhandled_suc == false_unhandled_suc
        unhandled_suc = true_unhandled_suc
        if !isempty(true_branch_stmt_vec)
            append!(if_stmt.args[2].args, true_branch_stmt_vec)
        end
        if !isempty(false_branch_stmt_vec)
            append!(if_stmt.args[3].args, false_branch_stmt_vec)
        end
        if !isempty(true_branch_stmt_vec) ||
            !isempty(false_branch_stmt_vec)
            append!(stmt_vec, if_stmt)
        end
    elseif bb.control_flow == :while ||
        bb.control_flow == :for
        if bb.control_flow == :while
            loop_stmt = :(while $(bb.stmts[end]) end)
        else
            loop_stmt = :(for i = 0:1
                          end)
            loop_stmt.args[1] = bb.stmts[end]
        end
        @assert length(bb.successors) == 2
        true_unhandled_suc = nothing
        false_unhandled_suc = nothing
        true_branch_stmt_vec = Vector{Any}()
        for suc in bb.successors
            @assert isa(suc, Tuple{Bool, BasicBlock})
            if suc[1]
                true_unhandled_suc = recreate_stmts_from_flow_graph(suc[2],
                                                                    bb_stmts_dict,
                                                                    transform_dist_array_read_func,
                                                                    true_branch_stmt_vec,
                                                                    appended_bbs,
                                                                    ssa_defs)
            end
        end
        @assert true_unhandled_suc == nothing
        if !isempty(true_branch_stmt_vec)
            append!(loop_stmt.args[2].args, true_branch_stmt_vec)
            push!(stmt_vec, loop_stmt)
        end

        for suc in bb.successors
            @assert isa(suc, Tuple{Bool, BasicBlock})
            if !suc[1]
                false_unhandled_suc = recreate_stmts_from_flow_graph(suc[2],
                                                                     bb_stmts_dict,
                                                                     transform_dist_array_read_func,
                                                                     stmt_vec,
                                                                     appended_bbs,
                                                                     ssa_defs)

            end
        end
        unhandled_suc = false_unhandled_suc
    elseif !isempty(bb.successors)
        @assert length(bb.successors) == 1
        suc = bb.successors[1]
        @assert suc[1] == nothing
        suc_bb = suc[2]
        if suc_bb.id in appended_bbs
            return nothing
        end

        num_unappended_preds = 0
        for pred in suc_bb.predecessors
            if !(pred.id in appended_bbs)
                num_unappended_preds += 1
            end
        end

        if (num_unappended_preds - suc_bb.num_backward_edges) == 0
            unhandled_suc = recreate_stmts_from_flow_graph(suc_bb,
                                                           bb_stmts_dict,
                                                           transform_dist_array_read_func,
                                                           stmt_vec,
                                                           appended_bbs,
                                                           ssa_defs)
        else
            unhandled_suc = suc_bb
        end
    end
    return unhandled_suc
end

function get_prefetch_stmts(flow_graph::BasicBlock,
                            dist_array_syms::Set{Symbol},
                            ssa_defs::Dict{Symbol, Tuple{Symbol, VarDef}})
    bb_list = flow_graph_to_list(flow_graph)

    syms_deleted = get_deleted_syms(bb_list)

    # The set of symbols to be defined include symbols that are
    # used to define DistArray access subscripts
    bb_stmt_dict = Dict{Int64, Dict{Int64, Any}}()
    syms_to_be_defined = Set{Symbol}()
    for bb in bb_list
        if !(bb.id in keys(bb_dist_array_access_dict))
            continue
        end
        stmt_dict = Dict{Int64, Vector{Any}}()
        stmt_access_dict = bb_dist_array_access_dict[bb.id]
        for (stmt_idx, access_dict) in stmt_access_dict
            stmt_defs = Set{Symbol}()
            stmt_uses = Set{Symbol}()
            for (da_sym, access_vec) in access_dict
                if !(da_sym in dist_array_syms)
                    continue
                end
                for access in access_vec
                    if !access.is_read
                        continue
                    end
                    subscripts_vec = Vector{Any}()
                    sub_stmt_defs = Set{Symbol}()
                    sub_stmt_uses = Set{Symbol}()
                    for sub in access.subscripts
                        push!(subscripts_vec, sub.expr)
                        compute_stmt_ssa_uses_stmt(sub.expr,
                                                   sub_stmt_uses)
                    end
                    if stmt_idx in keys(stmt_dict)
                        append!(stmt_dict[stmt_idx], (da_sym, subscripts_vec))
                    else
                        stmt_dict[stmt_idx] = [(da_sym, subscripts_vec)]
                    end
                    union!(stmt_defs, sub_stmt_defs)
                    union!(stmt_uses, sub_stmt_uses)
                end
                if !isempty(stmt_defs)
                    bb.stmt_ssa_defs[stmt_idx] = stmt_defs
                end
                if !isempty(stmt_uses)
                    bb.stmt_ssa_uses[stmt_idx] = stmt_uses
                end
                union!(syms_to_be_defined, stmt_uses)
            end
        end
        if !isempty(stmt_dict)
            bb_stmt_dict[bb.id] = stmt_dict
        end
    end
    println(syms_to_be_defined)
    stmts_are_added = true
    while stmts_are_added
        stmts_are_added = false
        for bb in bb_list
            branches_in = false
            if bb.control_flow == (:if, :else) ||
                bb.control_flow == :if ||
                bb.control_flow == :for ||
                bb.control_flow == :while
                if bb.control_flow == (:if, :else)
                    for suc in bb.successors
                        @assert isa(suc[1], Bool)
                        if suc[2].id in keys(bb_stmt_dict)
                            branches_in = true
                            break
                        end
                    end
                else
                    for suc in bb.successors
                        @assert isa(suc[1], Bool)
                        if suc[1] && suc[2].id in keys(bb_stmt_dict)
                            branches_in = true
                            break
                        end
                    end
                end
            end
            if branches_in
                condition_stmt_idx = length(bb.stmts)
                if bb.id in keys(bb_stmt_dict) &&
                    condition_stmt_idx in keys(bb_stmt_dict[bb.id])
                    continue
                end
                if bb.id in keys(bb_stmt_dict)
                    bb_stmt_dict[bb.id][condition_stmt_idx] = bb.stmts[condition_stmt_idx]
                else
                    bb_stmt_dict[bb.id] = Dict(condition_stmt_idx => bb.stmts[condition_stmt_idx])
                end
                if condition_stmt_idx in keys(bb.stmt_ssa_uses)
                    union!(syms_to_be_defined, bb.stmt_ssa_uses[condition_stmt_idx])
                end
                stmts_are_added = true
            end
            for stmt_idx in eachindex(bb.stmts)
                stmt = bb.stmts[stmt_idx]
                if bb.id in keys(bb_stmt_dict) &&
                    stmt_idx in keys(bb_stmt_dict[bb.id])
                    continue
                end
                if !(stmt_idx in keys(bb.stmt_ssa_defs)) ||
                    isempty(intersect(bb.stmt_ssa_defs[stmt_idx], syms_to_be_defined))
                    continue
                end
                if bb.id in keys(bb_stmt_dict)
                    bb_stmt_dict[bb.id][stmt_idx] = bb.stmts[stmt_idx]
                else
                    bb_stmt_dict[bb.id] = Dict(stmt_idx => bb.stmts[stmt_idx])
                end
                if stmt_idx in keys(bb.stmt_ssa_uses)
                    union!(syms_to_be_defined, bb.stmt_ssa_uses[stmt_idx])
                end
                stmts_are_added = true
            end
        end
    end

    prefetch_computation_stmts = quote end
    appended_bbs = Set{Int64}()
    stmt_vec = Vector{Any}()
    unhandled_suc = recreate_stmts_from_flow_graph(flow_graph,
                                                   bb_stmt_dict,
                                                   transform_dist_array_read_set,
                                                   stmt_vec,
                                                   appended_bbs,
                                                   ssa_defs)
    @assert unhandled_suc == nothing

    if length(stmt_vec) > 0
        append!(prefetch_computation_stmts.args, stmt_vec)
        return prefetch_computation_stmts
    else
        return nothing
    end
end
