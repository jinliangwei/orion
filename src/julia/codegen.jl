symbol_counter = 0
function gen_unique_symbol()::Symbol
    global symbol_counter += 1
    return Symbol("osym_", string(symbol_counter))
end


function gen_stmt_broadcast_var(var_set::Set{Symbol})::Array{Expr}
    var_set_sym = gen_unique_symbol()
    expr_array = Array{Expr, 1}()
    create_set_expr = :($var_set_sym = Set{Symbol}())
    push!(expr_array, create_set_expr)
    for var in var_set
        add_var_to_set_expr = Expr(:call,
                                   :(push!),
                                   var_set_sym,
                                   QuoteNode(var)
                                   )
        push!(expr_array, add_var_to_set_expr)
    end
    call_broadcast_func_expr = :(broadcast($var_set_sym))
    push!(expr_array, call_broadcast_func_expr)
    dump(expr_array)
    return expr_array
end
