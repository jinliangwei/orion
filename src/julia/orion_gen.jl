module OrionGen

function define(var::Symbol)
    eval(Expr(:global,
              Expr(:(=),
                   var,
                   :nothing)))
    set_func = Expr(:function,
                    Expr(:call,
                         Symbol("set_", string(var)),
                         :val),
                    Expr(:block,
                         Expr(:global,
                              Expr(:(=),
                                   var,
                                   :val))))

    eval(set_func)
    get_func = Expr(:function,
                    Expr(:call,
                         Symbol("get_", string(var))),
                    Expr(:block,
                         Expr(:return,
                              var)))
    eval(get_func)
end

end
