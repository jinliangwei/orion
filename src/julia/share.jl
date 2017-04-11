#const SHARE_FUNC_ARRAY = Array{Expr, 1}()

macro share(ex)
    if ex.head == :function
        eval_expr_on_all(ex, Void)
    else
    end
    esc(ex)
end
