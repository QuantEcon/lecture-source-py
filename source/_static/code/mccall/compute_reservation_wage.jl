"""
Computes the reservation wage of an instance of the McCall model
by finding the smallest w such that V(w) > U.

If V(w) > U for all w, then the reservation wage w_bar is set to
the lowest wage in mcm.w_vec.

If v(w) < U for all w, then w_bar is set to np.inf.

Parameters
----------
mcm : an instance of McCallModel
return_values : bool (optional, default=false)
    Return the value functions as well

Returns
-------
w_bar : scalar
    The reservation wage

"""
function compute_reservation_wage(mcm::McCallModel; return_values::Bool=false)

    V, U = solve_mccall_model(mcm)
    w_idx = searchsortedfirst(V - U, 0)

    if w_idx == length(V)
        w_bar = Inf
    else
        w_bar = mcm.w_vec[w_idx]
    end

    if return_values == false
        return w_bar
    else
        return w_bar, V, U
    end

end
