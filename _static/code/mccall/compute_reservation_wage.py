def compute_reservation_wage(mcm, return_values=False):
    """
    Computes the reservation wage of an instance of McCallModel
    by finding the smallest w such that e(w) > h.
        
    """

    e, h = solve_mccall_model(mcm)
    w_idx = np.searchsorted(e - h, 0)  

    if w_idx == len(e):
        # If v(w) < U for all w, set w_bar to np.inf.
        w_bar = np.inf
    else:
        # Set to lowest wage
        w_bar = mcm.w_vals[w_idx]

    if return_values == False:
        return w_bar
    else:
        return w_bar, e, h

