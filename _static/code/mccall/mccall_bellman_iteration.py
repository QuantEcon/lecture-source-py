import numpy as np
from quantecon.distributions import BetaBinomial
from numba import jit

# A default utility function

@jit
def u(c, σ):
    if c > 0:
        return (c**(1 - σ) - 1) / (1 - σ)
    else:
        return -10e6


class McCallModel:
    """
    Stores the parameters and functions associated with a given model.
    """

    def __init__(self, 
                 α=0.2,        # Job separation rate
                 β=0.98,       # Discount rate
                 γ=0.7,        # Job offer rate
                 c=6.0,        # Unemployment compensation
                 σ=2.0,        # Utility parameter
                 w_vals=None,  # Possible wage values
                 ϕ_vals=None): # Probabilities over w_vals

        self.α, self.β, self.γ, self.c = α, β, γ, c
        self.σ = σ

        # Add a default wage vector and probabilities over the vector using
        # the beta-binomial distribution
        if w_vals is None:
            n = 60  # number of possible outcomes for wage
            self.w_vals = np.linspace(10, 20, n)     # wages between 10 and 20
            a, b = 600, 400  # shape parameters
            dist = BetaBinomial(n-1, a, b)
            self.ϕ_vals = dist.pdf()  
        else:
            self.w_vals = w_vals
            self.ϕ_vals = ϕ_vals

@jit
def _update_bellman(α, β, γ, c, σ, w_vals, ϕ_vals, e, e_new, h):
    """
    A jitted function to update the Bellman equations.  Note that e_new is
    modified in place (i.e, modified by this function).  The new value of h is
    returned.

    """
    for w_idx, w in enumerate(w_vals):
        # w_idx indexes the vector of possible wages
        e_new[w_idx] = u(w, σ) + β * ((1 - α) * e[w_idx] + α * h)

    h_new = u(c, σ) + β * (1 - γ) * h + \
                    β * γ * np.sum(np.maximum(h, e) * ϕ_vals)

    return h_new


def solve_mccall_model(mcm, tol=1e-5, max_iter=2000):
    """
    Iterates to convergence on the Bellman equations
    
    mcm is an instance of McCallModel
    """

    e = np.ones_like(mcm.w_vals) # Initial guess of e
    e_new = np.empty_like(e)     # To store updates to e
    h = 1                        # Initial guess of h
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        h_new = _update_bellman(mcm.α, mcm.β, mcm.γ, 
                mcm.c, mcm.σ, mcm.w_vals, mcm.ϕ_vals, e, e_new, h)
        error_1 = np.max(np.abs(e_new - e))
        error_2 = np.abs(h_new - h)
        error = max(error_1, error_2)
        e[:] = e_new
        h = h_new
        i += 1

    return e, h

 