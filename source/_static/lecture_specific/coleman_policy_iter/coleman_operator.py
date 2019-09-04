import numpy as np
from interpolation import interp
from numba import njit, prange
from quantecon.optimize import brentq


def time_operator_factory(og, parallel_flag=True):
    """
    A function factory for building the Coleman-Reffett operator.
     Here og is an instance of OptimalGrowthModel.
    """
    β = og.β
    f, u = og.f, og.u
    f_prime, u_prime = og.f_prime, og.u_prime
    grid, shocks = og.grid, og.shocks

    @njit
    def objective(c, σ, y):
        """
        The right hand side of the operator
        """
        # First turn w into a function via interpolation
        σ_func = lambda x: interp(grid, σ, x)
        vals = u_prime(σ_func(f(y - c) * shocks)) * f_prime(y - c) * shocks
        return u_prime(c) - β * np.mean(vals)

    @njit(parallel=parallel_flag)
    def K(σ):
        """
        The Coleman-Reffett operator
        """
        σ_new = np.empty_like(σ)
        for i in prange(len(grid)):
            y = grid[i]
            # Solve for optimal c at y
            c_star = brentq(objective, 1e-10, y-1e-10, args=(σ, y))[0]
            σ_new[i] = c_star

        return σ_new

    return K
