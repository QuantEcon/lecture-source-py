import numpy as np
from scipy.stats import lognorm
from scipy.integrate import fixed_quad


class LucasTree:
    """
    Class to store parameters of a the Lucas tree model, a grid for the
    iteration step and some other helpful bits and pieces.

    Parameters
    ----------
    γ : scalar(float)
        The coefficient of risk aversion in the household's CRRA utility
        function
    β : scalar(float)
        The household's discount factor
    α : scalar(float)
        The correlation coefficient in the shock process
    σ : scalar(float)
        The volatility of the shock process
    grid_size : int
        The size of the grid to use

    Attributes
    ----------
    γ, β, α, σ, grid_size : see Parameters
    grid : ndarray
        Properties for grid upon which prices are evaluated
    ϕ : scipy.stats.lognorm
        The distribution for the shock process

    Examples
    --------
    >>> tree = LucasTree(γ=2, β=0.95, α=0.90, σ=0.1)
    >>> price_vals = solve_lucas_model(tree)

    """

    def __init__(self,
                 γ=2,
                 β=0.95,
                 α=0.90,
                 σ=0.1,
                 grid_size=100):

        self.γ, self.β, self.α, self.σ = γ, β, α, σ

        # == Set the grid interval to contain most of the mass of the
        # stationary distribution of the consumption endowment == #
        ssd = self.σ / np.sqrt(1 - self.α**2)
        grid_min, grid_max = np.exp(-4 * ssd), np.exp(4 * ssd)
        self.grid = np.linspace(grid_min, grid_max, grid_size)
        self.grid_size = grid_size

        # == set up distribution for shocks == #
        self.ϕ = lognorm(σ)
        self.draws = self.ϕ.rvs(500)

        # == h(y) = β * int G(y,z)^(1-γ) ϕ(dz) == #
        self.h = np.empty(self.grid_size)
        for i, y in enumerate(self.grid):
            self.h[i] = β * np.mean((y**α * self.draws)**(1 - γ))



## == Now the functions that act on a Lucas Tree == #

def lucas_operator(f, tree, Tf=None):
    """
    The approximate Lucas operator, which computes and returns the
    updated function Tf on the grid points.

    Parameters
    ----------
    f : array_like(float)
        A candidate function on R_+ represented as points on a grid
        and should be flat NumPy array with len(f) = len(grid)

    tree : instance of LucasTree
        Stores the parameters of the problem

    Tf : array_like(float)
        Optional storage array for Tf

    Returns
    -------
    Tf : array_like(float)
        The updated function Tf

    Notes
    -----
    The argument `Tf` is optional, but recommended. If it is passed
    into this function, then we do not have to allocate any memory
    for the array here. As this function is often called many times
    in an iterative algorithm, this can save significant computation
    time.

    """
    grid,  h = tree.grid, tree.h
    α, β = tree.α, tree.β
    z_vec = tree.draws

    # == turn f into a function == #
    Af = lambda x: np.interp(x, grid, f)

    # == set up storage if needed == #
    if Tf is None:
        Tf = np.empty_like(f)

    # == Apply the T operator to f using Monte Carlo integration == #
    for i, y in enumerate(grid):
        Tf[i] = h[i] + β * np.mean(Af(y**α * z_vec))

    return Tf

def solve_lucas_model(tree, tol=1e-6, max_iter=500):
    """
    Compute the equilibrium price function associated with Lucas
    tree

    Parameters
    ----------
    tree : An instance of LucasTree
        Contains parameters
    tol : float
        error tolerance
    max_iter : int
        the maximum number of iterations

    Returns
    -------
    price : array_like(float)
        The prices at the grid points in the attribute `grid` of the object

    """

    # == simplify notation == #
    grid, grid_size = tree.grid, tree.grid_size
    γ = tree.γ

    # == Create storage array for lucas_operator. Reduces  memory
    # allocation and speeds code up == #
    Tf = np.empty(grid_size)

    i = 0
    f = np.empty(grid_size)  # Initial guess of f
    error = tol + 1

    while error > tol and i < max_iter:
        f_new = lucas_operator(f, tree, Tf)
        error = np.max(np.abs(f_new - f))
        f[:] = f_new
        i += 1

    price = f * grid**γ  # Back out price vector

    return price