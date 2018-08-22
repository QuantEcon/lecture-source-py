import numpy as np
from scipy.optimize import fminbound, brentq

class ConsumerProblem:
    """
    A class that stores primitives for the income fluctuation problem.  The
    income process is assumed to be a finite state Markov chain.

    Parameters
    ----------
    r : scalar(float), optional(default=0.01)
        A strictly positive scalar giving the interest rate
    β : scalar(float), optional(default=0.96)
        The discount factor, must satisfy (1 + r) * β < 1
    Π : array_like(float), optional(default=((0.60, 0.40),(0.05, 0.95))
        A 2D NumPy array giving the Markov matrix for {z_t}
    z_vals : array_like(float), optional(default=(0.5, 0.95))
        The state space of {z_t}
    b : scalar(float), optional(default=0)
        The borrowing constraint
    grid_max : scalar(float), optional(default=16)
        Max of the grid used to solve the problem
    grid_size : scalar(int), optional(default=50)
        Number of grid points to solve problem, a grid on [-b, grid_max]
    u : callable, optional(default=np.log)
        The utility function
    du : callable, optional(default=lambda x: 1/x)
        The derivative of u

    Attributes
    ----------
    r, β, Π, z_vals, b, u, du : see Parameters
    asset_grid : np.ndarray
        One dimensional grid for assets

    """

    def __init__(self, 
                 r=0.01, 
                 β=0.96, 
                 Π=((0.6, 0.4), (0.05, 0.95)),
                 z_vals=(0.5, 1.0), 
                 b=0, 
                 grid_max=16, 
                 grid_size=50,
                 u=np.log, 
                 du=lambda x: 1/x):

        self.u, self.du = u, du
        self.r, self.R = r, 1 + r
        self.β, self.b = β, b
        self.Π, self.z_vals = np.array(Π), tuple(z_vals)
        self.asset_grid = np.linspace(-b, grid_max, grid_size)


def bellman_operator(V, cp, return_policy=False):
    """
    The approximate Bellman operator, which computes and returns the
    updated value function TV (or the V-greedy policy c if
    return_policy is True).

    Parameters
    ----------
    V : array_like(float)
        A NumPy array of dim len(cp.asset_grid) times len(cp.z_vals)
    cp : ConsumerProblem
        An instance of ConsumerProblem that stores primitives
    return_policy : bool, optional(default=False)
        Indicates whether to return the greed policy given V or the
        updated value function TV.  Default is TV.

    Returns
    -------
    array_like(float)
        Returns either the greed policy given V or the updated value
        function TV.

    """
    # === Simplify names, set up arrays === #
    R, Π, β, u, b = cp.R, cp.Π, cp.β, cp.u, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    new_V = np.empty(V.shape)
    new_c = np.empty(V.shape)
    z_idx = list(range(len(z_vals)))

    # === Linear interpolation of V along the asset grid === #
    vf = lambda a, i_z: np.interp(a, asset_grid, V[:, i_z])

    # === Solve r.h.s. of Bellman equation === #
    for i_a, a in enumerate(asset_grid):
        for i_z, z in enumerate(z_vals):
            def obj(c):  # objective function to be *minimized*
                y = sum(vf(R * a + z - c, j) * Π[i_z, j] for j in z_idx)
                return - u(c) - β * y
            c_star = fminbound(obj, 1e-8, R * a + z + b)
            new_c[i_a, i_z], new_V[i_a, i_z] = c_star, -obj(c_star)

    if return_policy:
        return new_c
    else:
        return new_V

def coleman_operator(c, cp):
    """
    The approximate Coleman operator.

    Iteration with this operator corresponds to time iteration on the Euler
    equation.  Computes and returns the updated consumption policy
    c.  The array c is replaced with a function cf that implements
    univariate linear interpolation over the asset grid for each
    possible value of z.

    Parameters
    ----------
    c : array_like(float)
        A NumPy array of dim len(cp.asset_grid) times len(cp.z_vals)
    cp : ConsumerProblem
        An instance of ConsumerProblem that stores primitives

    Returns
    -------
    array_like(float)
        The updated policy, where updating is by the Coleman
        operator.

    """
    # === simplify names, set up arrays === #
    R, Π, β, du, b = cp.R, cp.Π, cp.β, cp.du, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    z_size = len(z_vals)
    γ = R * β
    vals = np.empty(z_size)

    # === linear interpolation to get consumption function === #
    def cf(a):
        """
        The call cf(a) returns an array containing the values c(a,
        z) for each z in z_vals.  For each such z, the value c(a, z)
        is constructed by univariate linear approximation over asset
        space, based on the values in the array c
        """
        for i in range(z_size):
            vals[i] = np.interp(a, asset_grid, c[:, i])
        return vals

    # === solve for root to get Kc === #
    Kc = np.empty(c.shape)
    for i_a, a in enumerate(asset_grid):
        for i_z, z in enumerate(z_vals):
            def h(t):
                expectation = np.dot(du(cf(R * a + z - t)), Π[i_z, :])
                return du(t) - max(γ * expectation, du(R * a + z + b))
            Kc[i_a, i_z] = brentq(h, 1e-8, R * a + z + b)

    return Kc

def initialize(cp):
    """
    Creates a suitable initial conditions V and c for value function and time
    iteration respectively.

    Parameters
    ----------
    cp : ConsumerProblem
        An instance of ConsumerProblem that stores primitives

    Returns
    -------
    V : array_like(float)
        Initial condition for value function iteration
    c : array_like(float)
        Initial condition for Coleman operator iteration

    """
    # === Simplify names, set up arrays === #
    R, β, u, b = cp.R, cp.β, cp.u, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    shape = len(asset_grid), len(z_vals)
    V, c = np.empty(shape), np.empty(shape)

    # === Populate V and c === #
    for i_a, a in enumerate(asset_grid):
        for i_z, z in enumerate(z_vals):
            c_max = R * a + z + b
            c[i_a, i_z] = c_max
            V[i_a, i_z] = u(c_max) / (1 - β)

    return V, c
