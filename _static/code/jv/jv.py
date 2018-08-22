import numpy as np
from scipy.integrate import fixed_quad as integrate
from scipy.optimize import minimize
import scipy.stats as stats

ϵ = 1e-4  # A small number, used in the optimization routine

class JvWorker:
    r"""
    A Jovanovic-type model of employment with on-the-job search. The
    value function is given by

    .. math::

        V(x) = \max_{ϕ, s} w(x, ϕ, s)

    for

    .. math::

        w(x, ϕ, s) := x(1 - ϕ - s)
                        + β (1 - π(s)) V(G(x, ϕ))
                        + β π(s) E V[ \max(G(x, ϕ), U)]

    Here

    * x = human capital
    * s = search effort
    * :math:`ϕ` = investment in human capital
    * :math:`π(s)` = probability of new offer given search level s
    * :math:`x(1 - ϕ - s)` = wage
    * :math:`G(x, ϕ)` = new human capital when current job retained
    * U = RV with distribution F -- new draw of human capital

    Parameters
    ----------
    A : scalar(float), optional(default=1.4)
        Parameter in human capital transition function
    α : scalar(float), optional(default=0.6)
        Parameter in human capital transition function
    β : scalar(float), optional(default=0.96)
        Discount factor
    grid_size : scalar(int), optional(default=50)
        Grid size for discretization
    G : function, optional(default=lambda x, ϕ: A * (x * ϕ)**α)
        Transition function for human captial
    π : function, optional(default=sqrt)
        Function mapping search effort (:math:`s \in (0,1)`) to
        probability of getting new job offer
    F : distribution, optional(default=Beta(2,2))
        Distribution from which the value of new job offers is drawn

    Attributes
    ----------
    A, α, β : see Parameters
    x_grid : array_like(float)
        The grid over the human capital

    """

    def __init__(self, A=1.4, α=0.6, β=0.96, grid_size=50,
                 G=None, π=np.sqrt, F=stats.beta(2, 2)):
        self.A, self.α, self.β = A, α, β

        # === set defaults for G, π and F === #
        self.G = G if G is not None else lambda x, ϕ: A * (x * ϕ)**α
        self.π = π
        self.F = F

        # === Set up grid over the state space for DP === #
        # Max of grid is the max of a large quantile value for F and the
        # fixed point y = G(y, 1).
        grid_max = max(A**(1 / (1 - α)), self.F.ppf(1 - ϵ))
        self.x_grid = np.linspace(ϵ, grid_max, grid_size)


    def bellman_operator(self, V, brute_force=False, return_policies=False):
        """
        Returns the approximate value function TV by applying the
        Bellman operator associated with the model to the function V.

        Returns TV, or the V-greedy policies s_policy and ϕ_policy when
        return_policies=True.  In the function, the array V is replaced below
        with a function Vf that implements linear interpolation over the
        points (V(x), x) for x in x_grid.


        Parameters
        ----------
        V : array_like(float)
            Array representing an approximate value function
        brute_force : bool, optional(default=False)
            Default is False. If the brute_force flag is True, then grid
            search is performed at each maximization step.
        return_policies : bool, optional(default=False)
            Indicates whether to return just the updated value function
            TV or both the greedy policy computed from V and TV


        Returns
        -------
        s_policy : array_like(float)
            The greedy policy computed from V.  Only returned if
            return_policies == True
        new_V : array_like(float)
            The updated value function Tv, as an array representing the
            values TV(x) over x in x_grid.

        """
        # === simplify names, set up arrays, etc. === #
        G, π, F, β = self.G, self.π, self.F, self.β
        Vf = lambda x: np.interp(x, self.x_grid, V)
        N = len(self.x_grid)
        new_V, s_policy, ϕ_policy = np.empty(N), np.empty(N), np.empty(N)
        a, b = F.ppf(0.005), F.ppf(0.995)           # Quantiles, for integration
        c1 = lambda z: 1.0 - sum(z)                 # used to enforce s + ϕ <= 1
        c2 = lambda z: z[0] - ϵ                     # used to enforce s >= ϵ
        c3 = lambda z: z[1] - ϵ                     # used to enforce ϕ >= ϵ
        guess = (0.2, 0.2)
        constraints = [{"type": "ineq", "fun": i} for i in [c1, c2, c3]]

        # === solve r.h.s. of Bellman equation === #
        for i, x in enumerate(self.x_grid):

            # === set up objective function === #
            def w(z):
                s, ϕ = z
                h = lambda u: Vf(np.maximum(G(x, ϕ), u)) * F.pdf(u)
                integral, err = integrate(h, a, b)
                q = π(s) * integral + (1.0 - π(s)) * Vf(G(x, ϕ))
                # == minus because we minimize == #
                return - x * (1.0 - ϕ - s) - β * q

            # === either use SciPy solver === #
            if not brute_force:
                max_s, max_ϕ = minimize(w, guess, constraints=constraints,
                                        options={"disp": 0},
                                        method="COBYLA")["x"]
                max_val = -w((max_s, max_ϕ))

            # === or search on a grid === #
            else:
                search_grid = np.linspace(ϵ, 1.0, 15)
                max_val = -1.0
                for s in search_grid:
                    for ϕ in search_grid:
                        current_val = -w((s, ϕ)) if s + ϕ <= 1.0 else -1.0
                        if current_val > max_val:
                            max_val, max_s, max_ϕ = current_val, s, ϕ

            # === store results === #
            new_V[i] = max_val
            s_policy[i], ϕ_policy[i] = max_s, max_ϕ

        if return_policies:
            return s_policy, ϕ_policy
        else:
            return new_V
