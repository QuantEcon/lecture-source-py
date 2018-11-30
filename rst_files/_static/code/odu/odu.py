from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import fixed_quad
from numpy import maximum as npmax


class SearchProblem:
    """
    A class to store a given parameterization of the "offer distribution
    unknown" model.

    Parameters
    ----------
    β : scalar(float), optional(default=0.95)
        The discount parameter
    c : scalar(float), optional(default=0.6)
        The unemployment compensation
    F_a : scalar(float), optional(default=1)
        First parameter of β distribution on F
    F_b : scalar(float), optional(default=1)
        Second parameter of β distribution on F
    G_a : scalar(float), optional(default=3)
        First parameter of β distribution on G
    G_b : scalar(float), optional(default=1.2)
        Second parameter of β distribution on G
    w_max : scalar(float), optional(default=2)
        Maximum wage possible
    w_grid_size : scalar(int), optional(default=40)
        Size of the grid on wages
    π_grid_size : scalar(int), optional(default=40)
        Size of the grid on probabilities

    Attributes
    ----------
    β, c, w_max : see Parameters
    w_grid : np.ndarray
        Grid points over wages, ndim=1
    π_grid : np.ndarray
        Grid points over π, ndim=1
    grid_points : np.ndarray
        Combined grid points, ndim=2
    F : scipy.stats._distn_infrastructure.rv_frozen
        Beta distribution with params (F_a, F_b), scaled by w_max
    G : scipy.stats._distn_infrastructure.rv_frozen
        Beta distribution with params (G_a, G_b), scaled by w_max
    f : function
        Density of F
    g : function
        Density of G
    π_min : scalar(float)
        Minimum of grid over π
    π_max : scalar(float)
        Maximum of grid over π
    """

    def __init__(self, β=0.95, c=0.6, F_a=1, F_b=1, G_a=3, G_b=1.2,
                 w_max=2, w_grid_size=40, π_grid_size=40):

        self.β, self.c, self.w_max = β, c, w_max
        self.F = beta(F_a, F_b, scale=w_max)
        self.G = beta(G_a, G_b, scale=w_max)
        self.f, self.g = self.F.pdf, self.G.pdf    # Density functions
        self.π_min, self.π_max = 1e-3, 1 - 1e-3    # Avoids instability
        self.w_grid = np.linspace(0, w_max, w_grid_size)
        self.π_grid = np.linspace(self.π_min, self.π_max, π_grid_size)
        x, y = np.meshgrid(self.w_grid, self.π_grid)
        self.grid_points = np.column_stack((x.ravel(order='F'), y.ravel(order='F')))


    def q(self, w, π):
        """
        Updates π using Bayes' rule and the current wage observation w.

        Returns
        -------

        new_π : scalar(float)
            The updated probability

        """

        new_π = 1.0 / (1 + ((1 - π) * self.g(w)) / (π * self.f(w)))

        # Return new_π when in [π_min, π_max] and else end points
        new_π = np.maximum(np.minimum(new_π, self.π_max), self.π_min)

        return new_π

    def bellman_operator(self, v):
        """

        The Bellman operator.  Including for comparison. Value function
        iteration is not recommended for this problem.  See the
        reservation wage operator below.

        Parameters
        ----------
        v : array_like(float, ndim=1, length=len(π_grid))
            An approximate value function represented as a
            one-dimensional array.

        Returns
        -------
        new_v : array_like(float, ndim=1, length=len(π_grid))
            The updated value function

        """
        # == Simplify names == #
        f, g, β, c, q = self.f, self.g, self.β, self.c, self.q

        vf = LinearNDInterpolator(self.grid_points, v)
        N = len(v)
        new_v = np.empty(N)

        for i in range(N):
            w, π = self.grid_points[i, :]
            v1 = w / (1 - β)
            integrand = lambda m: vf(m, q(m, π)) * (π * f(m) + 
                        (1 - π) * g(m))
            integral, error = fixed_quad(integrand, 0, self.w_max)
            v2 = c + β * integral
            new_v[i] = max(v1, v2)

        return new_v

    def get_greedy(self, v):
        """
        Compute optimal actions taking v as the value function.

        Parameters
        ----------
        v : array_like(float, ndim=1, length=len(π_grid))
            An approximate value function represented as a
            one-dimensional array.

        Returns
        -------
        policy : array_like(float, ndim=1, length=len(π_grid))
            The decision to accept or reject an offer where 1 indicates
            accept and 0 indicates reject

        """
        # == Simplify names == #
        f, g, β, c, q = self.f, self.g, self.β, self.c, self.q

        vf = LinearNDInterpolator(self.grid_points, v)
        N = len(v)
        policy = np.zeros(N, dtype=int)

        for i in range(N):
            w, π = self.grid_points[i, :]
            v1 = w / (1 - β)
            integrand = lambda m: vf(m, q(m, π)) * (π * f(m) +
                                                     (1 - π) * g(m))
            integral, error = fixed_quad(integrand, 0, self.w_max)
            v2 = c + β * integral
            policy[i] = v1 > v2  # Evaluates to 1 or 0

        return policy

    def res_wage_operator(self, ϕ):
        """

        Updates the reservation wage function guess ϕ via the operator
        Q.

        Parameters
        ----------
        ϕ : array_like(float, ndim=1, length=len(π_grid))
            This is reservation wage guess

        Returns
        -------
        new_ϕ : array_like(float, ndim=1, length=len(π_grid))
            The updated reservation wage guess.

        """
        # == Simplify names == #
        β, c, f, g, q = self.β, self.c, self.f, self.g, self.q
        # == Turn ϕ into a function == #
        ϕ_f = lambda p: np.interp(p, self.π_grid, ϕ)

        new_ϕ = np.empty(len(ϕ))
        for i, π in enumerate(self.π_grid):
            def integrand(x):
                "Integral expression on right-hand side of operator"
                return npmax(x, ϕ_f(q(x, π))) * (π * f(x) + (1 - π) * g(x))
            integral, error = fixed_quad(integrand, 0, self.w_max)
            new_ϕ[i] = (1 - β) * c + β * integral

        return new_ϕ
