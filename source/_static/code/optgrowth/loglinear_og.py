class LogLinearOG:
    """
    Log linear optimal growth model, with log utility, CD production and
    multiplicative lognormal shock, so that

        y = f(k, z) = z k^α

    with z ~ LN(μ, s).

    The class holds parameters and true value and policy functions.
    """

    def __init__(self, α=0.4, β=0.96, μ=0, s=0.1):

        self.α, self.β, self.μ, self.s = α, β, μ, s 

        # == Some useful constants == #
        self.ab = α * β
        self.c1 = np.log(1 - self.ab) / (1 - β)
        self.c2 = (μ + α * np.log(self.ab)) / (1 - α)
        self.c3 = 1 / (1 - β)
        self.c4 = 1 / (1 - self.ab)

    def u(self, c):
        " Utility "
        return np.log(c)

    def u_prime(self, c):
        return 1 / c

    def f(self, k):
        " Deterministic part of production function.  "
        return k**self.α

    def f_prime(self, k):
        return self.α * k**(self.α - 1)

    def c_star(self, y):
        " True optimal policy.  "
        return (1 - self.α * self.β) * y

    def v_star(self, y):
        " True value function. "
        return self.c1 + self.c2 * (self.c3 - self.c4) + self.c4 * np.log(y)

