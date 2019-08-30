import numpy as np

class LogUtility:

    def __init__(self,
                 β=0.9,
                 ψ=0.69,
                 π=0.5*np.ones((2, 2)),
                 G=np.array([0.1, 0.2]),
                 Θ=np.ones(2),
                 transfers=False):

        self.β, self.ψ, self.π = β, ψ, π
        self.G, self.Θ, self.transfers = G, Θ, transfers

    # Utility function
    def U(self, c, n):
        return np.log(c) + self.ψ * np.log(1 - n)

    # Derivatives of utility function
    def Uc(self, c, n):
        return 1 / c

    def Ucc(self, c, n):
        return -c**(-2)

    def Un(self, c, n):
        return -self.ψ / (1 - n)

    def Unn(self, c, n):
        return -self.ψ / (1 - n)**2