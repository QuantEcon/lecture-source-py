""" 

@authors: Chase Coleman, Balint Skoze, Tom Sargent

"""

class AMF_LSS_VAR:
    """
    This class is written to transform a scalar additive functional
    into a linear state space system.
    """
    def __init__(self, A, B, D, F=0.0, ν=0.0):
        # Unpack required elements
        self.A, self.B, self.D, self.F, self.ν = A, B, D, F, ν
 
        # Create space for additive decomposition
        self.add_decomp = None
        self.mult_decomp = None
 
        # Construct BIG state space representation
        self.lss = self.construct_ss()
 
    def construct_ss(self):
        """
        This creates the state space representation that can be passed
        into the quantecon LSS class.
        """
        # Pull out useful info
        A, B, D, F, ν = self.A, self.B, self.D, self.F, self.ν
        nx, nk, nm = 1, 1, 1
        if self.add_decomp:
            ν, H, g = self.add_decomp
        else:
            ν, H, g = self.additive_decomp()
 
        # Build A matrix for LSS
        # Order of states is: [1, t, xt, yt, mt]
        A1 = np.hstack([1, 0, 0, 0, 0])       # Transition for 1
        A2 = np.hstack([1, 1, 0, 0, 0])       # Transition for t
        A3 = np.hstack([0, 0, A, 0, 0])       # Transition for x_{t+1}
        A4 = np.hstack([ν, 0, D, 1, 0])       # Transition for y_{t+1}
        A5 = np.hstack([0, 0, 0, 0, 1])       # Transition for m_{t+1}
        Abar = np.vstack([A1, A2, A3, A4, A5])
 
        # Build B matrix for LSS
        Bbar = np.vstack([0, 0, B, F, H])
 
        # Build G matrix for LSS
        # Order of observation is: [xt, yt, mt, st, tt]
        G1 = np.hstack([0, 0, 1, 0, 0])               # Selector for x_{t}
        G2 = np.hstack([0, 0, 0, 1, 0])               # Selector for y_{t}
        G3 = np.hstack([0, 0, 0, 0, 1])               # Selector for martingale
        G4 = np.hstack([0, 0, -g, 0, 0])              # Selector for stationary
        G5 = np.hstack([0, ν, 0, 0, 0])               # Selector for trend
        Gbar = np.vstack([G1, G2, G3, G4, G5])
 
        # Build H matrix for LSS
        Hbar = np.zeros((1, 1))
 
        # Build LSS type
        x0 = np.hstack([1, 0, 0, 0, 0])
        S0 = np.zeros((5, 5))
        lss = qe.lss.LinearStateSpace(Abar, Bbar, Gbar, Hbar, mu_0=x0, Sigma_0=S0)
 
        return lss
 
    def additive_decomp(self):
        """
        Return values for the martingale decomposition (Proposition 4.3.3.)
            - ν         : unconditional mean difference in Y
            - H         : coefficient for the (linear) martingale component (kappa_a)
            - g         : coefficient for the stationary component g(x)
            - Y_0       : it should be the function of X_0 (for now set it to 0.0)
        """
        A_res = 1 / (1 - self.A)
        g = self.D * A_res
        H = self.F + self.D * A_res * self.B
 
        return self.ν, H, g
 
    def multiplicative_decomp(self):
        """
        Return values for the multiplicative decomposition (Example 5.4.4.)
            - ν_tilde  : eigenvalue
            - H        : vector for the Jensen term
        """
        ν, H, g = self.additive_decomp()
        ν_tilde = ν + (.5) * H**2
 
        return ν_tilde, H, g
 
    def loglikelihood_path(self, x, y):
        A, B, D, F = self.A, self.B, self.D, self.F
        T = y.T.size
        FF = F**2
        FFinv = 1 / FF
        temp = y[1:] - y[:-1] - D * x[:-1]
        obs = temp * FFinv * temp
        obssum = np.cumsum(obs)
        scalar = (np.log(FF) + np.log(2 * np.pi)) * np.arange(1, T)
 
        return (-0.5) * (obssum + scalar)
 
    def loglikelihood(self, x, y):
        llh = self.loglikelihood_path(x, y)
 
        return llh[-1]
