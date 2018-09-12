import numpy as np
import quantecon as qe
import scipy.linalg as la


class ConsumptionProblem:
    """
    The data for a consumption problem, including some default values.
    """

    def __init__(self,
                 β=.96,
                 y=[2, 1.5],
                 b0=3,
                 P=np.asarray([[.8, .2],
                               [.4, .6]])):
        """

        Parameters
        ----------

        β : discount factor
        P : 2x2 transition matrix
        y : list containing the two income levels
        b0 : debt in period 0 (= state_1 debt level)

        """
        self.β = β
        self.y = y
        self.b0 = b0
        self.P = P


def consumption_complete(cp):
    """
    Computes endogenous values for the complete market case.

    Parameters
    ----------

    cp : instance of ConsumptionProblem

    Returns
    -------

        c_bar : constant consumption
        b1 : rolled over b0
        b2 : debt in state_2

    associated with the price system 

        Q = β * P

    """
    β, P, y, b0 = cp.β, cp.P, cp.y, cp.b0   # Unpack

    y1, y2 = y                              # extract income levels
    b1 = b0                                 # b1 is known to be equal to b0
    Q = β * P                               # assumed price system

    # Using equation (7) calculate b2
    b2 = (y2 - y1 - (Q[0, 0] - Q[1, 0] - 1) * b1) / (Q[0, 1] + 1 - Q[1, 1])

    # Using equation (5) calculate c_bar 
    c_bar = y1 - b0 + Q[0, :] @ np.asarray([b1, b2])

    return c_bar, b1, b2


def consumption_incomplete(cp, N_simul=150):
    """
    Computes endogenous values for the incomplete market case.

    Parameters
    ----------

    cp : instance of ConsumptionProblem
    N_simul : int

    """

    β, P, y, b0 = cp.β, cp.P, cp.y, cp.b0  # Unpack
    # For the simulation define a quantecon MC class
    mc = qe.MarkovChain(P)

    # Useful variables
    y = np.asarray(y).reshape(2, 1)
    v = np.linalg.inv(np.eye(2) - β * P) @ y

    # Simulat state path
    s_path = mc.simulate(N_simul, init=0)

    # Store consumption and debt path
    b_path, c_path = np.ones(N_simul + 1), np.ones(N_simul)
    b_path[0] = b0

    # Optimal decisions from (12) and (13)
    db = ((1 - β) * v - y) / β

    for i, s in enumerate(s_path):
        c_path[i] = (1 - β) * (v - b_path[i] * np.ones((2, 1)))[s, 0]
        b_path[i + 1] = b_path[i] + db[s, 0]

    return c_path, b_path[:-1], y[s_path], s_path

