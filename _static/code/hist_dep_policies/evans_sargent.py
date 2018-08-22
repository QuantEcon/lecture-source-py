"""

@author: dgevans
Edited by: Chase Coleman, John Stachurski

"""

import numpy as np
from quantecon import LQ
from quantecon.matrix_eqn import solve_discrete_lyapunov
from scipy.optimize import root


def computeG(A0, A1, d, Q0, τ0, β, μ):
    """
    Compute government income given μ and return tax revenues and
    policy matrixes for the planner.

    Parameters
    ----------
    A0 : float
        A constant parameter for the inverse demand function
    A1 : float
        A constant parameter for the inverse demand function
    d : float
        A constant parameter for quadratic adjustment cost of production
    Q0 : float
        An initial condition for production
    τ0 : float
        An initial condition for taxes
    β : float
        A constant parameter for discounting
    μ : float
        Lagrange multiplier

    Returns
    -------
    T0 : array(float)
        Present discounted value of government spending
    A : array(float)
        One of the transition matrices for the states
    B : array(float)
        Another transition matrix for the states
    F : array(float)
        Policy rule matrix
    P : array(float)
        Value function matrix
    """
    # Create Matrices for solving Ramsey problem
    R = np.array([[     0,  -A0 / 2,      0,      0],
                 [-A0 / 2,   A1 / 2, -μ / 2,      0],
                 [      0,   -μ / 2,      0,      0],
                 [      0,        0,      0,  d / 2]])

    A = np.array([[     1,      0,   0,              0],
                 [      0,      1,   0,              1],
                 [      0,      0,   0,              0],
                 [-A0 / d, A1 / d,   0, A1 / d + 1 / β]])

    B = np.array([0, 0, 1, 1/d]).reshape(-1, 1)

    Q = 0

    # Use LQ to solve the Ramsey Problem.
    lq = LQ(Q, -R, A, B, beta=β)
    P, F, d = lq.stationary_values()

    # Need y_0 to compute government tax revenue.
    P21 = P[3, :3]
    P22 = P[3, 3]
    z0 = np.array([1, Q0, τ0]).reshape(-1, 1)
    u0 = -P22**(-1) * P21 @ z0
    y0 = np.vstack([z0, u0])

    # Define A_F and S matricies
    AF = A - B @ F
    S = np.array([0, 1, 0, 0]).reshape(-1, 1) @ np.array([[0, 0, 1, 0]])

    # Solves equation (25)
    temp = β * AF.T @ S @ AF
    Ω = solve_discrete_lyapunov(np.sqrt(β) * AF.T, temp)
    T0 = y0.T @ Ω @ y0

    return T0, A, B, F, P


# == Primitives == #
T = 20
A0 = 100.0
A1 = 0.05
d = 0.20
β = 0.95

# == Initial conditions == #
μ0 = 0.0025
Q0 = 1000.0
τ0 = 0.0


def gg(μ):
    """
    Computes the tax revenues for the government given Lagrangian
    multiplier μ.
    """
    return computeG(A0, A1, d, Q0, τ0, β, μ)

# == Solve the Ramsey problem and associated government revenue == #
G0, A, B, F, P = gg(μ0)

# == Compute the optimal u0 == #
P21 = P[3, :3]
P22 = P[3, 3]
z0 = np.array([1, Q0, τ0]).reshape(-1, 1)
u0 = -P22**(-1) * P21 @ z0


# == Initialize vectors == #
y = np.zeros((4, T))
uhat = np.zeros(T)
uhatdif = np.zeros(T)
τhat = np.zeros(T)
τhatdif = np.zeros(T)
μ = np.zeros(T)
G = np.zeros(T)
GPay = np.zeros(T)

# == Initial conditions == #
G[0] = G0
μ[0] = μ0
uhatdif[0] = 0
τhatdif[0] = 0
uhat[0] = u0
y[:, 0] = np.vstack([z0, u0]).flatten()

for t in range(1, T):
    # Iterate government policy
    y[:, t] = (A - B @ F) @ y[:, t-1]

    # update G
    G[t] = (G[t-1] - β * y[1, t] * y[2, t]) / β
    GPay[t] = β * y[1, t] * y[2, t]

    # Compute the μ if the government were able to reset its plan
    # ff is the tax revenues the government would receive if they reset the
    # plan with Lagrange multiplier μ minus current G

    ff = lambda μ: (gg(μ)[0] - G[t]).flatten()

    # find ff = 0
    μ[t] = root(ff, μ[t-1]).x
    temp, Atemp, Btemp, Ftemp, Ptemp = gg(μ[t])

    # Compute alternative decisions
    P21temp = Ptemp[3, :3]
    P22temp = P[3, 3]
    uhat[t] = -P22temp**(-1) * P21temp @ y[:3, t]

    yhat = (Atemp - Btemp @ Ftemp) @ np.hstack([y[0:3, t-1], uhat[t-1]])
    τhat[t] = yhat[3]
    τhatdif[t-1] = τhat[t] - y[3, t]
    uhatdif[t] = uhat[t] - y[3, t]
