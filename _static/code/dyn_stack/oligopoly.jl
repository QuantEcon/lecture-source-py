#=

Author: Victoria Gregory

=#

using QuantEcon
using Roots

"""
This type contains all the parameters and
matrices that describe the oligopoly problem.
"""
struct Oligopoly{TF<:AbstractFloat}
    a0 :: TF
    a1 :: TF
    rho :: TF
    c_eps :: TF
    c :: TF
    d :: TF
    e :: TF
    g :: TF
    h :: TF
    beta :: TF
    A :: Matrix{TF}
    B :: Array{TF}
    Q :: TF
    R :: Matrix{TF}
    Rf :: Matrix{TF}
end

"""
Constructor for building all the elements of the oligopoly
problem.

### Arguments:
* `;a0::AbstractFloat(100)`: Intercept of demand curve
* `;a1::AbstractFloat(1)`: Slope of demand curve
* `;rho::AbstractFloat(0.8)`: Autocorrelation of demand disturbances
* `;c_eps::AbstractFloat(0.2)`: Variance of demand disturbances
* `;c::AbstractFloat(1)`: Cost parameter
* `;d::AbstractFloat(20)`: Cost parameter
* `;e::AbstractFloat(20)`: Cost parameter
* `;g::AbstractFloat(0.2)`: Cost parameter
* `;h::AbstractFloat(0.2)`: Cost parameter
* `;beta::AbstractFloat(0.95)`: Discount rate
"""
function Oligopoly(;a0::AbstractFloat=100.0,
                    a1::AbstractFloat=1.0,
                    rho::AbstractFloat=0.8,
                    c_eps::AbstractFloat=0.2,
                    c::AbstractFloat=1.0,
                    d::AbstractFloat=20.0,
                    e::AbstractFloat=20.0,
                    g::AbstractFloat=0.2,
                    h::AbstractFloat=0.2,
                    beta::AbstractFloat=0.95)

    # Left-hand side of (37)
    Alhs = eye(5)
    Alhs[5, :] = [a0-d 1 -a1 -a1-h c]
    Alhsinv = inv(Alhs)

    # Right-hand side of (37)
    Brhs = [0; 0; 1; 0; 0]
    Arhs = eye(5)
    Arhs[2, 2] = rho
    Arhs[4, 5] = 1
    Arhs[5, 5] = c / beta

    R = [0 0 (a0-e)/2 0 0;
         0 0 1/2 0 0;
         (a0-e)/2 1/2 -a1-0.5*g -a1/2 0;
         0 0 -a1/2 0 0;
         0 0 0 0 0]

    Rf = [0 0 0 0 0 (a0-d)/2;
          0 0 0 0 0 1/2;
          0 0 0 0 0 -a1/2;
          0 0 0 0 0 -a1/2;
          0 0 0 0 0 0;
          (a0-d)/2 1/2 -a1/2 -a1/2 0 -h/2]

    Q = c/2

    A = Alhsinv*Arhs
    B = Alhsinv*Brhs

    return Oligopoly(a0, a1, rho, c_eps, c, d,
                     e, g, h, beta, A, B, Q, R, Rf)

end

"""
Find the value function of the optimal linear regulator problem.
This is steps 2 and 3 in the lecture notes.

### Arguments:
* `olig::Oligopoly`: The oligopoly problem we would like
   to find the value function for

### Returns:
* (P, F, d)::Matrix{TF} where TF<:AbstractFloat : Matrices that describe the value
   function of the optimal linear regulator problem
"""
function find_PFd(olig::Oligopoly)
    lq = LQ(olig.Q, -olig.R, olig.A, olig.B; bet=olig.beta)
    P, F, d = stationary_values(lq)

    Af = vcat(hcat(olig.A-olig.B*F, [0; 0; 0; 0; 0]), [0 0 0 0 0 1])
    Bf = [0; 0; 0; 0; 0; 1]

    lqf = LQ(olig.Q, -olig.Rf, Af, Bf; bet=olig.beta)
    Pf, Ff, df = stationary_values(lqf)

    return P, F, d, Pf, Ff, df
end

"""
Taking the parameters as given, solve for the optimal decision rules
for the firm.

### Arguments:
* `olig::Oligopoly`: The oligopoly problem we would like
   to find the decision rules for
* `(eta0, Q0, q0)::AbstractFloat`: Initial conditions

### Returns:
* (P, -F, D0, Pf, -Ff)::Matrix{Float64}: Matrices that describe the
   decision rule of the optimal linear regulator problem
"""
function solve_for_opt_policy(olig::Oligopoly, eta0::AbstractFloat=0.0,
                              Q0::AbstractFloat=0.0, q0::AbstractFloat=0.0)

    # Step 1/2: Formulate/solve the optimal linear regulator
    P, F, d, Pf, Ff, df = find_PFd(olig)

    # Step 3: Convert implementation into state variables (find coeffs)
    P22 = P[end, end]
    P21 = P[end, 1:end-1]'
    P22inv = P22^(-1)

    # Step 4: Find optimal x_0 and \mu_{x, 0}
    z0 = [1; eta0; Q0; q0]
    x0 = -P22inv*P21*z0
    D0 = -P22inv*P21

    # Return -F and -Ff because we use u_t = -F y_t
    return P, -F, D0, Pf, -Ff

end

olig = Oligopoly()
P, F, D0, Pf, Ff = solve_for_opt_policy(olig)

# Checking time-inconsistency
# Arbitrary initial z_0
y0 = [1; 0; 25; 46]
# optimal x_0 = i_0
i0 = D0*y0
# iterate one period using the closed-loop system
y1 = (olig.A + olig.B*F)*[y0; i0]
# the last element of y_1 is x_1 = i_1
i1_0 = y1[end, 1]

# compare this to the case when the leader solves a Stackelberg problem
# in period 1. if in period 1 the leader could choose i1 given
# (1, v_1, Q_1, \bar{q}_1)
i1_1 = D0*y1[1:end-1, 1]
