#=

Author: Shunsuke Hori

=#

using QuantEcon

"""
This type and method are written to transform a scalar additive functional
into a linear state space system.
"""
type AMF_LSS_VAR{TR<:Real}
    A::TR
    B::TR
    D::TR
    F::TR
    ν::TR
    lss::LSS
end
function AMF_LSS_VAR(A::Real, B::Real, D::Real, F::Real=0.0, ν::Real=0.0)
    # Construct BIG state space representation
    lss = construct_ss(A, B, D, F, ν)
    return AMF_LSS_VAR(A, B, D, F, ν, lss)
end

"""
This creates the state space representation that can be passed
into the LSS type from QuantEcon.
"""
function construct_ss(A::Real, B::Real, D::Real, F::Real, ν::Real)
    H, g = additive_decomp(A, B, D, F)

    # Build A matrix for LSS
    # Order of states is: [1, t, xt, yt, mt]
    A1 = [1 0 0 0 0]       # Transition for 1
    A2 = [1 1 0 0 0]       # Transition for t
    A3 = [0 0 A 0 0]       # Transition for x_{t+1}
    A4 = [ν 0 D 1 0]       # Transition for y_{t+1}
    A5 = [0 0 0 0 1]       # Transition for m_{t+1}
    Abar = vcat(A1, A2, A3, A4, A5)

    # Build B matrix for LSS
    Bbar = [0, 0, B, F, H]

    # Build G matrix for LSS
    # Order of observation is: [xt, yt, mt, st, tt]
    G1 = [0 0 1 0 0]               # Selector for x_{t}
    G2 = [0 0 0 1 0]               # Selector for y_{t}
    G3 = [0 0 0 0 1]               # Selector for martingale
    G4 = [0 0 -g 0 0]              # Selector for stationary
    G5 = [0 ν 0 0 0]               # Selector for trend
    Gbar = vcat(G1, G2, G3, G4, G5)

    # Build LSS type
    x0 = [0, 0, 0, 0, 0]
    S0 = zeros(5, 5)
    lss = LSS(Abar, Bbar, Gbar, mu_0=x0, Sigma_0=S0)

    return lss
end

"""
Return values for the martingale decomposition (Proposition 4.3.3.)
    - `H`         : coefficient for the (linear) martingale component (kappa_a)
    - `g`         : coefficient for the stationary component g(x)
"""
function additive_decomp(A::Real,B::Real,D::Real,F::Real)
    A_res = 1 / (1 - A)
    g = D * A_res
    H = F + D * A_res * B

    return H, g
end

"""
Return values for the multiplicative decomposition (Example 5.4.4.)
    - `ν_tilde`  : eigenvalue
    - `H`         : vector for the Jensen term
"""
function multiplicative_decomp(A::Real, B::Real, D::Real, F::Real, ν::Real)
    H, g = additive_decomp(A, B, D, F)
    ν_tilde = ν + 0.5 * H^2

    return ν_tilde, H, g
end

function loglikelihood_path(amf::AMF_LSS_VAR, x::Vector, y::Vector)
    A, B, D, F = amf.A, amf.B, amf.D, amf.F
    T = length(y)
    FF = F^2
    FFinv = 1/FF
    temp = y[2:end] - y[1:end-1] - D*x[1:end-1]
    obs =  temp .* FFinv .* temp
    obssum = cumsum(obs)
    scalar = (log(FF) + log(2pi))*collect(1:T-1)

    return (-0.5)*(obssum + scalar)
end

function loglikelihood(amf::AMF_LSS_VAR, 
                       x::Vector, y::Vector)
    llh = loglikelihood_path(amf, x, y)
    return llh[end]
end
    

