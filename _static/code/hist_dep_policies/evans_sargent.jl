#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

=#
using QuantEcon
using Roots
using Plots
using LaTeXStrings
pyplot()

struct HistDepRamsey{TF<:AbstractFloat}
    # These are the parameters of the economy
    A0::AbstractFloat
    A1::AbstractFloat
    d::AbstractFloat
    Q0::AbstractFloat
    τ0::AbstractFloat
    μ0::AbstractFloat
    β::AbstractFloat

    # These are the LQ fields and stationary values
    R::Matrix{TF}
    A::Matrix{TF}
    B::Vector{TF}
    Q::TF
    P::Matrix{TF}
    F::Matrix{TF}
    lq::LQ
end


struct RamseyPath{TF<:AbstractFloat}
    y::Matrix{TF}
    uhat::Vector{TF}
    uhatdif::Vector{TF}
    τhat::Vector{TF}
    τhatdif::Vector{TF}
    μ::Vector{TF}
    G::Vector{TF}
    GPay::Vector{TF}
end


function HistDepRamsey(A0::AbstractFloat,
                       A1::AbstractFloat,
                       d::AbstractFloat,
                       Q0::AbstractFloat,
                       τ0::AbstractFloat,
                       μ::AbstractFloat,
                       β::AbstractFloat)
    # Create Matrices for solving Ramsey problem

    R = [     0.0  -A0 / 2      0.0      0.0
         -A0 / 2    A1 / 2   -μ / 2      0.0
             0.0    -μ / 2      0.0      0.0
             0.0       0.0      0.0    d / 2]


    A = [    1.0      0.0    0.0               0.0
             0.0      1.0    0.0               1.0
             0.0      0.0    0.0               0.0
         -A0 / d   A1 / d    0.0  A1 / d + 1.0 / β]

    B = [0.0; 0.0; 1.0; 1.0 / d]

    Q = 0.0

    # Use LQ to solve the Ramsey Problem.
    lq = LQ(Q, -R, A, B, bet=β)

    P, F, _d = stationary_values(lq)

    HistDepRamsey(A0, A1, d, Q0, τ0, μ0, β, R, A, B, Q, P, Array(F), lq)
end


function compute_G(hdr::HistDepRamsey, μ::AbstractFloat)
    # simplify notation
    Q0, τ0, A, B, Q = hdr.Q0, hdr.τ0, hdr.A, hdr.B, hdr.Q
    β = hdr.β

    R = hdr.R
    R[2, 3] = R[3, 2] = -μ / 2
    lq = LQ(Q, -R, A, B, bet=β)

    P, F, _d = stationary_values(lq)

    # Need y_0 to compute government tax revenue.
    u0 = compute_u0(hdr, P)
    y0 = vcat([1.0 Q0 τ0]', u0)

    # Define A_F and S matricies
    AF = A - B * F
    S = [0.0 1.0 0.0 0]' * [0.0 0.0 1.0 0]

    # Solves equation (25)
    Ω = solve_discrete_lyapunov(sqrt(β) .* AF', β * AF' * S * AF)
    T0 = y0' * Ω * y0

    return T0[1], A, B, F, P
end


function compute_u0(hdr::HistDepRamsey, P::Matrix)
    # simplify notation
    Q0, τ0 = hdr.Q0, hdr.τ0

    P21 = P[4, 1:3]
    P22 = P[4, 4]
    z0 = [1.0 Q0 τ0]'
    u0 = reshape(-P22 ^ (-1) .* P21, 1, 3) * (z0)

    return u0[1]
end


function init_path{TF<:AbstractFloat}(hdr::HistDepRamsey{TF}, μ0::TF, T::Integer=20)
    # Construct starting values for the path of the Ramsey economy
    G0, A, B, F, P = compute_G(hdr, μ0)

    # Compute the optimal u0
    u0 = compute_u0(hdr, P)

    # Initialize vectors
    y = Array{TF}(4, T)
    uhat = Vector{TF}(T)
    uhatdif = Vector{TF}(T)
    τhat = Vector{TF}(T)
    τhatdif = Vector{TF}(T)
    μ = Vector{TF}(T)
    G = Vector{TF}(T)
    GPay = Vector{TF}(T)

    # Initial conditions
    G[1] = G0
    μ[1] = μ0
    uhatdif[1] = 0
    uhat[1] = u0
    τhatdif[1] = 0
    y[:, 1] = vcat([1.0 hdr.Q0 hdr.τ0]', u0)

    return RamseyPath(y, uhat, uhatdif, τhat, τhatdif, μ, G, GPay)
end


function compute_ramsey_path!(hdr::HistDepRamsey, rp::RamseyPath)
    # simplify notation
    y, uhat, uhatdif, τhat, = rp.y, rp.uhat, rp.uhatdif, rp.τhat
    τhatdif, μ, G, GPay = rp.τhatdif, rp.μ, rp.G, rp.GPay
    β = hdr.β

    _, A, B, F, P = compute_G(hdr, μ[1])


    for t=2:T
        # iterate government policy
        y[:, t] = (A - B * F) * y[:, t-1]

        # update G
        G[t] = (G[t - 1] - β * y[2, t] * y[3, t]) / β
        GPay[t] = β .* y[2, t] * y[3, t]

        #=
        Compute the μ if the government were able to reset its plan
        ff is the tax revenues the government would receive if they reset the
        plan with Lagrange multiplier μ minus current G
        =#
        ff(μ) = compute_G(hdr, μ)[1] - G[t]

        # find ff = 0
        μ[t] = fzero(ff, μ[t - 1])
        _, Atemp, Btemp, Ftemp, Ptemp = compute_G(hdr, μ[t])

        # Compute alternative decisions
        P21temp = Ptemp[4, 1:3]
        P22temp = P[4, 4]
        uhat[t] = dot(-P22temp ^ (-1) .* P21temp, y[1:3, t])

        yhat = (Atemp-Btemp * Ftemp) * [y[1:3, t-1]; uhat[t-1]]
        τhat[t] = yhat[4]
        τhatdif[t] = τhat[t] - y[4, t]
        uhatdif[t] = uhat[t] - y[4, t]
    end

    return rp
end

# Primitives
T = 20
A0 = 100.0
A1 = 0.05
d = 0.20
β = 0.95

# Initial conditions
μ0 = 0.0025
Q0 = 1000.0
τ0 = 0.0

# Solve Ramsey problem and compute path
hdr = HistDepRamsey(A0, A1, d, Q0, τ0, μ0, β)
rp = init_path(hdr, μ0, T)
compute_ramsey_path!(hdr, rp);                 # updates rp in place
