#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>

=#
using QuantEcon
using PyPlot
using LaTeXStrings

abstract type AbstractStochProcess end


struct ContStochProcess{TF <: AbstractFloat} <: AbstractStochProcess
    A::Matrix{TF}
    C::Matrix{TF}
end


struct DiscreteStochProcess{TF <: AbstractFloat} <: AbstractStochProcess
    P::Matrix{TF}
    x_vals::Matrix{TF}
end


struct Economy{TF <: AbstractFloat, SP <: AbstractStochProcess}
    β::TF
    Sg::Matrix{TF}
    Sd::Matrix{TF}
    Sb::Matrix{TF}
    Ss::Matrix{TF}
    is_discrete::Bool
    proc::SP
end


struct Path{TF <: AbstractFloat}
    g::Vector{TF}
    d::Vector{TF}
    b::Vector{TF}
    s::Vector{TF}
    c::Vector{TF}
    l::Vector{TF}
    p::Vector{TF}
    τ::Vector{TF}
    rvn::Vector{TF}
    B::Vector{TF}
    R::Vector{TF}
    π::Vector{TF}
    Π::Vector{TF}
    ξ::Vector{TF}
end


function compute_exog_sequences(econ::Economy, x)
    # Compute exogenous variable sequences
    Sg, Sd, Sb, Ss = econ.Sg, econ.Sd, econ.Sb, econ.Ss
    g, d, b, s = [squeeze(S * x, 1) for S in (Sg, Sd, Sb, Ss)]

    #= Solve for Lagrange multiplier in the govt budget constraint
    In fact we solve for ν = λ / (1 + 2*λ).  Here ν is the
    solution to a quadratic equation a(ν^2 - ν) + b = 0 where
    a and b are expected discounted sums of quadratic forms of the state. =#
    Sm = Sb - Sd - Ss

    return g, d, b, s, Sm
end


function compute_allocation(econ::Economy, Sm::Array, ν::AbstractFloat,
                            x::Array, b::Array)
    Sg, Sd, Sb, Ss = econ.Sg, econ.Sd, econ.Sb, econ.Ss

    # Solve for the allocation given ν and x
    Sc = 0.5 .* (Sb + Sd - Sg - ν .* Sm)
    Sl = 0.5 .* (Sb - Sd + Sg - ν .* Sm)
    c = squeeze(Sc * x, 1)
    l = squeeze(Sl * x, 1)
    p = squeeze((Sb - Sc) * x, 1)  # Price without normalization
    τ = 1 .- l ./ (b .- c)
    rvn = l .* τ

    return Sc, Sl, c, l, p, τ, rvn
end


function compute_ν(a0::AbstractFloat, b0::AbstractFloat)
    disc = a0^2 - 4a0 * b0

    if disc >= 0
        ν = 0.5 *(a0 - sqrt(disc)) / a0
    else
        println("There is no Ramsey equilibrium for these parameters.")
        error("Government spending (economy.g) too low")
    end

    # Test that the Lagrange multiplier has the right sign
    if ν * (0.5 - ν) < 0
        print("Negative multiplier on the government budget constraint.")
        error("Government spending (economy.g) too low")
    end

    return ν
end


function compute_Π(B::Vector, R::Vector, rvn::Vector, g::Vector, ξ::Vector)
    π = B[2:end] - R[1:end-1] .* B[1:end-1] - rvn[1:end-1] + g[1:end-1]
    Π = cumsum(π .* ξ)
    return π, Π
end


function compute_paths{TF <: AbstractFloat}(econ::Economy{TF, DiscreteStochProcess{TF}},
                                            T::Integer)
    # simplify notation
    β, Sg, Sd, Sb, Ss = econ.β, econ.Sg, econ.Sd, econ.Sb, econ.Ss
    P, x_vals = econ.proc.P, econ.proc.x_vals

    mc = MarkovChain(P)
    state = simulate(mc, T, init=1)
    x = x_vals[:, state]

    # Compute exogenous sequence
    g, d, b, s, Sm = compute_exog_sequences(econ, x)

    # compute a0, b0
    ns = size(P, 1)
    F = eye(ns) - β.*P
    a0 = (F \ ((Sm * x_vals)'.^2))[1] ./ 2
    H = ((Sb - Sd + Sg) * x_vals) .* ((Sg - Ss)*x_vals)
    b0 = (F \ H')[1] ./ 2

    # compute lagrange multiplier
    ν = compute_ν(a0, b0)

    # Solve for the allocation given ν and x
    Sc, Sl, c, l, p, τ, rvn = compute_allocation(econ, Sm, ν, x, b)

    # compute remaining variables
    H = ((Sb - Sc) * x_vals) .* ((Sl - Sg) * x_vals) - (Sl * x_vals).^2
    temp = squeeze(F * H', 2)
    B = temp[state] ./ p
    H = squeeze(P[state, :] * ((Sb - Sc) * x_vals)', 2)
    R = p ./ (β .* H)
    temp = squeeze(P[state, :] *((Sb - Sc) * x_vals)', 2)
    ξ = p[2:end] ./ temp[1:end-1]

    # compute π
    π, Π = compute_Π(B, R, rvn, g, ξ)

    Path(g, d, b, s, c, l, p, τ, rvn, B, R, π, Π, ξ)
end


function compute_paths{TF<:AbstractFloat}(econ::Economy{TF, ContStochProcess{TF}}, 
                                          T::Integer)
    # Simplify notation
    β, Sg, Sd, Sb, Ss = econ.β, econ.Sg, econ.Sd, econ.Sb, econ.Ss
    A, C = econ.proc.A, econ.proc.C

    # Generate an initial condition x0 satisfying x0 = A x0
    nx, nx = size(A)
    x0 = nullspace((eye(nx) - A))
    x0 = x0[end] < 0 ? -x0 : x0
    x0 = x0 ./ x0[end]
    x0 = squeeze(x0, 2)

    # Generate a time series x of length T starting from x0
    nx, nw = size(C)
    x = Matrix{TF}(nx, T)
    w = randn(nw, T)
    x[:, 1] = x0
    for t=2:T
        x[:, t] = A *x[:, t-1] + C * w[:, t]
    end

    # Compute exogenous sequence
    g, d, b, s, Sm = compute_exog_sequences(econ, x)

    # compute a0 and b0
    H = Sm'Sm
    a0 = 0.5 * var_quadratic_sum(A, C, H, β, x0)
    H = (Sb - Sd + Sg)'*(Sg + Ss)
    b0 = 0.5 * var_quadratic_sum(A, C, H, β, x0)

    # compute lagrange multiplier
    ν = compute_ν(a0, b0)

    # Solve for the allocation given ν and x
    Sc, Sl, c, l, p, τ, rvn = compute_allocation(econ, Sm, ν, x, b)

    # compute remaining variables
    H = Sl'Sl - (Sb - Sc)' *(Sl - Sg)
    L = Vector{TF}(T)
    for t=1:T
        L[t] = var_quadratic_sum(A, C, H, β, x[:, t])
    end
    B = L ./ p
    Rinv = squeeze(β .* (Sb- Sc)*A*x, 1) ./ p
    R = 1 ./ Rinv
    AF1 = (Sb - Sc) * x[:, 2:end]
    AF2 = (Sb - Sc) * A * x[:, 1:end-1]
    ξ =  AF1 ./ AF2
    ξ = squeeze(ξ, 1)

    # compute π
    π, Π = compute_Π(B, R, rvn, g, ξ)

    Path(g, d, b, s, c, l, p, τ, rvn, B, R, π, Π, ξ)
end

function gen_fig_1(path::Path)
    T = length(path.c)

    figure(figsize=(12,8))

    ax1=subplot(2, 2, 1)
    ax1[:plot](path.rvn)
    ax1[:plot](path.g)
    ax1[:plot](path.c)
    ax1[:set_xlabel]("Time")
    ax1[:legend]([L"$\tau_t \ell_t$",L"$g_t$",L"$c_t$"])

    ax2=subplot(2, 2, 2)
    ax2[:plot](path.rvn)
    ax2[:plot](path.g)
    ax2[:plot](path.B[2:end])
    ax2[:set_xlabel]("Time")
    ax2[:legend]([L"$\tau_t \ell_t$",L"$g_t$",L"$B_{t+1}$"])

    ax3=subplot(2, 2, 3)
    ax3[:plot](path.R-1)
    ax3[:set_xlabel]("Time")
    ax3[:legend]([L"$R_{t - 1}$"])

    ax4=subplot(2, 2, 4)
    ax4[:plot](path.rvn)
    ax4[:plot](path.g)
    ax4[:plot](path.π)
    ax4[:set_xlabel]("Time")
    ax4[:legend]([L"$\tau_t \ell_t$",L"$g_t$",L"$\pi_{t+1}$"])
end

function gen_fig_2(path::Path)

    T = length(path.c)

    fig, axes = plt[:subplots](2, 1, figsize=(8, 7))
    
    plots = [path.ξ, path.Π]
    labels = [L"$\xi_t$", L"$\Pi_t$"]
    
    for (ax, plot, label) in zip(axes, plots, labels)
        ax[:plot](2:T, plot, label=label)
        ax[:set_xlabel]("Time")
        ax[:legend]
    end
end
