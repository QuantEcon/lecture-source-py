#=

Author: Shunsuke Hori

=#
using QuantEcon
using PyPlot
using Distributions

"""
This type transforms an additive (multipilcative)
functional into a QuantEcon linear state space system.
"""
struct AMF_LSS_VAR{TF<:AbstractFloat, TI<:Integer}
    A::Array{TF, 2}
    B::Array{TF, 2}
    D::Array{TF, 2}
    F::Array{TF, 2}
    ν::Array{TF, 2}
    nx::TI
    nk::TI
    nm::TI
    lss::LSS
end

function AMF_LSS_VAR(A::Array, B::Array,
                D::Union{RowVector, Array},
                F::Union{Void, Array}=nothing;
                ν::Union{Void, Array}=nothing)
    
    if typeof(B) <: Vector
        B = reshape(B, length(B), 1)
    end
    # Unpack required elements
    nx, nk = size(B)

    # checking the dimension of D (extended from the scalar case)
    if ndims(D) > 1
        nm = size(D, 1)
        if typeof(D) <: RowVector
            D = convert(Matrix, D)
        end
    else
        nm = 1
        D = reshape(D, 1, length(D))
    end

    # Set F
    if F==nothing
        F = zeros(nk, 1)
    elseif ndims(F) == 1
        F = reshape(F, length(F), 1)
    end

    # Set ν
    if ν==nothing
        ν = zeros(nm, 1)
    elseif ndims(ν) == 1
        ν = reshape(ν, length(ν), 1)
    else
        throw(ArgumentError("ν must be column vector!"))
    end

    if size(ν, 1) != size(D, 1)
        error("The size of ν is inconsistent with D!")
    end

    # Construct BIG state space representation
    lss = construct_ss(A, B, D, F, ν, nx, nk, nm)

    return AMF_LSS_VAR(A, B, D, F, ν, nx, nk, nm, lss)
end

AMF_LSS_VAR(A::Array, B::Array, D::Union{RowVector, Array}) = 
    AMF_LSS_VAR(A, B, D, nothing, ν=nothing)
AMF_LSS_VAR(A::Array, B::Array, D::Union{RowVector, Array}, F::Real, ν::Real) = 
    AMF_LSS_VAR(A, B, D, [F], ν=[ν])

"""
This creates the state space representation that can be passed
into the quantecon LSS {{ class_word }}.
"""
function construct_ss(A::Array, B::Array, D::Union{RowVector, Array}, F::Array,
                      ν, nx::TI, nk::TI, nm::TI) where TI <: Integer

    H, g = additive_decomp(A, B, D, F, nx)

    # Auxiliary blocks with 0's and 1's to fill out the lss matrices
    nx0c = zeros(nx, 1)
    nx0r = zeros(1, nx)
    nx1 = ones(1, nx)
    nk0 = zeros(1, nk)
    ny0c = zeros(nm, 1)
    ny0r = zeros(1, nm)
    ny1m = eye(nm)
    ny0m = zeros(nm, nm)
    nyx0m = zeros(D)

    # Build A matrix for LSS
    # Order of states is: [1, t, xt, yt, mt]
    A1 = hcat(1, 0, nx0r, ny0r, ny0r)          # Transition for 1
    A2 = hcat(1, 1, nx0r, ny0r, ny0r)          # Transition for t
    A3 = hcat(nx0c, nx0c, A, nyx0m', nyx0m')   # Transition for x_{t+1}
    A4 = hcat(ν, ny0c, D, ny1m, ny0m)          # Transition for y_{t+1}
    A5 = hcat(ny0c, ny0c, nyx0m, ny0m, ny1m)   # Transition for m_{t+1}
    Abar = vcat(A1, A2, A3, A4, A5)

    # Build B matrix for LSS
    Bbar = vcat(nk0, nk0, B, F, H)

    # Build G matrix for LSS
    # Order of observation is: [xt, yt, mt, st, tt]
    G1 = hcat(nx0c, nx0c, eye(nx), nyx0m', nyx0m')    # Selector for x_{t}
    G2 = hcat(ny0c, ny0c, nyx0m, ny1m, ny0m)          # Selector for y_{t}
    G3 = hcat(ny0c, ny0c, nyx0m, ny0m, ny1m)          # Selector for martingale
    G4 = hcat(ny0c, ny0c, -g, ny0m, ny0m)             # Selector for stationary
    G5 = hcat(ny0c, ν, nyx0m, ny0m, ny0m)             # Selector for trend
    Gbar = vcat(G1, G2, G3, G4, G5)

    # Build LSS type
    x0 = hcat(1, 0, nx0r, ny0r, ny0r)
    S0 = zeros(length(x0), length(x0))
    lss = LSS(Abar, Bbar, Gbar, zeros(nx+4nm, 1), x0, S0)

    return lss
end

"""
Return values for the martingale decomposition
    - ν         : unconditional mean difference in Y
    - H         : coefficient for the (linear) martingale component (kappa_a)
    - g         : coefficient for the stationary component g(x)
    - Y_0       : it should be the function of X_0 (for now set it to 0.0)
"""
function additive_decomp(A::Array, B::Array, D::Array, F::Union{Array, Real},
                          nx::Integer)
    I = eye(nx)
    A_res = \(I-A, I)
    g = D * A_res
    H = F + D * A_res * B

    return H, g
end


"""
Return values for the multiplicative decomposition (Example 5.4.4.)
    - ν_tilde  : eigenvalue
    - H         : vector for the Jensen term
"""
function multiplicative_decomp(A::Array, B::Array, D::Array, F::Union{Array, Real},
                                ν::Union{Array, Real}, nx::Integer)
    H, g = additive_decomp(A, B, D, F, nx)
    ν_tilde = ν + 0.5*diag(H*H')

    return H, g, ν_tilde
end

function loglikelihood_path(amf::AMF_LSS_VAR, x::Array, y::Array)
    A, B, D, F = amf.A, amf.B, amf.D, amf.F
    k, T = size(y)
    FF = F*F'
    FFinv = inv(FF)
    temp = y[:, 2:end]-y[:, 1:end-1] - D*x[:, 1:end-1]
    obs =  temp .* FFinv .* temp
    obssum = cumsum(obs)
    scalar = (log(det(FF)) + k*log(2*pi))*collect(1:T)

    return -0.5*(obssum + scalar)
end

function loglikelihood(amf::AMF_LSS_VAR, x::Array, y::Array)
    llh = loglikelihood_path(amf, x, y)

    return llh[end]
end

"""
Plots for the additive decomposition
"""
function plot_additive(amf::AMF_LSS_VAR, T::Integer;
                       npaths::Integer=25, show_trend::Bool=true)

    # Pull out right sizes so we know how to increment
    nx, nk, nm = amf.nx, amf.nk, amf.nm

    # Allocate space (nm is the number of additive functionals - we want npaths for each)
    mpath = Array{Real}(nm*npaths, T)
    mbounds = Array{Real}(nm*2, T)
    spath = Array{Real}(nm*npaths, T)
    sbounds = Array{Real}(nm*2, T)
    tpath = Array{Real}(nm*npaths, T)
    ypath = Array{Real}(nm*npaths, T)

    # Simulate for as long as we wanted
    moment_generator = moment_sequence(amf.lss)
    state = start(moment_generator)
    # Pull out population moments
    for t in 1:T
        tmoms, state = next(moment_generator, state)
        ymeans = tmoms[2]
        yvar = tmoms[4]

        # Lower and upper bounds - for each additive functional
        for ii in 1:nm
            li, ui = (ii-1)*2+1, ii*2
            if sqrt(yvar[nx+nm+ii, nx+nm+ii]) != 0.0
                madd_dist = Normal(ymeans[nx+nm+ii], sqrt(yvar[nx+nm+ii, nx+nm+ii]))
                mbounds[li, t] = quantile(madd_dist, 0.01)
                mbounds[ui, t] = quantile(madd_dist, 0.99)
            elseif sqrt(yvar[nx+nm+ii, nx+nm+ii]) == 0.0
                mbounds[li, t] = ymeans[nx+nm+ii]
                mbounds[ui, t] = ymeans[nx+nm+ii]
            else
                error("standard error is negative")
            end

            if sqrt(yvar[nx+2*nm+ii, nx+2*nm+ii]) != 0.0
                sadd_dist = Normal(ymeans[nx+2*nm+ii], sqrt(yvar[nx+2*nm+ii, nx+2*nm+ii]))
                sbounds[li, t] = quantile(sadd_dist, 0.01)
                sbounds[ui, t] = quantile(sadd_dist, 0.99)
            elseif sqrt(yvar[nx+2*nm+ii, nx+2*nm+ii]) == 0.0
                sbounds[li, t] = ymeans[nx+2*nm+ii]
                sbounds[ui, t] = ymeans[nx+2*nm+ii]
            else
                error("standard error is negative")
            end
                
        end
    end

    # Pull out paths
    for n in 1:npaths
        x, y = simulate(amf.lss,T)
        for ii in 0:nm-1
            ypath[npaths*ii+n, :] = y[nx+ii+1, :]
            mpath[npaths*ii+n, :] = y[nx+nm + ii+1, :]
            spath[npaths*ii+n, :] = y[nx+2*nm + ii+1, :]
            tpath[npaths*ii+n, :] = y[nx+3*nm + ii+1, :]
        end
    end

    add_figs = Array{Any}(nm)
    
    for ii in 0:nm-1
        li, ui = npaths*(ii), npaths*(ii+1)
        LI, UI = 2*(ii), 2*(ii+1)
        add_figs[ii+1] = 
            plot_given_paths(T, ypath[li+1:ui, :], mpath[li+1:ui, :], spath[li+1:ui, :],
                             tpath[li+1:ui, :], mbounds[LI+1:UI, :], sbounds[LI+1:UI, :],
                             show_trend=show_trend)

        add_figs[ii+1][:suptitle]( L"Additive decomposition of $y_{$(ii+1)}$", fontsize=14 )
    end

    return add_figs
end

"""
Plots for the multiplicative decomposition
"""
function plot_multiplicative(amf::AMF_LSS_VAR, T::Integer,
                            npaths::Integer=25, show_trend::Bool=true)
    # Pull out right sizes so we know how to increment
    nx, nk, nm = amf.nx, amf.nk, amf.nm
    # Matrices for the multiplicative decomposition
    H, g, ν_tilde = multiplicative_decomp(A, B, D, F, ν, nx)

    # Allocate space (nm is the number of functionals - we want npaths for each)
    mpath_mult = Array{Real}(nm*npaths, T)
    mbounds_mult = Array{Real}(nm*2, T)
    spath_mult = Array{Real}(nm*npaths, T)
    sbounds_mult = Array{Real}(nm*2, T)
    tpath_mult = Array{Real}(nm*npaths, T)
    ypath_mult = Array{Real}(nm*npaths, T)

    # Simulate for as long as we wanted
    moment_generator = moment_sequence(amf.lss)
    state = start(moment_generator)
    # Pull out population moments
    for t in 1:T
        tmoms, state = next(moment_generator, state)
        ymeans = tmoms[2]
        yvar = tmoms[4]

        # Lower and upper bounds - for each multiplicative functional
        for ii in 1:nm
            li, ui = (ii-1)*2+1, ii*2
            if yvar[nx+nm+ii, nx+nm+ii] != 0.0
                Mdist = LogNormal(ymeans[nx+nm+ii]- t*0.5*diag(H * H')[ii], 
                                sqrt(yvar[nx+nm+ii, nx+nm+ii]))
                mbounds_mult[li, t] = quantile(Mdist, 0.01)
                mbounds_mult[ui, t] = quantile(Mdist, 0.99)
            elseif yvar[nx+nm+ii, nx+nm+ii] == 0.0
                mbounds_mult[li, t] = exp.(ymeans[nx+nm+ii]- t*0.5*diag(H * H')[ii])
                mbounds_mult[ui, t] = exp.(ymeans[nx+nm+ii]- t*0.5*diag(H * H')[ii])
            else
                error("standard error is negative")
            end
            if yvar[nx+2*nm+ii, nx+2*nm+ii] != 0.0
                Sdist = LogNormal(-ymeans[nx+2*nm+ii],
                                sqrt(yvar[nx+2*nm+ii, nx+2*nm+ii]))
                sbounds_mult[li, t] = quantile(Sdist, 0.01)
                sbounds_mult[ui, t] = quantile(Sdist, 0.99)
            elseif yvar[nx+2*nm+ii, nx+2*nm+ii] == 0.0
                sbounds_mult[li, t] = exp.(-ymeans[nx+2*nm+ii])
                sbounds_mult[ui, t] = exp.(-ymeans[nx+2*nm+ii])
            else
                error("standard error is negative")
            end
        end
    end

    # Pull out paths
    for n in 1:npaths
        x, y = simulate(amf.lss,T)
        for ii in 0:nm-1
            ypath_mult[npaths*ii+n, :] = exp.(y[nx+ii+1, :])
            mpath_mult[npaths*ii+n, :] = 
                exp.(y[nx+nm + ii+1, :] - collect(1:T)*0.5*diag(H * H')[ii+1])
            spath_mult[npaths*ii+n, :] = 1./exp.(-y[nx+2*nm + ii+1, :])
            tpath_mult[npaths*ii+n, :] = 
                exp.(y[nx+3*nm + ii+1, :] + collect(1:T)*0.5*diag(H * H')[ii+1])
        end
    end

    mult_figs = Array{Any}(nm)
    
    for ii in 0:nm-1
        li, ui = npaths*(ii), npaths*(ii+1)
        LI, UI = 2*(ii), 2*(ii+1)
        mult_figs[ii+1] = 
            plot_given_paths(T, ypath_mult[li+1:ui, :], mpath_mult[li+1:ui, :],
                             spath_mult[li+1:ui, :], tpath_mult[li+1:ui, :],
                             mbounds_mult[LI+1:UI, :], sbounds_mult[LI+1:UI, :],
                             horline = 1.0, show_trend=show_trend)
        mult_figs[ii+1][:suptitle]( L"Multiplicative decomposition of $y_{$(ii+1)}$", 
                                    fontsize=14)
    end

    return mult_figs
end

function plot_martingales(amf::AMF_LSS_VAR, T::Integer, npaths::Integer=25)

    # Pull out right sizes so we know how to increment
    nx, nk, nm = amf.nx, amf.nk, amf.nm
    # Matrices for the multiplicative decomposition
    H, g, ν_tilde = multiplicative_decomp(amf.A, amf.B, amf.D, amf.F, amf.ν, amf.nx)

    # Allocate space (nm is the number of functionals - we want npaths for each)
    mpath_mult = Array{Real}(nm*npaths, T)
    mbounds_mult = Array{Real}(nm*2, T)

    # Simulate for as long as we wanted
    moment_generator = moment_sequence(amf.lss)
    state = start(moment_generator)
    # Pull out population moments
    for t in 1:T
        tmoms, state = next(moment_generator, state)
        ymeans = tmoms[2]
        yvar = tmoms[4]

        # Lower and upper bounds - for each functional
        for ii in 1:nm
            li, ui = (ii-1)*2+1, ii*2
            if yvar[nx+nm+ii, nx+nm+ii] != 0.0
                Mdist = LogNormal(ymeans[nx+nm+ii]-t*(.5)*diag(H*H')[ii],
                            sqrt(yvar[nx+nm+ii, nx+nm+ii]))
                mbounds_mult[li, t] = quantile(Mdist, 0.01)
                mbounds_mult[ui, t] = quantile(Mdist, 0.99)
            elseif yvar[nx+nm+ii, nx+nm+ii] == 0.0
                mbounds_mult[li, t] = ymeans[nx+nm+ii]-t*(.5)*diag(H*H')[ii]
                mbounds_mult[ui, t] = ymeans[nx+nm+ii]-t*(.5)*diag(H*H')[ii]
            else
                error("standard error is negative")
            end
        end
    end

    # Pull out paths
    for n in 1:npaths
        x, y = simulate(amf.lss, T)
        for ii in 0:nm-1
            mpath_mult[npaths*ii+n, :] = 
                exp.(y[nx+nm + ii+1, :] - (1:T)*0.5*diag(H*H')[ii+1])
        end
    end

    mart_figs = Array{Any}(nm)

    for ii in 0:nm-1
        li, ui = npaths*(ii), npaths*(ii+1)
        LI, UI = 2*(ii), 2*(ii+1)
        mart_figs[ii+1] = plot_martingale_paths(T, mpath_mult[li+1:ui, :],
                                                    mbounds_mult[LI+1:UI, :], horline=1)
        mart_figs[ii+1][:suptitle](L"Martingale components for many paths of $y_{ii+1}$",
                                   fontsize=14)
    end

    return mart_figs
end

function plot_given_paths(T::Integer,
                        ypath::Array, mpath::Array, spath::Array, tpath::Array,
                        mbounds::Array, sbounds::Array; horline::Real=0.0,
                        show_trend::Bool = true)

    # Allocate space
    trange = 1:T

    # Create figure
    fig, ax = subplots(2, 2, sharey=true, figsize=(15, 8))

    # Plot all paths together
    ax[1, 1][:plot](trange, ypath[1, :], label=L"$y_t$", color="k")
    ax[1, 1][:plot](trange, mpath[1, :], label=L"$m_t$", color="m")
    ax[1, 1][:plot](trange, spath[1, :], label=L"$s_t$", color="g")
    if show_trend == true
        ax[1, 1][:plot](trange, tpath[1, :], label=L"t_t", color="r")
    end
    ax[1, 1][:axhline](horline, color="k", linestyle = "-.")
    ax[1, 1][:set_title]("One Path of All Variables")
    ax[1, 1][:legend](loc="top left")

    # Plot Martingale Component
    ax[1, 2][:plot](trange, mpath[1, :], "m")
    ax[1, 2][:plot](trange, mpath', alpha=0.45, color="m")
    ub = mbounds[2, :]
    lb = mbounds[1, :]
    ax[1, 2][:fill_between](trange, lb, ub, alpha=0.25, color="m")
    ax[1, 2][:set_title]("Martingale Components for Many Paths")
    ax[1, 2][:axhline](horline, color="k", linestyle = "-.")

    # Plot Stationary Component
    ax[2, 1][:plot](spath[1, :], color="g")
    ax[2, 1][:plot](spath', alpha=0.25, color="g")
    ub = sbounds[2, :]
    lb = sbounds[1, :]
    ax[2, 1][:fill_between](trange, lb, ub, alpha=0.25, color="g")
    ax[2, 1][:axhline](horline, color="k", linestyle = "-.")
    ax[2, 1][:set_title]("Stationary Components for Many Paths")

    # Plot Trend Component
    if show_trend == true
        ax[2, 2][:plot](tpath', color="r")
    end
    ax[2, 2][:set_title]("Trend Components for Many Paths")
    ax[2, 2][:axhline](horline, color="k", linestyle = "-.")

    return fig
end

function plot_martingale_paths(T::Integer,
                        mpath::Array, mbounds::Array;
                        horline::Real=1,
                        show_trend::Bool = false)
    # Allocate space
    trange = 1:T

    # Create figure
    fig, ax = subplots(1, 1, figsize=(10, 6))

    # Plot Martingale Component
    ub = mbounds[2, :]
    lb = mbounds[1, :]
    ax[:fill_between](trange, lb, ub, color="#ffccff")
    ax[:axhline](horline, color="k", linestyle = "-.")
    ax[:plot](trange, mpath', linewidth=0.25, color="#4c4c4c")

    return fig
end
