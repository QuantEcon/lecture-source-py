using Dierckx

"""
Bellman equation for the continuation of the Lucas-Stokey Problem
"""
mutable struct BellmanEquation_Recursive{TP <: Model, TI <: Integer, TR <: Real}
    model::TP
    S::TI
    xbar::Array{TR}
    time_0::Bool
    z0::Array{Array}
    cFB::Vector{TR}
    nFB::Vector{TR}
    xFB::Vector{TR}
    zFB::Vector{Vector{TR}}
end

"""
Compute the planner's allocation by solving Bellman
equation.
"""
struct RecursiveAllocation{TP <: Model,
                           TI <: Integer,
                           TVg <: AbstractVector,
                           TT <: Tuple}
    model::TP
    mc::MarkovChain
    S::TI
    T::BellmanEquation_Recursive
    μgrid::TVg
    xgrid::TVg
    Vf::Array
    policies::TT
end

"""
Initializes the type from the calibration Model
"""
function RecursiveAllocation(model::Model, μgrid::AbstractArray)
    G = model.G
    S = size(model.Π, 1)             # number of states
    mc = MarkovChain(model.Π)
    # now find the first best allocation
    Vf, policies, T, xgrid = solve_time1_bellman(model, μgrid)
    T.time_0 = true                  # Bellman equation now solves time 0 problem
    return RecursiveAllocation(model, mc, S, T, μgrid, xgrid, Vf, policies)
end

"""
Solve the time  1 Bellman equation for calibration Model and initial grid μgrid
"""
function solve_time1_bellman{TR <: Real}(model::Model{TR}, μgrid::AbstractArray)
    Π = model.Π
    S = size(model.Π, 1)

    # First get initial fit from lucas stockey solution.
    # Need to change things to be ex_ante
    PP = SequentialAllocation(model)

    function incomplete_allocation(PP::SequentialAllocation,
                                   μ_::AbstractFloat,
                                   s_::Integer)
        c, n, x, V = time1_value(PP, μ_)
        return c, n, dot(Π[s_, :], x), dot(Π[s_, :], V)
    end

    cf = Array{Function}(S, S)
    nf = Array{Function}(S, S)
    xprimef = Array{Function}(S, S)
    Vf = Vector{Function}(S)
    xgrid = Array{TR}(S, length(μgrid))

    for s_ in 1:S
        c = Array{TR}(length(μgrid), S)
        n = Array{TR}(length(μgrid), S)
        x = Array{TR}(length(μgrid))
        V = Array{TR}(length(μgrid))
        for (i_μ, μ) in enumerate(μgrid)
            c[i_μ, :], n[i_μ, :], x[i_μ], V[i_μ] =
                incomplete_allocation(PP, μ, s_)
        end
        xprimes = repmat(x, 1, S)
        xgrid[s_, :] = x
        for sprime = 1:S
            splc = Spline1D(x[end:-1:1], c[:, sprime][end:-1:1], k=3)
            spln = Spline1D(x[end:-1:1], n[:, sprime][end:-1:1], k=3)
            splx = Spline1D(x[end:-1:1], xprimes[:, sprime][end:-1:1], k=3)
            cf[s_, sprime] = y -> splc(y)
            nf[s_, sprime] = y -> spln(y)
            xprimef[s_, sprime] = y -> splx(y)
            # cf[s_, sprime] = LinInterp(x[end:-1:1], c[:, sprime][end:-1:1])
            # nf[s_, sprime] = LinInterp(x[end:-1:1], n[:, sprime][end:-1:1])
            # xprimef[s_, sprime] = LinInterp(x[end:-1:1], xprimes[:, sprime][end:-1:1])
        end
        splV = Spline1D(x[end:-1:1], V[end:-1:1], k=3)
        Vf[s_] = y -> splV(y)
        # Vf[s_] = LinInterp(x[end:-1:1], V[end:-1:1])
    end

    policies = [cf, nf, xprimef]

    # Create xgrid
    xbar = [maximum(minimum(xgrid)), minimum(maximum(xgrid))]
    xgrid = linspace(xbar[1], xbar[2], length(μgrid))

    # Now iterate on Bellman equation
    T = BellmanEquation_Recursive(model, xgrid, policies)
    diff = 1.0
    while diff > 1e-4
        PF = (i_x, x, s) -> get_policies_time1(T, i_x, x, s, Vf, xbar)
        Vfnew, policies = fit_policy_function(T, PF, xgrid)

        diff = 0.0
        for s=1:S
            diff = max(diff, maximum(abs, (Vf[s].(xgrid) - Vfnew[s].(xgrid)) ./ Vf[s].(xgrid)))
        end

        println("diff = $diff")
        Vf = copy(Vfnew)
    end

    return Vf, policies, T, xgrid
end

"""
Fits the policy functions
"""
function fit_policy_function{TF<:AbstractFloat}(T::BellmanEquation_Recursive,
                                                PF::Function,
						xgrid::AbstractVector{TF})
    S = T.S
    # preallocation
    PFvec = Array{TF}(4S + 1, length(xgrid))
    cf = Array{Function}(S, S)
    nf = Array{Function}(S, S)
    xprimef = Array{Function}(S, S)
    TTf = Array{Function}(S, S)
    Vf = Vector{Function}(S)
    # fit policy fuctions
    for s_ in 1:S
        for (i_x, x) in enumerate(xgrid)
            PFvec[:, i_x] = PF(i_x, x, s_)
        end
        splV = Spline1D(xgrid, PFvec[1,:], k=3)
        Vf[s_] = y -> splV(y)
        # Vf[s_] = LinInterp(xgrid, PFvec[1, :])
        for sprime=1:S
            splc = Spline1D(xgrid, PFvec[1 + sprime, :], k=3)
            spln = Spline1D(xgrid, PFvec[1 + S + sprime, :], k=3)
            splxprime = Spline1D(xgrid, PFvec[1 + 2S + sprime, :], k=3)
            splTT = Spline1D(xgrid, PFvec[1 + 3S + sprime, :], k=3)
            cf[s_, sprime] = y -> splc(y)
            nf[s_, sprime] = y -> spln(y)
            xprimef[s_, sprime] = y -> splxprime(y)
            TTf[s_, sprime] = y -> splTT(y)
        end
    end
    policies = (cf, nf, xprimef, TTf)
    return Vf, policies
end

"""
Computes Tau given c,n
"""
function Tau(pab::RecursiveAllocation,
             c::AbstractArray,
	     n::AbstractArray)
    model = pab.model
    Uc, Un = model.Uc(c, n), model.Un(c, n)
    return 1 + Un ./ (model.Θ .* Uc)
end

Tau(pab::RecursiveAllocation, c::Real, n::Real) = Tau(pab, [c], [n])

"""
Finds the optimal allocation given initial government debt B_ and state s_0
"""
function time0_allocation(pab::RecursiveAllocation, B_::Real, s0::Integer)
    T, Vf = pab.T, pab.Vf
    xbar = T.xbar
    z0 = get_policies_time0(T, B_, s0, Vf, xbar)

    c0, n0, xprime0, T0 = z0[2], z0[3], z0[4], z0[5]
    return c0, n0, xprime0, T0
end

"""
Simulates planners policies for `T` periods
"""
function simulate{TF <: AbstractFloat}(pab::RecursiveAllocation,
                                       B_::TF, s_0::Integer, T::Integer,
                                       sHist::Vector=simulate(pab.mc, T, init=s_0))
    model, mc, Vf, S = pab.model, pab.mc, pab.Vf, pab.S
    Π, Uc = model.Π, model.Uc
    cf, nf, xprimef, TTf = pab.policies

    cHist = Array{TF}(T)
    nHist = Array{TF}(T)
    Bhist = Array{TF}(T)
    xHist = Array{TF}(T)
    TauHist = Array{TF}(T)
    THist = Array{TF}(T)
    μHist = Array{TF}(T)

    #time0
    cHist[1], nHist[1], xHist[1], THist[1]  = time0_allocation(pab, B_, s_0)
    TauHist[1] = Tau(pab, cHist[1], nHist[1])[s_0]
    Bhist[1] = B_
    μHist[1] = Vf[s_0](xHist[1])

    #time 1 onward
    for t in 2:T
        s_, x, s = sHist[t-1], xHist[t-1], sHist[t]
        c = Array{TF}(S)
        n = Array{TF}(S)
        xprime = Array{TF}(S)
        TT = Array{TF}(S)
        for sprime=1:S
            c[sprime], n[sprime], xprime[sprime], TT[sprime] =
                cf[s_, sprime](x), nf[s_, sprime](x),
                xprimef[s_, sprime](x), TTf[s_, sprime](x)
        end

        Tau_val = Tau(pab, c, n)[s]
        u_c = Uc(c, n)
        Eu_c = dot(Π[s_, :], u_c)

        μHist[t] = Vf[s](xprime[s])

        cHist[t], nHist[t], Bhist[t], TauHist[t] = c[s], n[s], x/Eu_c, Tau_val
        xHist[t], THist[t] = xprime[s], TT[s]
    end
    return cHist, nHist, Bhist, xHist, TauHist, THist, μHist, sHist
end

"""
Initializes the class from the calibration Model
"""
function BellmanEquation_Recursive{TF <: AbstractFloat}(model::Model{TF},
                                                        xgrid::AbstractVector{TF},
                                                        policies0::Array)

    S = size(model.Π, 1)                                # number of states
    xbar = [minimum(xgrid), maximum(xgrid)]
    time_0 = false
    z0 = Array{Array}(length(xgrid), S)
    cf, nf, xprimef = policies0[1], policies0[2], policies0[3]
    for s in 1:S
        for (i_x, x) in enumerate(xgrid)
            cs = Array{TF}(S)
            ns = Array{TF}(S)
            xprimes = Array{TF}(S)
            for j = 1:S
                cs[j], ns[j], xprimes[j] = cf[s, j](x), nf[s, j](x), xprimef[s, j](x)
            end
            z0[i_x, s] = vcat(cs, ns, xprimes, zeros(S))
        end
    end
    cFB, nFB, IFB, xFB, zFB = find_first_best(model, S, 2)
    return BellmanEquation_Recursive(model, S, xbar, time_0, z0, cFB, nFB, xFB, zFB)
end


"""
Finds the optimal policies
"""
function get_policies_time1(T::BellmanEquation_Recursive,
                            i_x::Integer, 
                            x::Real, 
                            s_::Integer, 
                            Vf::AbstractArray{Function}, 
                            xbar::AbstractVector)
    model, S = T.model, T.S
    β, Θ, G, Π = model.β, model.Θ, model.G, model.Π
    U,Uc,Un = model.U, model.Uc, model.Un

    S_possible = sum(Π[s_, :].>0)
    sprimei_possible = find(Π[s_, :].>0)

    function objf(z, grad)
        c, xprime = z[1:S_possible], z[S_possible+1:2S_possible]
        n = (c + G[sprimei_possible]) ./ Θ[sprimei_possible]
        Vprime = [Vf[sprimei_possible[si]](xprime[si]) for si in 1:S_possible]
        return -dot(Π[s_, sprimei_possible], U.(c, n) + β * Vprime)
    end

    function cons(out, z, grad)
        c, xprime, TT =
            z[1:S_possible], z[S_possible + 1:2S_possible], z[2S_possible + 1:3S_possible]
        n = (c+G[sprimei_possible]) ./ Θ[sprimei_possible]
        u_c = Uc.(c, n)
        Eu_c = dot(Π[s_, sprimei_possible], u_c)
        out .= x * u_c/Eu_c - u_c .* (c - TT) - Un(c, n) .* n - β * xprime
    end
    function cons_no_trans(out, z, grad)
        c, xprime = z[1:S_possible], z[S_possible + 1:2S_possible]
        n = (c + G[sprimei_possible]) ./ Θ[sprimei_possible]
        u_c = Uc.(c, n)
        Eu_c = dot(Π[s_, sprimei_possible], u_c)
        out .= x * u_c / Eu_c - u_c .* c - Un(c, n) .* n - β * xprime
    end

    if model.transfers == true
        lb = vcat(zeros(S_possible), ones(S_possible)*xbar[1], zeros(S_possible))
        if model.n_less_than_one == true
            ub = vcat(ones(S_possible) - G[sprimei_possible],
                      ones(S_possible) * xbar[2], ones(S_possible))
        else
            ub = vcat(100 * ones(S_possible),
                      ones(S_possible) * xbar[2],
                      100 * ones(S_possible))
        end
        init = vcat(T.z0[i_x, s_][sprimei_possible],
                    T.z0[i_x, s_][2S + sprimei_possible],
                    T.z0[i_x, s_][3S + sprimei_possible])
        opt = Opt(:LN_COBYLA, 3S_possible)
        equality_constraint!(opt, cons, zeros(S_possible))
    else
        lb = vcat(zeros(S_possible), ones(S_possible)*xbar[1])
        if model.n_less_than_one == true
            ub = vcat(ones(S_possible)-G[sprimei_possible], ones(S_possible)*xbar[2])
        else
            ub = vcat(ones(S_possible), ones(S_possible) * xbar[2])
        end
        init = vcat(T.z0[i_x, s_][sprimei_possible],
                    T.z0[i_x, s_][2S+sprimei_possible])
        opt = Opt(:LN_COBYLA, 2S_possible)
        equality_constraint!(opt, cons_no_trans, zeros(S_possible))
    end
    init[init .> ub] = ub[init .> ub]
    init[init .< lb] = lb[init .< lb]

    min_objective!(opt, objf)
    lower_bounds!(opt, lb)
    upper_bounds!(opt, ub)
    maxeval!(opt, 10000000)
    maxtime!(opt, 10)
    ftol_rel!(opt, 1e-8)
    ftol_abs!(opt, 1e-8)

    (minf, minx, ret) = optimize(opt, init)

    if ret != :SUCCESS && ret != :ROUNDOFF_LIMITED && ret != :MAXEVAL_REACHED &&
         ret != :FTOL_REACHED && ret != :MAXTIME_REACHED
        error("optimization failed: ret = $ret")
    end

    T.z0[i_x, s_][sprimei_possible] = minx[1:S_possible]
    T.z0[i_x, s_][S + sprimei_possible] = minx[1:S_possible] + G[sprimei_possible]
    T.z0[i_x, s_][2S + sprimei_possible] = minx[S_possible + 1:2S_possible]
    if model.transfers == true
        T.z0[i_x, s_][3S + sprimei_possible] = minx[2S_possible + 1:3S_possible]
    else
        T.z0[i_x, s_][3S + sprimei_possible] = zeros(S)
    end

    return vcat(-minf, T.z0[i_x, s_])
end

"""
Finds the optimal policies
"""
function get_policies_time0(T::BellmanEquation_Recursive,
                            B_::Real, 
                            s0::Integer, 
                            Vf::AbstractArray{Function}, 
                            xbar::AbstractVector)
    model = T.model
    β, Θ, G = model.β, model.Θ, model.G
    U, Uc, Un = model.U, model.Uc, model.Un

    function objf(z, grad)
        c, xprime = z[1], z[2]
        n = (c + G[s0]) / Θ[s0]
        return -(U(c, n) + β * Vf[s0](xprime))
    end

    function cons(z,grad)
        c, xprime, TT = z[1], z[2], z[3]
        n = (c + G[s0]) / Θ[s0]
        return -Uc(c, n) * (c - B_ - TT) - Un(c, n) * n - β * xprime
    end
    cons_no_trans(z, grad) = cons(vcat(z, 0), grad)

    if model.transfers == true
        lb = [0.0, xbar[1], 0.0]
        if model.n_less_than_one == true
            ub = [1 - G[s0], xbar[2], 100]
        else
            ub = [100.0, xbar[2], 100.0]
        end
        init = vcat(T.zFB[s0][1], T.zFB[s0][3], T.zFB[s0][4])
        init = [0.95124922, -1.15926816,  0.0]
        opt = Opt(:LN_COBYLA, 3)
        equality_constraint!(opt, cons)
    else
        lb = [0.0, xbar[1]]
        if model.n_less_than_one == true
            ub = [1-G[s0], xbar[2]]
        else
            ub = [100, xbar[2]]
        end
        init = vcat(T.zFB[s0][1], T.zFB[s0][3])
        init = [0.95124922, -1.15926816]
        opt = Opt(:LN_COBYLA, 2)
        equality_constraint!(opt, cons_no_trans)
    end
    init[init .> ub] = ub[init .> ub]
    init[init .< lb] = lb[init .< lb]


    min_objective!(opt, objf)
    lower_bounds!(opt, lb)
    upper_bounds!(opt, ub)
    maxeval!(opt, 100000000)
    maxtime!(opt, 30)

    (minf, minx, ret) = optimize(opt, init)

    if ret != :SUCCESS && ret != :ROUNDOFF_LIMITED && ret != :MAXEVAL_REACHED && ret != :FTOL_REACHED
        error("optimization failed: ret = $ret")
    end

    if model.transfers == true
        return -minf, minx[1], minx[1]+G[s0], minx[2], minx[3]
    else
        return -minf, minx[1], minx[1]+G[s0], minx[2], 0
    end
end
