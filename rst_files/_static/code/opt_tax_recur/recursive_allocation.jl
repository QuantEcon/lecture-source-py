
"""
Compute the planner's allocation by solving Bellman
equation.
"""
struct RecursiveAllocation{TP <: Model, TI <: Integer,
                           TVg <: AbstractVector, TVv <: AbstractVector,
                           TVp <: AbstractArray}
    model::TP
    mc::MarkovChain
    S::TI
    T::BellmanEquation
    μgrid::TVg
    xgrid::TVg
    Vf::TVv
    policies::TVp
end

"""
Initializes the {{ class_word }} from the calibration `Model`
"""
function RecursiveAllocation(model::Model, μgrid::AbstractArray)
    mc = MarkovChain(model.Π)
    G = model.G
    S = size(model.Π, 1) # Number of states
    # Now find the first best allocation
    Vf, policies, T, xgrid = solve_time1_bellman(model, μgrid)
    T.time_0 = true      # Bellman equation now solves time 0 problem
    return RecursiveAllocation(model, mc, S, T, μgrid, xgrid, Vf, policies)
end

"""
Solve the time 1 Bellman equation for calibration `Model` and initial grid `μgrid0`
"""
function solve_time1_bellman{TF <: AbstractFloat}(model::Model{TF}, μgrid::AbstractArray)
    μgrid0 = μgrid
    S = size(model.Π, 1)
    # First get initial fit
    PP = SequentialAllocation(model)
    c = Matrix{TF}(length(μgrid), 2)
    n = Matrix{TF}(length(μgrid), 2)
    x = Matrix{TF}(length(μgrid), 2)
    V = Matrix{TF}(length(μgrid), 2)
    for (i, μ) in enumerate(μgrid0)
        c[i, :], n[i, :], x[i, :], V[i, :] = time1_value(PP, μ)
    end
    Vf = Vector{LinInterp}(2)
    cf = Vector{LinInterp}(2)
    nf = Vector{LinInterp}(2)
    xprimef = Array{LinInterp}(2, S)
    for s in 1:2
        cf[s] = LinInterp(x[:, s][end:-1:1], c[:, s][end:-1:1])
        nf[s] = LinInterp(x[:, s][end:-1:1], n[:, s][end:-1:1])
        Vf[s] = LinInterp(x[:, s][end:-1:1], V[:, s][end:-1:1])
        for sprime in 1:S
            xprimef[s, sprime] = LinInterp(x[:, s][end:-1:1], x[:, s][end:-1:1])
        end
    end
    policies = [cf, nf, xprimef]
    # Create xgrid
    xbar = [maximum(minimum(x, 1)), minimum(maximum(x, 1))]
    xgrid = linspace(xbar[1], xbar[2], length(μgrid0))
    # Now iterate on bellman equation
    T = BellmanEquation(model, xgrid, policies)
    diff = 1.0
    while diff > 1e-6
        if T.time_0 == false
            Vfnew, policies =
                fit_policy_function(PP,
                (i_x, x, s) -> get_policies_time1(T, i_x, x, s, Vf), xgrid)
        elseif T.time_0 == true
            Vfnew, policies =
                fit_policy_function(PP,
                (i_x, B_, s0) -> get_policies_time0(T, i_x, B_, s0, Vf), xgrid)
        else
            error("T.time_0 is $(T.time_0), which is invalid")
        end
        diff = 0.0
        for s in 1:S
            diff = max(diff, maximum(abs, (Vf[s].(xgrid)-Vfnew[s].(xgrid))./Vf[s].(xgrid)))
        end
        print("diff = $diff \n")
        Vf = Vfnew
    end
    # Store value function policies and Bellman Equations
    return Vf, policies, T, xgrid
end

"""
Fits the policy functions PF using the points `xgrid` using interpolation
"""
function fit_policy_function(PP::SequentialAllocation,
                            PF::Function, xgrid::AbstractArray)
    S = PP.S
    Vf = Vector{LinInterp}(S)
    cf = Vector{LinInterp}(S)
    nf = Vector{LinInterp}(S)
    xprimef = Array{LinInterp}(S, S)
    for s in 1:S
        PFvec = Array{typeof(PP.model).parameters[1]}(length(xgrid), 3+S)
        for (i_x, x) in enumerate(xgrid)
            PFvec[i_x, :] = PF(i_x, x, s)
        end
        Vf[s] = LinInterp(xgrid, PFvec[:, 1])
        cf[s] = LinInterp(xgrid, PFvec[:, 2])
        nf[s] = LinInterp(xgrid, PFvec[:, 3])
        for sprime in 1:S
            xprimef[s, sprime] = LinInterp(xgrid, PFvec[:, 3+sprime])
        end
    end
    return Vf, [cf, nf, xprimef]
end

"""
Finds the optimal allocation given initial government debt `B_` and state `s_0`
"""
function time0_allocation(pab::RecursiveAllocation,
                          B_::AbstractFloat, s0::Integer)
    xgrid = pab.xgrid
    if pab.T.time_0 == false
        z0 = get_policies_time1(pab.T, i_x, x, s, pab.Vf)
    elseif pab.T.time_0 == true
        z0 = get_policies_time0(pab.T, B_, s0, pab.Vf)
    else
        error("T.time_0 is $(T.time_0), which is invalid")
    end
    c0, n0, xprime0 = z0[2], z0[3], z0[4:end]
    return c0, n0, xprime0
end

"""
Simulates Ramsey plan for `T` periods
"""
function simulate(pab::RecursiveAllocation,
                B_::AbstractFloat, s_0::Integer, T::Integer,
                sHist::Vector=QuantEcon.simulate(mc, s_0, T))
    model, S, policies = pab.model, pab.S, pab.policies
    β, Π, Uc = model.β, model.Π, model.Uc
    cf, nf, xprimef = policies[1], policies[2], policies[3]
    TF = typeof(model).parameters[1]
    cHist = Vector{TF}(T)
    nHist = Vector{TF}(T)
    Bhist = Vector{TF}(T)
    ΤHist = Vector{TF}(T)
    μHist = Vector{TF}(T)
    RHist = Vector{TF}(T-1)
    # time 0
    cHist[1], nHist[1], xprime = time0_allocation(pab, B_, s_0)
    ΤHist[1] = Τ(pab.model, cHist[1], nHist[1])[s_0]
    Bhist[1] = B_
    μHist[1] = 0.0
    # time 1 onward
    for t in 2:T
        s, x = sHist[t], xprime[sHist[t]]
        n = nf[s](x)
        c = [cf[shat](x) for shat in 1:S]
        xprime = [xprimef[s, sprime](x) for sprime in 1:S]
        ΤHist[t] = Τ(pab.model, c, n)[s]
        u_c = Uc(c, n)
        Eu_c = dot(Π[sHist[t-1], :], u_c)
        μHist[t] = pab.Vf[s](x)
        RHist[t-1] = Uc(cHist[t-1], nHist[t-1]) / (β * Eu_c)
        cHist[t], nHist[t], Bhist[t] = c[s], n, x / u_c[s]
    end
    return cHist, nHist, Bhist, ΤHist, sHist, μHist, RHist
end
