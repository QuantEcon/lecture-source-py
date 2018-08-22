using QuantEcon

"""
The data for a consumption problem, including some default values.
"""
struct ConsumptionProblem{TF<:AbstractFloat}
    β::TF
    y::Vector{TF}
    b0::TF
    P::Matrix{TF}
end

"""
Parameters
----------

β : discount factor
P : 2x2 transition matrix
y : Array containing the two income levels
b0 : debt in period 0 (= state_1 debt level)
"""
function ConsumptionProblem(β = 0.96,
                            y = [2.0, 1.5],
                            b0 = 3.0,
                            P = [0.8 0.2;
                                 0.4 0.6])

    ConsumptionProblem(β, y, b0, P)
end

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

function consumption_complete(cp::ConsumptionProblem)

    β, P, y, b0 = cp.β, cp.P, cp.y, cp.b0   # Unpack

    y1, y2 = y                              # extract income levels
    b1 = b0                                 # b1 is known to be equal to b0
    Q = β * P                               # assumed price system

    # Using equation (7) calculate b2
    b2 = (y2 - y1 - (Q[1, 1] - Q[2, 1] - 1) * b1) / (Q[1, 2] + 1 - Q[2, 2])

    # Using equation (5) calculae c_bar
    c_bar = y1 - b0 + ([b1 b2] * Q[1, :] )[1]

    return c_bar, b1, b2
end

"""
Computes endogenous values for the incomplete market case.

Parameters
----------

cp : instance of ConsumptionProblem
N_simul : Integer

"""
function consumption_incomplete(cp::ConsumptionProblem;
                                N_simul::Integer=150)

    β, P, y, b0 = cp.β, cp.P, cp.y, cp.b0  # Unpack
    # For the simulation use the MarkovChain type
    mc = MarkovChain(P)

    # Useful variables
    y = y''
    v = inv(eye(2) - β * P) * y

    # Simulat state path
    s_path = simulate(mc, N_simul, init=1)

    # Store consumption and debt path
    b_path, c_path = ones(N_simul + 1), ones(N_simul)
    b_path[1] = b0

    # Optimal decisions from (12) and (13)
    db = ((1 - β) * v - y) / β

    for (i, s) in enumerate(s_path)
        c_path[i] = (1 - β) * (v - b_path[i] * ones(2, 1))[s, 1]
        b_path[i + 1] = b_path[i] + db[s, 1]
    end

    return c_path, b_path[1:end-1], y[s_path], s_path
end