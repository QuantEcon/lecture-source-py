using QuantEcon.compute_fixed_point, QuantEcon.DiscreteRV, QuantEcon.draw, QuantEcon.LinInterp

"""
For a given probability return expected loss of choosing model 0
"""
expect_loss_choose_0(p::Real, L0::Real) = (1 - p) * L0

"""
For a given probability return expected loss of choosing model 1
"""
expect_loss_choose_1(p::Real, L1::Real) = p * L1

"""
We will need to be able to evaluate the expectation of our Bellman
equation J. In order to do this, we need the current probability
that model 0 is correct (p), the distributions (f0, f1), and a
function that can evaluate the Bellman equation
"""
function EJ(p::Real, f0::AbstractVector, f1::AbstractVector, J::LinInterp)
    # Get the current distribution we believe (p * f0 + (1 - p) * f1)
    curr_dist = p * f0 + (1 - p) * f1

    # Get tomorrow's expected distribution through Bayes law
    tp1_dist = clamp.((p * f0) ./ (p * f0 + (1 - p) * f1), 0, 1)

    # Evaluate the expectation
    EJ = dot(curr_dist, J.(tp1_dist))

    return EJ
end

expect_loss_cont(p::Real, c::Real,
                 f0::AbstractVector, f1::AbstractVector, J::LinInterp) =
    c + EJ(p, f0, f1, J)

"""
Evaluates the value function for a given continuation value
function; that is, evaluates

    J(p) = min(pL0, (1-p)L1, c + E[J(p')])

Uses linear interpolation between points
"""
function bellman_operator(pgrid::AbstractVector,
                          c::Real,
                          f0::AbstractVector,
                          f1::AbstractVector,
                          L0::Real,
                          L1::Real,
                          J::AbstractVector)
    m = length(pgrid)
    @assert m == length(J)

    J_out = zeros(m)
    J_interp = LinInterp(pgrid, J)

    for (p_ind, p) in enumerate(pgrid)
        # Payoff of choosing model 0
        p_c_0 = expect_loss_choose_0(p, L0)
        p_c_1 = expect_loss_choose_1(p, L1)
        p_con = expect_loss_cont(p, c, f0, f1, J_interp)

        J_out[p_ind] = min(p_c_0, p_c_1, p_con)
    end

    return J_out
end

# Create two distributions over 50 values for k
# We are using a discretized beta distribution

p_m1 = linspace(0, 1, 50)
f0 = clamp.(pdf.(Beta(1, 1), p_m1), 1e-8, Inf)
f0 = f0 / sum(f0)
f1 = clamp.(pdf.(Beta(9, 9), p_m1), 1e-8, Inf)
f1 = f1 / sum(f1)

# To solve
pg = linspace(0, 1, 251)
J1 = compute_fixed_point(x -> bellman_operator(pg, 0.5, f0, f1, 5.0, 5.0, x),
    zeros(length(pg)), err_tol=1e-6, print_skip=5);
