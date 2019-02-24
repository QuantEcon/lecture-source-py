# Firms' parameters
const A = 1
const N = 1
const α = 0.33
const β = 0.96
const δ = 0.05

"""
Compute wage rate given an interest rate, r
"""
function r_to_w(r::Real)
    return A * (1 - α) * (A * α / (r + δ)) ^ (α / (1 - α))
end

"""
Inverse demand curve for capital. The interest rate
associated with a given demand for capital K.
"""
function rd(K::Real)
    return A * α * (N / K) ^ (1 - α) - δ
end

"""
Map prices to the induced level of capital stock.

##### Arguments
- `am::Household` : Household instance for problem we want to solve
- `r::Real` : interest rate

##### Returns
- The implied level of aggregate capital
"""
function prices_to_capital_stock(am::Household, r::Real)

    # Set up problem
    w = r_to_w(r)
    setup_R!(am, r, w)
    aiyagari_ddp = DiscreteDP(am.R, am.Q, am.β)

    # Compute the optimal policy
    results = solve(aiyagari_ddp, PFI)

    # Compute the stationary distribution
    stationary_probs = stationary_distributions(results.mc)[:, 1][1]

    # Return K
    return dot(am.s_vals[:, 1], stationary_probs)
end

# Create an instance of Household
am = Household(β=β, a_max=20.0)

# Create a grid of r values at which to compute demand and supply of capital
num_points = 20
r_vals = linspace(0.005, 0.04, num_points)

# Compute supply of capital
k_vals = prices_to_capital_stock.(am, r_vals)

# Plot against demand for capital by firms
demand = rd.(k_vals)
labels =  ["demand for capital" "supply of capital"]
plot(k_vals, [demand r_vals], label=labels, lw=2, alpha=0.6)
plot!(xlabel="capital", ylabel="interest rate", xlim=(2, 14), ylim=(0.0, 0.1))
