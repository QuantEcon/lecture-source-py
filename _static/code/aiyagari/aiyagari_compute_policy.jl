using LaTeXStrings
using Plots
pyplot()

# Example prices
r = 0.03
w = 0.956

# Create an instance of Household
am = Household(a_max=20.0, r=r, w=w)

# Use the instance to build a discrete dynamic program
am_ddp = DiscreteDP(am.R, am.Q, am.β)

# Solve using policy function iteration
results = solve(am_ddp, PFI)

# Simplify names
z_size, a_size  = am.z_size, am.a_size
z_vals, a_vals = am.z_chain.state_values, am.a_vals
n = am.n

# Get all optimal actions across the set of
# a indices with z fixed in each column
a_star = reshape([a_vals[results.sigma[s_i]] for s_i in 1:n], a_size, z_size)

labels = [latexstring("\z = $(z_vals[1])") latexstring("\z = $(z_vals[2])")]
plot(a_vals, a_star, label=labels, lw=2, alpha=0.6)
plot!(a_vals, a_vals, label="", color=:black, linestyle=:dash)
plot!(xlabel="current assets", ylabel="next period assets", grid=false)
