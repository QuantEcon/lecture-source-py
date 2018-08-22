#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>

=#
using QuantEcon
using Plots
pyplot()

# model parameters
a_0 = 100
a_1 = 0.5
ρ = 0.9
σ_d = 0.05
β = 0.95
c = 2
γ = 50.0
θ = 0.002
ac = (a_0 - c) / 2.0

# Define LQ matrices
R = [ 0   ac    0;
     ac -a_1  0.5;
     0.  0.5    0]
R = -R  # For minimization
Q = Matrix([γ / 2.0]')
A = [1. 0. 0.;
     0. 1. 0.;
     0. 0.  ρ]
B = [0. 1. 0.]'
C = [0. 0. σ_d]'

## Functions

function evaluate_policy(θ::AbstractFloat, F::AbstractArray)
    rlq = RBLQ(Q, R, A, B, C, β, θ)
    K_F, P_F, d_F, O_F, o_F = evaluate_F(rlq, F)
    x0 = [1.0 0.0 0.0]'
    value = - x0' * P_F * x0 - d_F
    entropy = x0' * O_F * x0 + o_F
    return value[1], entropy[1]    # return scalars
end


function value_and_entropy{TF<:AbstractFloat}(emax::AbstractFloat,
                                              F::AbstractArray{TF},
                                              bw::String,
                                              grid_size::Integer=1000)
    if lowercase(bw) == "worst"
        θs = 1 ./ linspace(1e-8, 1000, grid_size)
    else
        θs = -1 ./ linspace(1e-8, 1000, grid_size)
    end

    data = Array{TF}(grid_size, 2)

    for (i, θ) in enumerate(θs)
        data[i, :] = collect(evaluate_policy(θ, F))
        if data[i, 2] >= emax      # stop at this entropy level
            data = data[1:i, :]
            break
        end
    end
    return data
end

## Main

# compute optimal rule
optimal_lq = LQ(Q, R, A, B, C, zero(B'A), bet=β)
Po, Fo, Do = stationary_values(optimal_lq)

# compute robust rule for our θ
baseline_robust = RBLQ(Q, R, A, B, C, β, θ)
Fb, Kb, Pb = robust_rule(baseline_robust)

# Check the positive definiteness of worst-case covariance matrix to
# ensure that θ exceeds the breakdown point
test_matrix = eye(size(Pb, 1)) - (C' * Pb * C ./ θ)[1]
eigenvals, eigenvecs = eig(test_matrix)
@assert all(eigenvals .>= 0)

emax = 1.6e6

# compute values and entropies
optimal_best_case = value_and_entropy(emax, Fo, "best")
robust_best_case = value_and_entropy(emax, Fb, "best")
optimal_worst_case = value_and_entropy(emax, Fo, "worst")
robust_worst_case = value_and_entropy(emax, Fb, "worst")

# we reverse order of "worst_case"s so values are ascending
data_pairs = ((optimal_best_case, optimal_worst_case),
              (robust_best_case, robust_worst_case))

egrid = linspace(0, emax, 100)
egrid_data = Array{Float64}[]
for data_pair in data_pairs
    for data in data_pair
        x, y = data[:, 2], data[:, 1]
        curve = LinInterp(x, y)
        push!(egrid_data, curve.(egrid))
    end
end
plot(egrid, egrid_data, color=[:red :red :blue :blue])
plot!(egrid, egrid_data[1], fillrange=egrid_data[2],
      fillcolor=:red, fillalpha=0.1, color=:red, legend=:none)
plot!(egrid, egrid_data[3], fillrange=egrid_data[4],
      fillcolor=:blue, fillalpha=0.1, color=:blue, legend=:none)
plot!(xlabel="Entropy", ylabel="Value")
