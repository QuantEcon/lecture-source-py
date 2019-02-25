"""
Computes the path of consumption and debt for the previously described
complete markets model where exogenous income follows a linear
state space
"""
function complete_ss(β::AbstractFloat,
                     b0::Union{AbstractFloat, Array},
                     x0::Union{AbstractFloat, Array},
                     A::Union{AbstractFloat, Array},
                     C::Union{AbstractFloat, Array},
                     S_y::Union{AbstractFloat, Array},
                     T::Integer=12)

    # Create a linear state space for simulation purposes
    # This adds "b" as a state to the linear state space system
    # so that setting the seed places shocks in same place for
    # both the complete and incomplete markets economy
    # Atilde = vcat(hcat(A, zeros(size(A,1), 1)),
    #               zeros(1, size(A,2) + 1))
    # Ctilde = vcat(C, zeros(1, 1))
    # S_ytilde = hcat(S_y, zeros(1, 1))

    lss = LSS(A, C, S_y, mu_0=x0)

    # Add extra state to initial condition
    # x0 = hcat(x0, 0)

    # Compute the (I - β*A)^{-1}
    rm = inv(eye(size(A, 1)) - β * A)

    # Constant level of consumption
    cbar = (1 - β) * (S_y * rm * x0 - b0)
    c_hist = ones(T) * cbar[1]

    # Debt
    x_hist, y_hist = simulate(lss, T)
    b_hist = (S_y * rm * x_hist - cbar[1] / (1.0 - β))


    return c_hist, vec(b_hist), vec(y_hist), x_hist
end

N_simul = 150

# Define parameters
α, ρ1, ρ2 = 10.0, 0.9, 0.0
σ = 1.0
# N_simul = 1
# T = N_simul
A = [1.0 0.0 0.0;
     α    ρ1  ρ2;
     0.0 1.0 0.0]
C = [0.0, σ, 0.0]
S_y = [1.0 1.0 0.0]
β, b0 = 0.95, -10.0
x0 = [1.0, α / (1 - ρ1), α / (1 - ρ1)]

# Do simulation for complete markets
s = rand(1:10000)
srand(s)  # Seeds get set the same for both economies
out = complete_ss(β, b0, x0, A, C, S_y, 150)
c_hist_com, b_hist_com, y_hist_com, x_hist_com = out


fig, ax = subplots(1, 2, figsize=(15, 5))

# Consumption plots
ax[1][:set_title]("Cons and income")
ax[1][:plot](1:N_simul, c_hist_com, label="consumption")
ax[1][:plot](1:N_simul, y_hist_com, label="income",
            lw=2, alpha=0.6, ls="--")
ax[1][:legend]()
ax[1][:set_xlabel]("Periods")
ax[1][:set_ylim]([-5.0, 110])

# Debt plots
ax[2][:set_title]("Debt and income")
ax[2][:plot](1:N_simul, b_hist_com, label="debt")
ax[2][:plot](1:N_simul, y_hist_com, label="Income",
            lw=2, alpha=0.6, ls="--")
ax[2][:legend]()
ax[2][:axhline](0, color="k", ls="--")
ax[2][:set_xlabel]("Periods")