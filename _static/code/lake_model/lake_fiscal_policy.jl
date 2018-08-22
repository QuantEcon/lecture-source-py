# Some global variables that will stay constant
α = 0.013
α_q = (1 - (1 - α)^3)
b_param = 0.0124
d_param = 0.00822
β = 0.98
γ = 1.0
σ = 2.0

# The default wage distribution: a discretized log normal
log_wage_mean, wage_grid_size, max_wage = 20, 200, 170
w_vec = linspace(1e-3, max_wage, wage_grid_size + 1)
logw_dist = Normal(log(log_wage_mean), 1)
cdf_logw = cdf.(logw_dist, log.(w_vec))
pdf_logw = cdf_logw[2:end] - cdf_logw[1:end-1]
p_vec = pdf_logw ./ sum(pdf_logw)
w_vec = (w_vec[1:end-1] + w_vec[2:end]) / 2

"""
Compute the reservation wage, job finding rate and value functions of the
workers given c and τ.

"""
function compute_optimal_quantities(c::AbstractFloat, τ::AbstractFloat)
    mcm = McCallModel(α_q,
                      β,
                      γ,
                      c-τ,                # post-tax compensation
                      σ,
                      collect(w_vec-τ),   # post-tax wages
                      p_vec)


    w_bar, V, U = compute_reservation_wage(mcm, return_values=true)
    λ = γ * sum(p_vec[w_vec - τ .> w_bar])

    return w_bar, λ, V, U
end

"""
Compute the steady state unemployment rate given c and tau using optimal
quantities from the McCall model and computing corresponding steady state
quantities

"""
function compute_steady_state_quantities(c::AbstractFloat, τ::AbstractFloat)
    w_bar, λ_param, V, U = compute_optimal_quantities(c, τ)

    # Compute steady state employment and unemployment rates
    lm = LakeModel(λ=λ_param, α=α_q, b=b_param, d=d_param)
    x = rate_steady_state(lm)
    u_rate, e_rate = x

    # Compute steady state welfare
    w = sum(V .* p_vec .* (w_vec - τ .> w_bar)) / sum(p_vec .* (w_vec - τ .> w_bar))
    welfare = e_rate .* w + u_rate .* U

    return u_rate, e_rate, welfare
end

"""
Find tax level that will induce a balanced budget.

"""
function find_balanced_budget_tax(c::Real)
    function steady_state_budget(t::Real)
      u_rate, e_rate, w = compute_steady_state_quantities(c, t)
      return t - u_rate * c
    end

    τ = brent(steady_state_budget, 0.0, 0.9 * c)

    return τ
end

# Levels of unemployment insurance we wish to study
Nc = 60
c_vec = linspace(5.0, 140.0, Nc)

tax_vec = Vector{Float64}(Nc)
unempl_vec = Vector{Float64}(Nc)
empl_vec = Vector{Float64}(Nc)
welfare_vec = Vector{Float64}(Nc)

for i = 1:Nc
    t = find_balanced_budget_tax(c_vec[i])
    u_rate, e_rate, welfare = compute_steady_state_quantities(c_vec[i], t)
    tax_vec[i] = t
    unempl_vec[i] = u_rate
    empl_vec[i] = e_rate
    welfare_vec[i] = welfare
end

fig, axes = subplots(2, 2, figsize=(15, 10))

plots = [unempl_vec, tax_vec, empl_vec, welfare_vec]
titles = ["Unemployment", "Tax", "Employment", "Welfare"]

for (ax, plot, title) in zip(axes, plots, titles)
     ax[:plot](c_vec, plot, "b-", lw=2, alpha=0.7)
     ax[:set](title=title)
     ax[:grid]("on")
end

fig[:tight_layout]()
