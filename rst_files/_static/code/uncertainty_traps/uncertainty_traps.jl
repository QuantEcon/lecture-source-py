mutable struct UncertaintyTrapEcon{TF<:AbstractFloat, TI<:Integer}
    a::TF          # Risk aversion
    γ_x::TF        # Production shock precision
    ρ::TF          # Correlation coefficient for θ
    σ_θ::TF        # Standard dev of θ shock
    num_firms::TI  # Number of firms
    σ_F::TF        # Std dev of fixed costs
    c::TF          # External opportunity cost
    μ::TF          # Initial value for μ
    γ::TF          # Initial value for γ
    θ::TF          # Initial value for θ
    σ_x::TF        # Standard deviation of shock
end

function UncertaintyTrapEcon(;a::AbstractFloat=1.5, γ_x::AbstractFloat=0.5,
                             ρ::AbstractFloat=0.99, σ_θ::AbstractFloat=0.5,
                             num_firms::Integer=100, σ_F::AbstractFloat=1.5,
                             c::AbstractFloat=-420.0, μ_init::AbstractFloat=0.0,
                             γ_init::AbstractFloat=4.0,
                             θ_init::AbstractFloat=0.0)
    σ_x = sqrt(a / γ_x)
    UncertaintyTrapEcon(a, γ_x, ρ, σ_θ, num_firms, σ_F, c, μ_init,
                        γ_init, θ_init, σ_x)

end

function ψ(uc::UncertaintyTrapEcon, F::Real)
    temp1 = -uc.a * (uc.μ - F)
    temp2 = 0.5 * uc.a^2 * (1 / uc.γ + 1 / uc.γ_x)
    return (1 / uc.a) * (1 - exp(temp1 + temp2)) - uc.c
end

"""
Update beliefs (μ, γ) based on aggregates X and M.
"""
function update_beliefs!(uc::UncertaintyTrapEcon, X::Real, M::Real)
    # Simplify names
    γ_x, ρ, σ_θ = uc.γ_x, uc.ρ, uc.σ_θ

    # Update μ
    temp1 = ρ * (uc.γ * uc.μ + M * γ_x * X)
    temp2 = uc.γ + M * γ_x
    uc.μ =  temp1 / temp2

    # Update γ
    uc.γ = 1 / (ρ^2 / (uc.γ + M * γ_x) + σ_θ^2)
end

update_θ!(uc::UncertaintyTrapEcon, w::Real) =
    (uc.θ = uc.ρ * uc.θ + uc.σ_θ * w)

"""
Generate aggregates based on current beliefs (μ, γ).  This
is a simulation step that depends on the draws for F.
"""
function gen_aggregates(uc::UncertaintyTrapEcon)
    F_vals = uc.σ_F * randn(uc.num_firms)

    M = sum(ψ.(uc, F_vals) .> 0)  # Counts number of active firms
    if M > 0
        x_vals = uc.θ + uc.σ_x * randn(M)
        X = mean(x_vals)
    else
        X = 0.0
    end
    return X, M
end
