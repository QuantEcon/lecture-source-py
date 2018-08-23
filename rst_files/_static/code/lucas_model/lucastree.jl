#=

@authors : Spencer Lyon <spencer.lyon@nyu.edu>, John Stachurski

=#

using QuantEcon
using Distributions

"""
The Lucas asset pricing model --- parameters and grid data
"""
struct LucasTree{TF<:AbstractFloat}
    γ::TF                 # coefficient of risk aversion
    β::TF                 # Discount factor in (0, 1)
    α::TF                 # Correlation coefficient in the shock process
    σ::TF                 # Volatility of shock process
    ϕ::Distribution       # Distribution for shock process
    grid::Vector{TF}      # Grid of points on which to evaluate prices
    shocks::Vector{TF}    # Draws of the shock
    h::Vector{TF}         # The h function represented as a vector
end

"""
Constructor for the Lucas asset pricing model
"""
function LucasTree(;γ::AbstractFloat=2.0,
                β::AbstractFloat=0.95,
                α::AbstractFloat=0.9,
                σ::AbstractFloat=0.1,
                grid_size::Integer=100)

    ϕ = LogNormal(0.0, σ)
    shocks = rand(ϕ, 500)

    # == build a grid with mass around stationary distribution == #
    ssd = σ / sqrt(1 - α^2)
    grid_min, grid_max = exp(-4 * ssd), exp(4 * ssd)
    grid = collect(linspace(grid_min, grid_max, grid_size))

    # == set h(y) = β * int u'(G(y,z)) G(y,z) ϕ(dz) == #
    h = similar(grid)
    for (i, y) in enumerate(grid)
        h[i] = β * mean((y^α .* shocks).^(1 - γ))
    end

    return LucasTree(γ,
                     β,
                     α,
                     σ,
                     ϕ,
                     grid,
                     shocks,
                     h)
end


"""
The approximate Lucas operator, which computes and returns updated function
Tf on the grid points.
"""
function lucas_operator(lt::LucasTree, f::Vector)

    # == unpack names == #
    grid, α, β, h = lt.grid, lt.α, lt.β, lt.h
    z = lt.shocks

    Af = LinInterp(grid, f)

    Tf = [h[i] + β * mean(Af.(grid[i]^α.*z)) for i in 1:length(grid)]
    return Tf
end


"""
Compute the equilibrium price function associated with Lucas tree `lt`
"""
function solve_lucas_model(lt::LucasTree;
                           tol::AbstractFloat=1e-6, 
                           max_iter::Integer=500)
    
    # == simplify notation == #
    grid, γ = lt.grid, lt.γ
    
    i = 0
    f = zeros(grid)  # Initial guess of f
    error = tol + 1
    
    while (error > tol) && (i < max_iter)
        f_new = lucas_operator(lt, f)
        error = maximum(abs, f_new - f)
        f = f_new
        i += 1
    end
    
    # p(y) = f(y) * y ^ γ
    price = f .* grid.^γ
    
    return price
end
