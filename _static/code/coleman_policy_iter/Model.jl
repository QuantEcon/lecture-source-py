#=

Author: Shunsuke Hori

=#

struct Model{TF <: AbstractFloat, TR <: Real, TI <: Integer}
    α::TR              # Productivity parameter
    β::TF              # Discount factor
    γ::TR              # risk aversion
    μ::TR              # First parameter in lognorm(μ, σ)
    s::TR              # Second parameter in lognorm(μ, σ)
    grid_min::TR       # Smallest grid point
    grid_max::TR       # Largest grid point
    grid_size::TI      # Number of grid points
    u::Function        # utility function
    u_prime::Function  # derivative of utility function
    f::Function        # production function
    f_prime::Function  # derivative of production function
    grid::Vector{TR}   # grid
end

"""
construct Model instance using the information of parameters and functional form

arguments: see above

return: Model type instance
"""
function Model(; α::Real=0.65,                      # Productivity parameter
                 β::AbstractFloat=0.95,             # Discount factor
                 γ::Real=1.0,                       # risk aversion
                 μ::Real=0.0,                       # First parameter in lognorm(μ, σ)
                 s::Real=0.1,                       # Second parameter in lognorm(μ, σ)
                 grid_min::Real=1e-6,               # Smallest grid point
                 grid_max::Real=4.0,                # Largest grid point
                 grid_size::Integer=200,            # Number of grid points
                 u::Function= c->(c^(1-γ)-1)/(1-γ), # utility function
                 u_prime::Function = c-> c^(-γ),    # u'
                 f::Function = k-> k^α,             # production function
                 f_prime::Function = k -> α*k^(α-1) # f'
                 )

    grid = collect(linspace(grid_min, grid_max, grid_size))

    if γ == 1                                       # when γ==1, log utility is assigned
        u_log(c) = log(c)
        m = Model(α, β, γ, μ, s, grid_min, grid_max,
                  grid_size, u_log, u_prime, f, f_prime, grid)
    else
        m = Model(α, β, γ, μ, s, grid_min, grid_max,
                  grid_size, u, u_prime, f, f_prime, grid)
    end
    return m
end
