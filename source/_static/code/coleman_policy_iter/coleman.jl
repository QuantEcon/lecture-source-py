#=

Author: Shunsuke Hori

=#

using QuantEcon

"""
g: input policy function
grid: grid points
β: discount factor
u_prime: derivative of utility function
f: production function
f_prime: derivative of production function
shocks::shock draws, used for Monte Carlo integration to compute expectation
Kg: output value is stored
"""
function coleman_operator!(g::AbstractVector,
                           grid::AbstractVector,
                           β::AbstractFloat,
                           u_prime::Function,
                           f::Function,
                           f_prime::Function,
                           shocks::AbstractVector,
                           Kg::AbstractVector=similar(g))

    # This function requires the container of the output value as argument Kg

    # Construct linear interpolation object #
    g_func = LinInterp(grid, g)

    # solve for updated consumption value #
    for (i, y) in enumerate(grid)
        function h(c)
            vals = u_prime.(g_func.(f(y - c) * shocks)) .* f_prime(y - c) .* shocks
            return u_prime(c) - β * mean(vals)
        end
        Kg[i] = brent(h, 1e-10, y - 1e-10)
    end
    return Kg
end

# The following function does NOT require the container of the output value as argument
function coleman_operator(g::AbstractVector,
                          grid::AbstractVector,
                          β::AbstractFloat,
                          u_prime::Function,
                          f::Function,
                          f_prime::Function,
                          shocks::AbstractVector)

    return coleman_operator!(g, grid, β, u_prime,
                             f, f_prime, shocks, similar(g))
end
