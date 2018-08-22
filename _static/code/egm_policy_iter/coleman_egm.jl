#=

Authors: Shunsuke Hori

=#

"""
The approximate Coleman operator, updated using the endogenous grid
method.

Parameters
----------
g : Function
    The current guess of the policy function
k_grid : AbstractVector
    The set of *exogenous* grid points, for capital k = y - c
β : AbstractFloat
    The discount factor
u_prime : Function
    The derivative u'(c) of the utility function
u_prime_inv : Function
    The inverse of u' (which exists by assumption)
f : Function
    The production function f(k)
f_prime : Function
    The derivative f'(k)
shocks : AbstractVector
    An array of draws from the shock, for Monte Carlo integration (to
    compute expectations).
"""

function coleman_egm(g::Function,
                     k_grid::AbstractVector,
                     β::AbstractFloat,
                     u_prime::Function,
                     u_prime_inv::Function,
                     f::Function,
                     f_prime::Function,
                     shocks::AbstractVector)

    # Allocate memory for value of consumption on endogenous grid points
    c = similar(k_grid)

    # Solve for updated consumption value
    for (i, k) in enumerate(k_grid)
        vals = u_prime.(g.(f(k) * shocks)) .* f_prime(k) .* shocks
        c[i] = u_prime_inv(β * mean(vals))
    end

    # Determine endogenous grid
    y = k_grid + c  # y_i = k_i + c_i

    # Update policy function and return
    Kg = LinInterp(y,c)
    Kg_f(x) = Kg(x)
    return Kg_f
end
