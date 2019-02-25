#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>

=#

using Distributions
using QuantEcon

# NOTE: only brute-force approach is available in bellman operator.
# Waiting on a simple constrained optimizer to be written in pure Julia

"""
A Jovanovic-type model of employment with on-the-job search.

The value function is given by

\\[V(x) = \\max_{ϕ, s} w(x, ϕ, s)\\]

for

    w(x, ϕ, s) := x(1 - ϕ - s) + β (1 - π(s)) V(G(x, ϕ)) +
                    β π(s) E V[ max(G(x, ϕ), U)

where

* `x`: human capital
* `s` : search effort
* `ϕ` : investment in human capital
* `π(s)` : probability of new offer given search level s
* `x(1 - ϕ - s)` : wage
* `G(x, ϕ)` : new human capital when current job retained
* `U` : Random variable with distribution F -- new draw of human capital

##### Fields

- `A::Real` : Parameter in human capital transition function
- `α::Real` : Parameter in human capital transition function
- `β::AbstractFloat` : Discount factor in (0, 1)
- `x_grid::AbstractVector` : Grid for potential levels of x
- `G::Function` : Transition `function` for human captial
- `π_func::Function` : `function` mapping search effort to
   the probability of getting a new job offer
- `F::UnivariateDistribution` : A univariate distribution from which
   the value of new job offers is drawn
- `quad_nodes::Vector` : Quadrature nodes for integrating over ϕ
- `quad_weights::Vector` : Quadrature weights for integrating over ϕ
- `ϵ::AbstractFloat` : A small number, used in optimization routine

"""
struct JvWorker{TR <: Real,
                TF <: AbstractFloat,
                TUD <: UnivariateDistribution,
                TAV <: AbstractVector,
                TV <: Vector}
    A::TR
    α::TR
    β::TF
    x_grid::TAV
    G::Function
    π_func::Function
    F::TUD
    quad_nodes::TV
    quad_weights::TV
    ϵ::TF

end

"""
Constructor with default values for `JvWorker`

##### Arguments

 - `A::Real(1.4)` : Parameter in human capital transition function
 - `α::Real(0.6)` : Parameter in human capital transition function
 - `β::Real(0.96)` : Discount factor in (0, 1)
 - `grid_size::Integer(50)` : Number of points in discrete grid for `x`
 - `ϵ::Float(1e-4)` : A small number, used in optimization routine

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter

"""
# use key word argument
function JvWorker(;A::Real=1.4,
                   α::Real=0.6,
                   β::Real=0.96,
                   grid_size::Integer=50,
                   ϵ::AbstractFloat=1e-4)

    G(x, ϕ) = A .* (x .* ϕ).^α
    π_func = sqrt
    F = Beta(2, 2)

    # integration bounds
    a, b = quantile(F, 0.005), quantile(F, 0.995)

    # quadrature nodes/weights
    nodes, weights = qnwlege(21, a, b)

    # Set up grid over the state space for DP
    # Max of grid is the max of a large quantile value for F and the
    # fixed point y = G(y, 1).
    grid_max = max(A^(1.0 / (1.0 - α)), quantile(F, 1 - ϵ))

    # range for linspace(ϵ, grid_max, grid_size). Needed for
    # CoordInterpGrid below
    x_grid = linspace(ϵ, grid_max, grid_size)

    JvWorker(A, α, β, x_grid, G, π_func, F, nodes, weights, ϵ)
end


"""
Apply the Bellman operator for a given model and initial value,
returning only the value function

##### Arguments

- `jv::JvWorker` : Instance of `JvWorker`
- `V::Vector`: Current guess for the value function
- `new_V::Vector` : Storage for updated value function

##### Returns

None, `new_V` is updated in place with the value function.

##### Notes

Currently, only the brute-force approach is available.
We are waiting on a simple constrained optimizer to be written in pure Julia

"""
function bellman_operator!(jv::JvWorker, V::AbstractVector, new_V::AbstractVector)

    # simplify notation
    G, π_func, F, β, ϵ = jv.G, jv.π_func, jv.F, jv.β, jv.ϵ
    nodes, weights = jv.quad_nodes, jv.quad_weights

    # prepare interpoland of value function
    Vf = LinInterp(jv.x_grid, V)

    # instantiate the linesearch variables
    max_val = -1.0
    cur_val = 0.0
    max_s = 1.0
    max_ϕ = 1.0
    search_grid = linspace(ϵ, 1.0, 15)

    for (i, x) in enumerate(jv.x_grid)

        function w(z)
            s, ϕ = z
            h(u) = [Vf(max(G(x, ϕ), uval)) * pdf(F, uval) for uval in u]
            integral = do_quad(h, nodes, weights)
            q = π_func(s) * integral + (1.0 - π_func(s)) * Vf(G(x, ϕ))

            return - x * (1.0 - ϕ - s) - β * q
        end

        for s in search_grid
            for ϕ in search_grid
                cur_val = ifelse(s + ϕ <= 1.0, -w((s, ϕ)), -1.0)
                if cur_val > max_val
                    max_val, max_s, max_ϕ = cur_val, s, ϕ
                end
            end
        end

        new_V[i] = max_val
    end
end

"""
Apply the Bellman operator for a given model and initial value, returning policies

##### Arguments

- `jv::JvWorker` : Instance of `JvWorker`
- `V::Vector`: Current guess for the value function
- `out::Tuple{Vector, Vector}` : Storage for the two policy rules

##### Returns

None, `out` is updated in place with the two policy functions.

##### Notes

Currently, only the brute-force approach is available.
We are waiting on a simple constrained optimizer to be written in pure Julia

"""
function bellman_operator!(jv::JvWorker,
                           V::AbstractVector,
                           out::Tuple{AbstractVector, AbstractVector})

    # simplify notation
    G, π_func, F, β, ϵ = jv.G, jv.π_func, jv.F, jv.β, jv.ϵ
    nodes, weights = jv.quad_nodes, jv.quad_weights

    # prepare interpoland of value function
    Vf = LinInterp(jv.x_grid, V)

    # instantiate variables
    s_policy, ϕ_policy = out[1], out[2]

    # instantiate the linesearch variables
    max_val = -1.0
    cur_val = 0.0
    max_s = 1.0
    max_ϕ = 1.0
    search_grid = linspace(ϵ, 1.0, 15)

    for (i, x) in enumerate(jv.x_grid)

        function w(z)
            s, ϕ = z
            h(u) = [Vf(max(G(x, ϕ), uval)) * pdf(F, uval) for uval in u]
            integral = do_quad(h, nodes, weights)
            q = π_func(s) * integral + (1.0 - π_func(s)) * Vf(G(x, ϕ))

            return - x * (1.0 - ϕ - s) - β * q
        end

        for s in search_grid
            for ϕ in search_grid
                cur_val = ifelse(s + ϕ <= 1.0, -w((s, ϕ)), -1.0)
                if cur_val > max_val
                    max_val, max_s, max_ϕ = cur_val, s, ϕ
                end
            end
        end

      s_policy[i], ϕ_policy[i] = max_s, max_ϕ
  end
end

function bellman_operator(jv::JvWorker, V::AbstractVector; ret_policies::Bool=false)
    out = ifelse(ret_policies, (similar(V), similar(V)), similar(V))
    bellman_operator!(jv, V, out)
    return out
end
