using QuantEcon
using Optim

# utility and marginal utility functions
u(x) = log(x)
du(x) = 1 / x

"""
Income fluctuation problem

##### Fields

- `r::Real` : Strictly positive interest rate
- `R::Real` : The interest rate plus 1 (strictly greater than 1)
- `β::Real` : Discount rate in (0, 1)
- `b::Real` :  The borrowing constraint
- `Π::Matrix{T} where T<:Real` : Transition matrix for `z`
- `z_vals::Vector{T} where T<:Real` : Levels of productivity
- `asset_grid::AbstractArray` : Grid of asset values

"""
struct ConsumerProblem{T <: Real}
    r::T
    R::T
    β::T
    b::T
    Π::Matrix{T}
    z_vals::Vector{T}
    asset_grid::AbstractArray
end

function ConsumerProblem(;r=0.01, 
                         β=0.96, 
                         Π=[0.6 0.4; 0.05 0.95],
                         z_vals=[0.5, 1.0], 
                         b=0.0, 
                         grid_max=16, 
                         grid_size=50)
    R = 1 + r
    asset_grid = linspace(-b, grid_max, grid_size)

    ConsumerProblem(r, R, β, b, Π, z_vals, asset_grid)
end


"""
Apply the Bellman operator for a given model and initial value.

##### Arguments

- `cp::ConsumerProblem` : Instance of `ConsumerProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output
- `;ret_policy::Bool(false)`: Toggles return of value or policy functions

##### Returns

None, `out` is updated in place. If `ret_policy == true` out is filled with the
policy function, otherwise the value function is stored in `out`.

"""
function bellman_operator!(cp::ConsumerProblem, 
                           V::Matrix, 
                           out::Matrix;
                           ret_policy::Bool=false)

    # simplify names, set up arrays
    R, Π, β, b = cp.R, cp.Π, cp.β, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    z_idx = 1:length(z_vals)
    
    # value function when the shock index is z_i
    vf = interp(asset_grid, V)
    
    opt_lb = 1e-8

    # solve for RHS of Bellman equation
    for (i_z, z) in enumerate(z_vals)
        for (i_a, a) in enumerate(asset_grid)

            function obj(c)
                EV = dot(vf.(R * a + z - c, z_idx), Π[i_z, :]) # compute expectation
                return -u(c)  - β * EV
            end
            res = optimize(obj, opt_lb, R .* a .+ z .+ b)
            c_star = Optim.minimizer(res)

            if ret_policy
                out[i_a, i_z] = c_star
            else
               out[i_a, i_z] = - Optim.minimum(res)
            end

        end
    end
    out
end

bellman_operator(cp::ConsumerProblem, V::Matrix; ret_policy=false) =
    bellman_operator!(cp, V, similar(V); ret_policy=ret_policy)

"""
Extract the greedy policy (policy function) of the model.

##### Arguments

- `cp::CareerWorkerProblem` : Instance of `CareerWorkerProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output

##### Returns

None, `out` is updated in place to hold the policy function

"""
get_greedy!(cp::ConsumerProblem, V::Matrix, out::Matrix) =
    update_bellman!(cp, V, out, ret_policy=true)

get_greedy(cp::ConsumerProblem, V::Matrix) =
    update_bellman(cp, V, ret_policy=true)

"""
The approximate Coleman operator.

Iteration with this operator corresponds to policy function
iteration. Computes and returns the updated consumption policy
c.  The array c is replaced with a function cf that implements
univariate linear interpolation over the asset grid for each
possible value of z.

##### Arguments

- `cp::CareerWorkerProblem` : Instance of `CareerWorkerProblem`
- `c::Matrix`: Current guess for the policy function
- `out::Matrix` : Storage for output

##### Returns

None, `out` is updated in place to hold the policy function

"""
function coleman_operator!(cp::ConsumerProblem, c::Matrix, out::Matrix)
    # simplify names, set up arrays
    R, Π, β, b = cp.R, cp.Π, cp.β, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    z_idx = 1:length(z_vals)
    gam = R * β
    
    # policy function when the shock index is z_i
    cf = interp(asset_grid, c)

    # compute lower_bound for optimization
    opt_lb = 1e-8

    for (i_z, z) in enumerate(z_vals)
        for (i_a, a) in enumerate(asset_grid)
            function h(t)
                cps = cf.(R * a + z - t, z_idx) # c' for each z'
                expectation = dot(du.(cps), Π[i_z, :])
                return abs(du(t) - max(gam * expectation, du(R * a + z + b)))
            end
            opt_ub = R*a + z + b  # addresses issue #8 on github
            res = optimize(h, min(opt_lb, opt_ub - 1e-2), opt_ub,
                           method=Optim.Brent())
            out[i_a, i_z] = Optim.minimizer(res)
        end
    end
    out
end


"""
Apply the Coleman operator for a given model and initial value

See the specific methods of the mutating version of this function for more
details on arguments
"""
coleman_operator(cp::ConsumerProblem, c::Matrix) =
    coleman_operator!(cp, c, similar(c))

function initialize(cp::ConsumerProblem)
    # simplify names, set up arrays
    R, β, b = cp.R, cp.β, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    shape = length(asset_grid), length(z_vals)
    V, c = Array{Float64}(shape...), Array{Float64}(shape...)

    # Populate V and c
    for (i_z, z) in enumerate(z_vals)
        for (i_a, a) in enumerate(asset_grid)
            c_max = R * a + z + b
            c[i_a, i_z] = c_max
            V[i_a, i_z] = u(c_max) ./ (1 - β)
        end
    end

    return V, c
end
