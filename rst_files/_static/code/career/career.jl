"""
Career/job choice model of Derek Neal (1999)

##### Fields

- `β::AbstractFloat` : Discount factor in (0, 1)
- `N::Integer` : Number of possible realizations of both ϵ and θ
- `B::AbstractFloat` : upper bound for both ϵ and θ
- `θ::AbstractVector` : A grid of values on [0, B]
- `ϵ::AbstractVector` : A grid of values on [0, B]
- `F_probs::AbstractVector` : The pdf of each value associated with of F
- `G_probs::AbstractVector` : The pdf of each value associated with of G
- `F_mean::AbstractFloat` : The mean of the distribution F
- `G_mean::AbstractFloat` : The mean of the distribution G

"""

struct CareerWorkerProblem{TF<:AbstractFloat,
                           TI<:Integer,
                           TAV<:AbstractVector{TF},
                           TAV2<:AbstractVector{TF}}
    β::TF
    N::TI
    B::TF
    θ::TAV
    ϵ::TAV
    F_probs::TAV2
    G_probs::TAV2
    F_mean::TF
    G_mean::TF
end


"""
Constructor with default values for `CareerWorkerProblem`

##### Arguments

- `β::Real(0.95)` : Discount factor in (0, 1)
- `B::Real(5.0)` : upper bound for both ϵ and θ
- `N::Real(50)` : Number of possible realizations of both ϵ and θ
- `F_a::Real(1), F_b::Real(1)` : Parameters of the distribution F
- `G_a::Real(1), G_b::Real(1)` : Parameters of the distribution F

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter
"""
# use key word argument
function CareerWorkerProblem{TF<:AbstractFloat}(;β::TF=0.95,
                                                 B::TF=5.0,
                                                 N::Integer=50,
                                                 F_a::TF=1.0,
                                                 F_b::TF=1.0,
                                                 G_a::TF=1.0,
                                                 G_b::TF=1.0)
    θ = linspace(0, B, N)
    ϵ = copy(θ)
    dist_F = BetaBinomial(N-1, F_a, F_b)
    dist_G = BetaBinomial(N-1, G_a, G_b)
    F_probs = pdf.(dist_F, support(dist_F))
    G_probs = pdf.(dist_G, support(dist_G))
    F_mean = sum(θ .* F_probs)
    G_mean = sum(ϵ .* G_probs)
    CareerWorkerProblem(β, N, B, θ, ϵ, F_probs, G_probs, F_mean, G_mean)
end


"""
Apply the Bellman operator for a given model and initial value.

##### Arguments

- `cp::CareerWorkerProblem` : Instance of `CareerWorkerProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output
- `;ret_policy::Bool(false)`: Toggles return of value or policy functions

##### Returns

None, `out` is updated in place. If `ret_policy == true` out is filled with the
policy function, otherwise the value function is stored in `out`.

"""
function update_bellman!(cp::CareerWorkerProblem,
                         v::Array,
                         out::Array;
                         ret_policy::Bool=false)

    # new life. This is a function of the distribution parameters and is
    # always constant. No need to recompute it in the loop
    v3 = (cp.G_mean + cp.F_mean + cp.β .*
          cp.F_probs' * v * cp.G_probs)[1]        # don't need 1 element array

    for j=1:cp.N
        for i=1:cp.N
            # stay put
            v1 = cp.θ[i] + cp.ϵ[j] + cp.β * v[i, j]

            # new job
            v2 = (cp.θ[i] .+ cp.G_mean .+ cp.β .*
                  v[i, :]' * cp.G_probs)[1]       # don't need a single element array

            if ret_policy
                if v1 > max(v2, v3)
                    action = 1
                elseif v2 > max(v1, v3)
                    action = 2
                else
                    action = 3
                end
                out[i, j] = action
            else
                out[i, j] = max(v1, v2, v3)
            end
        end
    end
end


function update_bellman(cp::CareerWorkerProblem, v::Array; ret_policy::Bool=false)
    out = similar(v)
    update_bellman!(cp, v, out, ret_policy=ret_policy)
    return out
end

"""
Extract the greedy policy (policy function) of the model.

##### Arguments

- `cp::CareerWorkerProblem` : Instance of `CareerWorkerProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output

##### Returns

None, `out` is updated in place to hold the policy function

"""
function get_greedy!(cp::CareerWorkerProblem, v::Array, out::Array)
    update_bellman!(cp, v, out, ret_policy=true)
end


function get_greedy(cp::CareerWorkerProblem, v::Array)
    update_bellman(cp, v, ret_policy=true)
end
