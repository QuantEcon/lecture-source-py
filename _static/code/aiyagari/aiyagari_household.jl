using QuantEcon

"""
Stores all the parameters that define the household's
problem.

##### Fields

- `r::Real` : interest rate
- `w::Real` : wage
- `σ::Real` : risk aversion
- `β::AbstractFloat` : discount factor
- `z_chain::MarkovChain` : MarkovChain for income
- `a_min::Real` : minimum on asset grid
- `a_max::Real` : maximum on asset grid
- `a_size::Integer` : number of points on asset grid
- `z_size::Integer` : number of points on income grid
- `n::Integer` : number of points in state space: (a, z)
- `s_vals::Array{TF} where TF<:AbstractFloat` : stores all the possible (a, z) combinations
- `s_i_vals::Array{TI} where TI<:Integer` : stores indices of all the possible (a, z) combinations
- `R::Array{TF} where TF<:AbstractFloat` : reward array
- `Q::Array{TF} where TF<:AbstractFloat` : transition probability array
- `u::Function` : utility function
"""
mutable struct Household{TR<:Real, TF<:AbstractFloat, TI<:Integer}
    r::TR
    w::TR
    σ::TR
    β::TF
    z_chain::MarkovChain{TF, Array{TF, 2}, Array{TF, 1}}
    a_min::TR
    a_max::TR
    a_size::TI
    a_vals::AbstractVector{TF}
    z_size::TI
    n::TI
    s_vals::Array{TF}
    s_i_vals::Array{TI}
    R::Array{TR}
    Q::Array{TR}
    u::Function
end

"""
Constructor for `Household`

##### Arguments
- `r::Real(0.01)` : interest rate
- `w::Real(1.0)` : wage
- `β::AbstractFloat(0.96)` : discount factor
- `z_chain::MarkovChain` : MarkovChain for income
- `a_min::Real(1e-10)` : minimum on asset grid
- `a_max::Real(18.0)` : maximum on asset grid
- `a_size::TI(200)` : number of points on asset grid

"""
function Household{TF<:AbstractFloat}(;
                    r::Real=0.01,
                    w::Real=1.0,
                    σ::Real=1.0,
                    β::TF=0.96,
                    z_chain::MarkovChain{TF,Array{TF,2},Array{TF,1}}
                        =MarkovChain([0.9 0.1; 0.1 0.9], [0.1; 1.0]),
                    a_min::Real=1e-10,
                    a_max::Real=18.0,
                    a_size::Integer=200)

    # set up grids
    a_vals = linspace(a_min, a_max, a_size)
    z_size = length(z_chain.state_values)
    n = a_size*z_size
    s_vals = gridmake(a_vals, z_chain.state_values)
    s_i_vals = gridmake(1:a_size, 1:z_size)

    # set up Q
    Q = zeros(TF, n, a_size, n)
    for next_s_i in 1:n
        for a_i in 1:a_size
            for s_i in 1:n
                z_i = s_i_vals[s_i, 2]
                next_z_i = s_i_vals[next_s_i, 2]
                next_a_i = s_i_vals[next_s_i, 1]
                if next_a_i == a_i
                    Q[s_i, a_i, next_s_i] = z_chain.p[z_i, next_z_i]
                end
            end
        end
    end

    if σ == 1   # log utility
        u = x -> log(x)
    else
        u = x -> (x^(1-σ)-1)/(1-σ)
    end

    # placeholder for R
    R = fill(-Inf, n, a_size)
    h = Household(r, w, σ, β, z_chain, a_min, a_max, a_size,
                  a_vals, z_size, n, s_vals, s_i_vals, R, Q, u)

    setup_R!(h, r, w)

    return h

end

"""
Update the reward array of a Household object, given
a new interest rate and wage.

##### Arguments
- `h::Household` : instance of Household type
- `r::Real(0.01)`: interest rate
- `w::Real(1.0)` : wage
"""
function setup_R!(h::Household, r::Real, w::Real)

    # set up R
    R = h.R
    for new_a_i in 1:h.a_size
        a_new = h.a_vals[new_a_i]
        for s_i in 1:h.n
            a = h.s_vals[s_i, 1]
            z = h.s_vals[s_i, 2]
            c = w * z + (1 + r) * a - a_new
            if c > 0
                R[s_i, new_a_i] = h.u(c)
            end
        end
    end

    h.r = r
    h.w = w
    h.R = R
    h

end
