#=

@author : Victoria Gregory, John Stachurski

=#


struct LakeModel{TF <: AbstractFloat}
    λ::TF
    α::TF
    b::TF
    d::TF
    g::TF
    A::Matrix{TF}
    A_hat::Matrix{TF}
end

"""
Constructor with default values for `LakeModel`

##### Fields of `LakeModel`

  - λ : job finding rate
  - α : dismissal rate
  - b : entry rate into labor force
  - d : exit rate from labor force
  - g : net entry rate
  - A : updates stock
  - A_hat : updates rate

"""
function LakeModel(;λ::AbstractFloat=0.283,
                    α::AbstractFloat=0.013,
                    b::AbstractFloat=0.0124,
                    d::AbstractFloat=0.00822)

    g = b - d
    A = [(1-λ) * (1-d) + b  (1-d) * α + b;
         (1-d) * λ          (1-d) * (1-α)]
    A_hat = A ./ (1 + g)

    return LakeModel(λ, α, b, d, g, A, A_hat)
end

"""
Finds the steady state of the system :math:`x_{t+1} = \hat A x_{t}`

##### Arguments

  - lm : instance of `LakeModel`
  - tol: convergence tolerance

##### Returns

  - x : steady state vector of employment and unemployment rates

"""

function rate_steady_state(lm::LakeModel, tol::AbstractFloat=1e-6)
    x = 0.5 * ones(2)
    error = tol + 1
    while (error > tol)
        new_x = lm.A_hat * x
        error = maximum(abs, new_x - x)
        x = new_x
    end
    return x
end

"""
Simulates the the sequence of Employment and Unemployent stocks

##### Arguments

  - X0 : contains initial values (E0, U0)
  - T : number of periods to simulate

##### Returns

  - X_path : contains sequence of employment and unemployment stocks

"""

function simulate_stock_path{TF<:AbstractFloat}(lm::LakeModel,
                                                X0::AbstractVector{TF},
                                                T::Integer)
    X_path = Array{TF}(2, T)
    X = copy(X0)
    for t in 1:T
        X_path[:, t] = X
        X = lm.A * X
    end
    return X_path
end

"""
Simulates the the sequence of employment and unemployent rates.

##### Arguments

  - X0 : contains initial values (E0, U0)
  - T : number of periods to simulate

##### Returns

  - X_path : contains sequence of employment and unemployment rates

"""
function simulate_rate_path{TF<:AbstractFloat}(lm::LakeModel,
                                               x0::Vector{TF},
                                               T::Integer)
    x_path = Array{TF}(2, T)
    x = copy(x0)
    for t in 1:T
        x_path[:, t] = x
        x = lm.A_hat * x
    end
    return x_path
end
