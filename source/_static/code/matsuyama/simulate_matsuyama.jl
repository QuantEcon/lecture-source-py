#=

Author: Shunsuke Hori

=#

using PyPlot

"""
If we expand the implicit function for h_j(n_k) then we find that
it is a quadratic. We know that h_j(n_k) > 0 so we can get its
value by using the quadratic form
"""

function h_j(j::Integer, nk::Real, s1::Real, s2::Real,
             θ::Real, δ::Real, ρ::Real)
    # Find out who's h we are evaluating
    if j == 1
        sj = s1
        sk = s2
    else
        sj = s2
        sk = s1
    end

    # Coefficients on the quadratic a x^2 + b x + c = 0
    a = 1.0
    b = ((ρ + 1 / ρ) * nk - sj - sk)
    c = (nk * nk - (sj * nk) / ρ - sk * ρ * nk)

    # Positive solution of quadratic form
    root = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)

    return root
end

"""
Determine whether (n1, n2) is in the set DLL
"""
DLL(n1::Real, n2::Real,
    s1_ρ::Real, s2_ρ::Real,
    s1::Real, s2::Real,
    θ::Real, δ::Real, ρ::Real) =
        (n1 <= s1_ρ) && (n2 <= s2_ρ)

"""
Determine whether (n1, n2) is in the set DHH
"""
DHH(n1::Real, n2::Real,
    s1_ρ::Real, s2_ρ::Real,
    s1::Real, s2::Real,
    θ::Real, δ::Real, ρ::Real) =
        (n1 >= h_j(1, n2, s1, s2, θ, δ, ρ)) && (n2 >= h_j(2, n1, s1, s2, θ, δ, ρ))

"""
Determine whether (n1, n2) is in the set DHL
"""
DHL(n1::Real, n2::Real,
    s1_ρ::Real, s2_ρ::Real,
    s1::Real, s2::Real,
    θ::Real, δ::Real, ρ::Real) =
        (n1 >= s1_ρ) && (n2 <= h_j(2, n1, s1, s2, θ, δ, ρ))

"""
Determine whether (n1, n2) is in the set DLH
"""
DLH(n1::Real, n2::Real,
    s1_ρ::Real, s2_ρ::Real,
    s1::Real, s2::Real,
    θ::Real, δ::Real, ρ::Real) =
        (n1 <= h_j(1, n2, s1, s2, θ, δ, ρ)) && (n2 >= s2_ρ)

"""
Takes a current value for (n_{1, t}, n_{2, t}) and returns the
values (n_{1, t+1}, n_{2, t+1}) according to the law of motion.
"""
function one_step(n1::Real, n2::Real,
                  s1_ρ::Real, s2_ρ::Real,
                  s1::Real, s2::Real,
                  θ::Real, δ::Real, ρ::Real)
    # Depending on where we are, evaluate the right branch
    if DLL(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)
        n1_tp1 = δ * (θ * s1_ρ + (1 - θ) * n1)
        n2_tp1 = δ * (θ * s2_ρ + (1 - θ) * n2)
    elseif DHH(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)
        n1_tp1 = δ * n1
        n2_tp1 = δ * n2
    elseif DHL(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)
        n1_tp1 = δ * n1
        n2_tp1 = δ * (θ * h_j(2, n1, s1, s2, θ, δ, ρ) + (1 - θ) * n2)
    elseif DLH(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)
        n1_tp1 = δ * (θ * h_j(1, n2, s1, s2, θ, δ, ρ) + (1 - θ) * n1)
        n2_tp1 = δ * n2
    end

    return n1_tp1, n2_tp1
end

"""
Given an initial condition, continues to yield new values of `n1` and `n2`
"""
new_n1n2(n1_0::Real, n2_0::Real,
         s1_ρ::Real, s2_ρ::Real,
         s1::Real, s2::Real,
         θ::Real, δ::Real, ρ::Real) =
            one_step(n1_0, n2_0, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)

"""
Takes initial values and iterates forward to see whether
the histories eventually end up in sync.

If countries are symmetric then as soon as the two countries have the
same measure of firms then they will by synchronized -- However, if
they are not symmetric then it is possible they have the same measure
of firms but are not yet synchronized. To address this, we check whether
firms stay synchronized for `npers` periods with Euclidean norm

##### Parameters
----------
- `n1_0` : `Real`,
Initial normalized measure of firms in country one
- `n2_0` : `Real`,
Initial normalized measure of firms in country two
- `maxiter` : `Integer`,
Maximum number of periods to simulate
- `npers` : `Integer`,
Number of periods we would like the countries to have the same measure for

##### Returns
-------
- `synchronized` : `Bool`,
Did they two economies end up synchronized
- `pers_2_sync` : `Integer`,
The number of periods required until they synchronized
"""
function pers_till_sync(n1_0::Real, n2_0::Real,
                        s1_ρ::Real, s2_ρ::Real,
                        s1::Real, s2::Real,
                        θ::Real, δ::Real, ρ::Real,
                        maxiter::Integer, npers::Integer)

    # Initialize the status of synchronization
    synchronized = false
    pers_2_sync = maxiter
    iters = 0

    nsync = 0

    while (~synchronized) && (iters < maxiter)
        # Increment the number of iterations and get next values
        iters += 1

        n1_t, n2_t = new_n1n2(n1_0, n2_0, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)

        # Check whether same in this period
        if abs(n1_t - n2_t) < 1e-8
            nsync += 1
        # If not, then reset the nsync counter
        else
            nsync = 0
        end

        # If we have been in sync for npers then stop and countries
        # became synchronized nsync periods ago
        if nsync > npers
            synchronized = true
            pers_2_sync = iters - nsync
        end
        n1_0, n2_0 = n1_t, n2_t
    end
    return synchronized, pers_2_sync
end

function create_attraction_basis{TR <: Real}(s1_ρ::TR, s2_ρ::TR,
                                 s1::TR, s2::TR, θ::TR, δ::TR, ρ::TR,
                                 maxiter::Integer, npers::Integer, npts::Integer)
    # Create unit range with npts
    synchronized, pers_2_sync = false, 0
    unit_range = linspace(0.0, 1.0, npts)

    # Allocate space to store time to sync
    time_2_sync = Matrix{TR}(npts, npts)
    # Iterate over initial conditions
    for (i, n1_0) in enumerate(unit_range)
        for (j, n2_0) in enumerate(unit_range)
            synchronized, pers_2_sync = pers_till_sync(n1_0, n2_0, s1_ρ, s2_ρ,
                                                        s1, s2, θ, δ, ρ,
                                                        maxiter, npers)
            time_2_sync[i, j] = pers_2_sync
        end
    end

    return time_2_sync
end


# == Now we define a type for the model == #

"""
The paper "Globalization and Synchronization of Innovation Cycles" presents
a two country model with endogenous innovation cycles. Combines elements
from Deneckere Judd (1985) and Helpman Krugman (1985) to allow for a
model with trade that has firms who can introduce new varieties into
the economy.

We focus on being able to determine whether two countries eventually
synchronize their innovation cycles. To do this, we only need a few
of the many parameters. In particular, we need the parameters listed
below

##### Parameters
----------
- `s1` : `Real`,
Amount of total labor in country 1 relative to total worldwide labor
- `θ` : `Real`,
A measure of how mcuh more of the competitive variety is used in
production of final goods
- `δ` : `Real`,
Percentage of firms that are not exogenously destroyed every period
- `ρ` : `Real`,
Measure of how expensive it is to trade between countries
"""
struct MSGSync{TR <: Real}
    s1::TR
    s2::TR
    s1_ρ::TR
    s2_ρ::TR
    θ::TR
    δ::TR
    ρ::TR
end

function MSGSync(s1::Real=0.5, θ::Real=2.5,
                 δ::Real=0.7, ρ::Real=0.2)
    # Store other cutoffs and parameters we use
    s2 = 1 - s1
    s1_ρ = min((s1 - ρ * s2) / (1 - ρ), 1)
    s2_ρ = 1 - s1_ρ

    model=MSGSync(s1, s2, s1_ρ, s2_ρ, θ, δ, ρ)

    return model
end

unpack_params(model::MSGSync) =
    model.s1, model.s2, model.θ, model.δ, model.ρ, model.s1_ρ, model.s2_ρ

"""
Simulates the values of `n1` and `n2` for `T` periods

##### Parameters
----------
- `n1_0` : `Real`, Initial normalized measure of firms in country one
- `n2_0` : `Real`, Initial normalized measure of firms in country two
- `T` : `Integer`, Number of periods to simulate

##### Returns
-------
- `n1` : `Vector{TR}(ndim=1) where TR <: Real`,
A history of normalized measures of firms in country one
- `n2` : `Vector{TR}(ndim=1) where TR <: Real`,
A history of normalized measures of firms in country two
"""
function simulate_n{TR <: Real}(model::MSGSync, n1_0::TR, n2_0::TR, T::Integer)
    # Unpack parameters
    s1, s2, θ, δ, ρ, s1_ρ, s2_ρ = unpack_params(model)

    # Allocate space
    n1 = Vector{TR}(T)
    n2 = Vector{TR}(T)

    # Simulate for T periods
    for t in 1:T
        # Get next values
        n1[t], n2[t] = n1_0, n2_0
        n1_0, n2_0 = new_n1n2(n1_0, n2_0, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)
    end

    return n1, n2
end

"""
Takes initial values and iterates forward to see whether
the histories eventually end up in sync.

If countries are symmetric then as soon as the two countries have the
same measure of firms then they will by synchronized -- However, if
they are not symmetric then it is possible they have the same measure
of firms but are not yet synchronized. To address this, we check whether
firms stay synchronized for `npers` periods with Euclidean norm

##### Parameters
----------
- `n1_0` : `Real`,
Initial normalized measure of firms in country one
- `n2_0` : `Real`,
Initial normalized measure of firms in country two
- `maxiter` : `Integer`,
Maximum number of periods to simulate
- `npers` : `Integer`,
Number of periods we would like the countries to have the same measure for

##### Returns
-------
- `synchronized` : `Bool`,
Did they two economies end up synchronized
- `pers_2_sync` : `Integer`,
The number of periods required until they synchronized
"""
function pers_till_sync(model::MSGSync, n1_0::Real, n2_0::Real,
                        maxiter::Integer=500, npers::Integer=3)
    # Unpack parameters
    s1, s2, θ, δ, ρ, s1_ρ, s2_ρ = unpack_params(model)

    return pers_till_sync(n1_0, n2_0, s1_ρ, s2_ρ, s1, s2,
                          θ, δ, ρ, maxiter, npers)
end

"""
Creates an attraction basis for values of n on [0, 1] X [0, 1] with npts in each dimension
"""
function create_attraction_basis(model::MSGSync;
                                 maxiter::Integer=250,
                                 npers::Integer=3,
                                 npts::Integer=50)
    # Unpack parameters
    s1, s2, θ, δ, ρ, s1_ρ, s2_ρ = unpack_params(model)

    ab = create_attraction_basis(s1_ρ, s2_ρ, s1, s2, θ, δ,
                                  ρ, maxiter, npers, npts)
    return ab
end
