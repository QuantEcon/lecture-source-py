using QuantEcon: tauchen, MarkovChain, simulate


# ------------------------------------------------------------------- #
# Define the main Arellano Economy type
# ------------------------------------------------------------------- #

"""
Arellano 2008 deals with a small open economy whose government
invests in foreign assets in order to smooth the consumption of
domestic households. Domestic households receive a stochastic
path of income.

##### Fields
* `β::AbstractFloat`: Time discounting parameter
* `γ::AbstractFloat`: Risk aversion parameter
* `r::AbstractFloat`: World interest rate
* `ρ::AbstractFloat`: Autoregressive coefficient on income process
* `η::AbstractFloat`: Standard deviation of noise in income process
* `θ::AbstractFloat`: Probability of re-entering the world financial sector after default
* `ny::Integer`: Number of points to use in approximation of income process
* `nB::Integer`: Number of points to use in approximation of asset holdings
* `ygrid::Vector{AbstractFloat}`: This is the grid used to approximate income process
* `ydefgrid::Vector{AbstractFloat}`: When in default get less income than process
  would otherwise dictate
* `Bgrid::Vector{AbstractFloat}`: This is grid used to approximate choices of asset
  holdings
* `Π::Matrix{AbstractFloat}`: Transition probabilities between income levels
* `vf::Matrix{AbstractFloat}`: Place to hold value function
* `vd::Matrix{AbstractFloat}`: Place to hold value function when in default
* `vc::Matrix{AbstractFloat}`: Place to hold value function when choosing to
  continue
* `policy::Matrix{AbstractFloat}`: Place to hold asset policy function
* `q::Matrix{AbstractFloat}`: Place to hold prices at different pairs of (y, B')
* `defprob::Matrix{AbstractFloat}`: Place to hold the default probabilities for
  pairs of (y, B')
"""
struct ArellanoEconomy{TF<:AbstractFloat, TI<:Integer}
    # Model Parameters
    β::TF
    γ::TF
    r::TF
    ρ::TF
    η::TF
    θ::TF

    # Grid Parameters
    ny::TI
    nB::TI
    ygrid::Vector{TF}
    ydefgrid::Vector{TF}
    Bgrid::Vector{TF}
    Π::Matrix{TF}

    # Value function
    vf::Matrix{TF}
    vd::Matrix{TF}
    vc::Matrix{TF}
    policy::Matrix{TF}
    q::Matrix{TF}
    defprob::Matrix{TF}
end

"""
This is the default constructor for building an economy as presented
in Arellano 2008.

##### Arguments
* `;β::AbstractFloat(0.953)`: Time discounting parameter
* `;γ::AbstractFloat(2.0)`: Risk aversion parameter
* `;r::AbstractFloat(0.017)`: World interest rate
* `;ρ::AbstractFloat(0.945)`: Autoregressive coefficient on income process
* `;η::AbstractFloat(0.025)`: Standard deviation of noise in income process
* `;θ::AbstractFloat(0.282)`: Probability of re-entering the world financial sector
  after default
* `;ny::Integer(21)`: Number of points to use in approximation of income process
* `;nB::Integer(251)`: Number of points to use in approximation of asset holdings
"""
function ArellanoEconomy{TF<:AbstractFloat}(;β::TF=.953, γ::TF=2., r::TF=0.017, 
                          ρ::TF=0.945, η::TF=0.025, θ::TF=0.282,
                          ny::Integer=21, nB::Integer=251)

    # Create grids
    Bgrid = collect(linspace(-.4, .4, nB))
    mc = tauchen(ny, ρ, η)
    Π = mc.p
    ygrid = exp.(mc.state_values)
    ydefgrid = min.(.969 * mean(ygrid), ygrid)

    # Define value functions (Notice ordered different than Python to take
    # advantage of column major layout of Julia)
    vf = zeros(nB, ny)
    vd = zeros(1, ny)
    vc = zeros(nB, ny)
    policy = Array{TF}(nB, ny)
    q = ones(nB, ny) .* (1 / (1 + r))
    defprob = Array{TF}(nB, ny)

    return ArellanoEconomy(β, γ, r, ρ, η, θ, ny, nB, ygrid, ydefgrid, Bgrid, Π,
                            vf, vd, vc, policy, q, defprob)
end

u(ae::ArellanoEconomy, c) = c^(1 - ae.γ) / (1 - ae.γ)
_unpack(ae::ArellanoEconomy) =
    ae.β, ae.γ, ae.r, ae.ρ, ae.η, ae.θ, ae.ny, ae.nB
_unpackgrids(ae::ArellanoEconomy) =
    ae.ygrid, ae.ydefgrid, ae.Bgrid, ae.Π, ae.vf, ae.vd, ae.vc, ae.policy, ae.q, ae.defprob


# ------------------------------------------------------------------- #
# Write the value function iteration
# ------------------------------------------------------------------- #
"""
This function performs the one step update of the value function for the
Arellano model-- Using current value functions and their expected value,
it updates the value function at every state by solving for the optimal
choice of savings

##### Arguments

* `ae::ArellanoEconomy`: This is the economy we would like to update the
  value functions for
* `EV::Matrix{TF}`: Expected value function at each state
* `EVd::Matrix{TF}`: Expected value function of default at each state
* `EVc::Matrix{TF}`: Expected value function of continuing at each state

##### Notes

* This function updates value functions and policy functions in place.
"""
function one_step_update!{TF<:AbstractFloat}(ae::ArellanoEconomy, EV::Matrix{TF},
                          EVd::Matrix{TF}, EVc::Matrix{TF})

    # Unpack stuff
    β, γ, r, ρ, η, θ, ny, nB = _unpack(ae)
    ygrid, ydefgrid, Bgrid, Π, vf, vd, vc, policy, q, defprob = _unpackgrids(ae)
    zero_ind = searchsortedfirst(Bgrid, 0.)

    for iy=1:ny
        y = ae.ygrid[iy]
        ydef = ae.ydefgrid[iy]

        # Value of being in default with income y
        defval = u(ae, ydef) + β*(θ*EVc[zero_ind, iy] + (1-θ)*EVd[1, iy])
        ae.vd[1, iy] = defval

        for ib=1:nB
            B = ae.Bgrid[ib]

            current_max = -1e14
            pol_ind = 0
            for ib_next=1:nB
                c = max(y - ae.q[ib_next, iy]*Bgrid[ib_next] + B, 1e-14)
                m = u(ae, c) + β * EV[ib_next, iy]

                if m > current_max
                    current_max = m
                    pol_ind = ib_next
                end

            end

            # Update value and policy functions
            ae.vc[ib, iy] = current_max
            ae.policy[ib, iy] = pol_ind
            ae.vf[ib, iy] = defval > current_max ? defval: current_max
        end
    end

    Void
end

"""
This function takes the Arellano economy and its value functions and
policy functions and then updates the prices for each (y, B') pair

##### Arguments

* `ae::ArellanoEconomy`: This is the economy we would like to update the
  prices for

##### Notes

* This function updates the prices and default probabilities in place
"""
function compute_prices!(ae::ArellanoEconomy)
    # Unpack parameters
    β, γ, r, ρ, η, θ, ny, nB = _unpack(ae)

    # Create default values with a matching size
    vd_compat = repmat(ae.vd, nB)
    default_states = vd_compat .> ae.vc

    # Update default probabilities and prices
    copy!(ae.defprob, default_states * ae.Π')
    copy!(ae.q, (1 - ae.defprob) / (1 + r))

    Void
end

"""
This performs value function iteration and stores all of the data inside
the ArellanoEconomy type.

##### Arguments

* `ae::ArellanoEconomy`: This is the economy we would like to solve
* `;tol::Float64(1e-8)`: Level of tolerance we would like to achieve
* `;maxit::Int(10000)`: Maximum number of iterations

##### Notes

* This updates all value functions, policy functions, and prices in place.

"""
function vfi!(ae::ArellanoEconomy; tol=1e-8, maxit=10000)

    # Unpack stuff
    β, γ, r, ρ, η, θ, ny, nB = _unpack(ae)
    ygrid, ydefgrid, Bgrid, Π, vf, vd, vc, policy, q, defprob = _unpackgrids(ae)
    Πt = Π'

    # Iteration stuff
    it = 0
    dist = 10.

    # Allocate memory for update
    V_upd = zeros(ae.vf)

    while dist > tol && it < maxit
        it += 1

        # Compute expectations for this iterations
        # (We need Π' because of order value function dimensions)
        copy!(V_upd, ae.vf)
        EV = ae.vf * Πt
        EVd = ae.vd * Πt
        EVc = ae.vc * Πt

        # Update Value Function
        one_step_update!(ae, EV, EVd, EVc)

        # Update prices
        compute_prices!(ae)

        dist = maximum(abs, V_upd - ae.vf)

        if it%25 == 0
            println("Finished iteration $(it) with dist of $(dist)")
        end
    end

    Void
end

"""
This function simulates the Arellano economy

##### Arguments

* `ae::ArellanoEconomy`: This is the economy we would like to solve
* `capT::Integer`: Number of periods to simulate
* `;y_init::AbstratFloat(mean(ae.ygrid)`: The level of income we would like to
  start with
* `;B_init::AbstratFloat(mean(ae.Bgrid)`: The level of asset holdings we would like
  to start with

##### Returns

* `B_sim_val::Vector{TI}`: Simulated values of assets
* `y_sim_val::Vector{TF}`: Simulated values of income
* `q_sim_val::Vector{TF}`: Simulated values of prices
* `default_status::Vector{Bool}`: Simulated default status
  (true if in default)

##### Notes

* This updates all value functions, policy functions, and prices in place.

"""
function QuantEcon.simulate{TI<:Integer, TF<:AbstractFloat}(ae::ArellanoEconomy, 
                            capT::TI=5000;
                            y_init::TF=mean(ae.ygrid), 
                            B_init::TF=mean(ae.Bgrid))

    # Get initial indices
    zero_index = searchsortedfirst(ae.Bgrid, 0.)
    y_init_ind = searchsortedfirst(ae.ygrid, y_init)
    B_init_ind = searchsortedfirst(ae.Bgrid, B_init)

    # Create a QE MarkovChain
    mc = MarkovChain(ae.Π)
    y_sim_indices = simulate(mc, capT+1; init=y_init_ind)

    # Allocate and Fill output
    y_sim_val = Vector{TF}(capT+1)
    B_sim_val, q_sim_val = similar(y_sim_val), similar(y_sim_val)
    B_sim_indices = Vector{TI}(capT+1)
    default_status = fill(false, capT+1)
    B_sim_indices[1], default_status[1] = B_init_ind, false
    y_sim_val[1], B_sim_val[1] = ae.ygrid[y_init_ind], ae.Bgrid[B_init_ind]

    for t=1:capT
        # Get today's indexes
        yi, Bi = y_sim_indices[t], B_sim_indices[t]
        defstat = default_status[t]

        # If you are not in default
        if !defstat
            default_today = ae.vc[Bi, yi] < ae.vd[yi] ? true: false

            if default_today
                # Default values
                default_status[t] = true
                default_status[t+1] = true
                y_sim_val[t] = ae.ydefgrid[y_sim_indices[t]]
                B_sim_indices[t+1] = zero_index
                B_sim_val[t+1] = 0.
                q_sim_val[t] = ae.q[zero_index, y_sim_indices[t]]
            else
                default_status[t] = false
                y_sim_val[t] = ae.ygrid[y_sim_indices[t]]
                B_sim_indices[t+1] = ae.policy[Bi, yi]
                B_sim_val[t+1] = ae.Bgrid[B_sim_indices[t+1]]
                q_sim_val[t] = ae.q[B_sim_indices[t+1], y_sim_indices[t]]
            end

        # If you are in default
        else
            B_sim_indices[t+1] = zero_index
            B_sim_val[t+1] = 0.
            y_sim_val[t] = ae.ydefgrid[y_sim_indices[t]]
            q_sim_val[t] = ae.q[zero_index, y_sim_indices[t]]

            # With probability θ exit default status
            if rand() < ae.θ
                default_status[t+1] = false
            else
                default_status[t+1] = true
            end
        end
    end

    return (y_sim_val[1:capT], B_sim_val[1:capT], q_sim_val[1:capT],
            default_status[1:capT])
end
