"""
Function to Solve Optimistic Beliefs
"""

function price_optimistic_beliefs(transitions::Vector, dividend_payoff::Vector;
                                  β::AbstractFloat=.75, max_iter::Integer=50000,
                                  tol::AbstractFloat=1e-16)

    # We will guess an initial price vector of [0, 0]
    p_new = [0,0]
    p_old = [10.0,10.0]

    # We know this is a contraction mapping, so we can iterate to conv
    for i in 1:max_iter
        p_old = p_new
        temp = [maximum((q * p_old) + (q * dividend_payoff))
                               for q in transitions]
        p_new = β * temp

        # If we succed in converging, break out of for loop
        if maximum(sqrt, ((p_new - p_old).^2)) < 1e-12
            break
        end
    end

    temp=[minimum((q * p_old) + (q * dividend_payoff)) for q in transitions]
    ptwiddle = β * temp

    phat_a = [p_new[1], ptwiddle[2]]
    phat_b = [ptwiddle[1], p_new[2]]

    return p_new, phat_a, phat_b
end
