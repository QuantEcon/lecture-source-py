#=

Authors: Shunsuke Hori

=#
"""
Function to Solve Single Beliefs
"""
function price_single_beliefs(transition::Matrix, dividend_payoff::Vector;
                              β::AbstractFloat=.75)
    # First compute inverse piece
    imbq_inv = inv(eye(size(transition, 1)) - β * transition)

    # Next compute prices
    prices = β * ((imbq_inv * transition) * dividend_payoff)

    return prices
end
