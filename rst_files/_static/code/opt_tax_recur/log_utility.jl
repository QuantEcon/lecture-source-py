function log_utility(;β = 0.9,
                      ψ = 0.69,
                      Π = 0.5 * ones(2, 2),
                      G = [0.1, 0.2],
                      Θ = ones(2),
                      transfers = false)
    # Derivatives of utility function
    U(c,n) = log(c) + ψ * log(1 - n)
    Uc(c,n) = 1 ./ c
    Ucc(c,n) = -c.^(-2.0)
    Un(c,n) = -ψ ./ (1.0 - n)
    Unn(c,n) = -ψ ./ (1.0 - n).^2.0
    n_less_than_one = true
    return Model(β, Π, G, Θ, transfers,
                 U, Uc, Ucc, Un, Unn, n_less_than_one)
end