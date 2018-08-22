def price_optimistic_beliefs(transitions, dividend_payoff, β=.75,
                             max_iter=50000, tol=1e-16):
    """
    Function to Solve Optimistic Beliefs
    """
    # We will guess an initial price vector of [0, 0]
    p_new = np.array([[0], [0]])
    p_old = np.array([[10.], [10.]])

    # We know this is a contraction mapping, so we can iterate to conv
    for i in range(max_iter):
        p_old = p_new
        p_new = β * np.max([q @ p_old + q @ dividend_payoff for q in transitions], 1)

        # If we succed in converging, break out of for loop
        if np.max(np.sqrt((p_new - p_old)**2)) < 1e-12:
            break

    ptwiddle = β * np.min([q @ p_old + q @ dividend_payoff for q in transitions], 1)

    phat_a = np.array([p_new[0], ptwiddle[1]])
    phat_b = np.array([ptwiddle[0], p_new[1]])

    return p_new, phat_a, phat_b
