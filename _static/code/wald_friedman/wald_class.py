class WaldFriedman:
    """
    Insert relevant docstrings here
    """
    def __init__(self, c, L0, L1, f0, f1, m=25):
        self.c = c
        self.L0, self.L1 = L0, L1
        self.m = m
        self.pgrid = np.linspace(0.0, 1.0, m)

        # Renormalize distributions so nothing is "too" small
        f0 = np.clip(f0, 1e-8, 1-1e-8)
        f1 = np.clip(f1, 1e-8, 1-1e-8)
        self.f0 = f0 / np.sum(f0)
        self.f1 = f1 / np.sum(f1)
        self.J = np.zeros(m)

    def current_distribution(self, p):
        """
        This function takes a value for the probability with which
        the correct model is model 0 and returns the mixed
        distribution that corresponds with that belief.
        """
        return p*self.f0 + (1-p)*self.f1

    def bayes_update_k(self, p, k):
        """
        This function takes a value for p, and a realization of the
        random variable and calculates the value for p tomorrow.
        """
        f0_k = self.f0[k]
        f1_k = self.f1[k]

        p_tp1 = p * f0_k / (p * f0_k + (1 - p) * f1_k)

        return np.clip(p_tp1, 0, 1)

    def bayes_update_all(self, p):
        """
        This is similar to `bayes_update_k` except it returns a
        new value for p for each realization of the random variable
        """
        return np.clip(p * self.f0 / (p * self.f0 + (1 - p) * self.f1), 0, 1)

    def payoff_choose_f0(self, p):
        "For a given probability specify the cost of accepting model 0"
        return (1 - p) * self.L0

    def payoff_choose_f1(self, p):
        "For a given probability specify the cost of accepting model 1"
        return p * self.L1

    def EJ(self, p, J):
        """
        This function evaluates the expectation of the value function
        at period t+1. It does so by taking the current probability
        distribution over outcomes:

            p(z_{k+1}) = p_k f_0(z_{k+1}) + (1-p_k) f_1(z_{k+1})

        and evaluating the value function at the possible states
        tomorrow J(p_{t+1}) where

            p_{t+1} = p f0 / ( p f0 + (1-p) f1)

        Parameters
        ----------
        p : Scalar(Float64)
            The current believed probability that model 0 is the true
            model.
        J : Function
            The current value function for a decision to continue

        Returns
        -------
        EJ : Scalar(Float64)
            The expected value of the value function tomorrow
        """
        # Pull out information
        f0, f1 = self.f0, self.f1

        # Get the current believed distribution and tomorrows possible dists
        # Need to clip to make sure things don't blow up (go to infinity)
        curr_dist = self.current_distribution(p)
        tp1_dist = self.bayes_update_all(p)

        # Evaluate the expectation
        EJ = curr_dist @ J(tp1_dist)

        return EJ

    def payoff_continue(self, p, J):
        """
        For a given probability distribution and value function give
        cost of continuing the search for correct model
        """
        return self.c + self.EJ(p, J)

    def bellman_operator(self, J):
        """
        Evaluates the value function for a given continuation value
        function; that is, evaluates

            J(p) = min(pL0, (1-p)L1, c + E[J(p')])

        Uses linear interpolation between points
        """
        payoff_choose_f0 = self.payoff_choose_f0
        payoff_choose_f1 = self.payoff_choose_f1
        payoff_continue = self.payoff_continue
        c, L0, L1, f0, f1 = self.c, self.L0, self.L1, self.f0, self.f1
        m, pgrid = self.m, self.pgrid

        J_out = np.empty(m)
        J_interp = interp.UnivariateSpline(pgrid, J, k=1, ext=0)

        for (p_ind, p) in enumerate(pgrid):
            # Payoff of choosing model 0
            p_c_0 = payoff_choose_f0(p)
            p_c_1 = payoff_choose_f1(p)
            p_con = payoff_continue(p, J_interp)

            J_out[p_ind] = min(p_c_0, p_c_1, p_con)

        return J_out

    def solve_model(self):
        J = qe.compute_fixed_point(self.bellman_operator, np.zeros(self.m),
                                   error_tol=1e-7, verbose=False)

        self.J = J
        return J

    def find_cutoff_rule(self, J):
        """
        This function takes a value function and returns the corresponding
        cutoffs of where you transition between continue and choosing a
        specific model
        """
        payoff_choose_f0 = self.payoff_choose_f0
        payoff_choose_f1 = self.payoff_choose_f1
        m, pgrid = self.m, self.pgrid

        # Evaluate cost at all points on grid for choosing a model
        p_c_0 = payoff_choose_f0(pgrid)
        p_c_1 = payoff_choose_f1(pgrid)

        # The cutoff points can be found by differencing these costs with
        # the Bellman equation (J is always less than or equal to p_c_i)
        lb = pgrid[np.searchsorted(p_c_1 - J, 1e-10) - 1]
        ub = pgrid[np.searchsorted(J - p_c_0, -1e-10)]

        return (lb, ub)

    def simulate(self, f, p0=0.5):
        """
        This function takes an initial condition and simulates until it
        stops (when a decision is made).
        """
        # Check whether vf is computed
        if np.sum(self.J) < 1e-8:
            self.solve_model()

        # Unpack useful info
        lb, ub = self.find_cutoff_rule(self.J)
        update_p = self.bayes_update_k
        curr_dist = self.current_distribution

        # Initialize a couple useful variables
        decision_made = False
        p = p0
        t = 0

        while decision_made is False:
            # Maybe should specify which distribution is correct one so that
            # the draws come from the "right" distribution
            k = int(qe.random.draw(np.cumsum(f)))
            t = t+1
            p = update_p(p, k)
            if p < lb:
                decision_made = True
                decision = 1
            elif p > ub:
                decision_made = True
                decision = 0

        return decision, p, t

    def simulate_tdgp_f0(self, p0=0.5):
        """
        Uses the distribution f0 as the true data generating
        process
        """
        decision, p, t = self.simulate(self.f0, p0)

        if decision == 0:
            correct = True
        else:
            correct = False

        return correct, p, t

    def simulate_tdgp_f1(self, p0=0.5):
        """
        Uses the distribution f1 as the true data generating
        process
        """
        decision, p, t = self.simulate(self.f1, p0)

        if decision == 1:
            correct = True
        else:
            correct = False

        return correct, p, t

    def stopping_dist(self, ndraws=250, tdgp="f0"):
        """
        Simulates repeatedly to get distributions of time needed to make a
        decision and how often they are correct.
        """
        if tdgp == "f0":
            simfunc = self.simulate_tdgp_f0
        else:
            simfunc = self.simulate_tdgp_f1

        # Allocate space
        tdist = np.empty(ndraws, int)
        cdist = np.empty(ndraws, bool)

        for i in range(ndraws):
            correct, p, t = simfunc()
            tdist[i] = t
            cdist[i] = correct

        return cdist, tdist
