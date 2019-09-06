.. _odu:

.. include:: /_static/includes/header.raw

.. highlight:: python3

************************************
Job Search III: Search with Learning
************************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install interpolation

Overview
========

In this lecture, we consider an extension of the :doc:`previously studied <mccall_model>` job search model of McCall :cite:`McCall1970`.

In the McCall model, an unemployed worker decides when to accept a permanent position at a specified wage, given

* his or her discount factor

* the level of unemployment compensation

* the distribution from which wage offers are drawn



In the version considered below, the wage distribution is unknown and must be learned.

* The following is based on the presentation in :cite:`Ljungqvist2012`, section 6.6.

Let's start with some imports

.. code-block:: ipython

    from numba import njit, prange, vectorize
    from interpolation import mlinterp, interp
    from math import gamma
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    from matplotlib import cm


Model Features
--------------

* Infinite horizon dynamic programming with two states and one binary control.

* Bayesian updating to learn the unknown distribution.




Model
=====

.. index::
    single: Models; McCall

Let's first review the basic McCall model :cite:`McCall1970` and then add the variation we want to consider.



The Basic McCall Model
----------------------

.. index::
    single: McCall Model

Recall that, :doc:`in the baseline model <mccall_model>`, an unemployed worker is presented in each period with a
permanent job offer at wage :math:`W_t`.

At time :math:`t`, our worker either

#. accepts the offer and works permanently at constant wage :math:`W_t`

#. rejects the offer, receives unemployment compensation :math:`c` and reconsiders next period

The wage sequence :math:`\{W_t\}` is IID and generated from known density :math:`q`.

The worker aims to maximize the expected discounted sum of earnings :math:`\mathbb{E} \sum_{t=0}^{\infty} \beta^t y_t`
The function :math:`V` satisfies the recursion

.. math::
    :label: odu_odu_pv

    v(w)
    = \max \left\{
    \frac{w}{1 - \beta}, \, c + \beta \int v(w')q(w') dw'
    \right\}


The optimal policy has the form :math:`\mathbf{1}\{w \geq \bar w\}`, where
:math:`\bar w` is a constant depending called the *reservation wage*.


Offer Distribution Unknown
--------------------------

Now let's extend the model by considering the variation presented in :cite:`Ljungqvist2012`, section 6.6.

The model is as above, apart from the fact that

* the density :math:`q` is unknown

* the worker learns about :math:`q` by starting with a prior and updating based on wage offers that he/she observes

The worker knows there are two possible distributions :math:`F` and :math:`G` --- with densities :math:`f` and :math:`g`.

At the start of time, "nature" selects :math:`q` to be either :math:`f` or
:math:`g` --- the wage distribution from which the entire sequence :math:`\{W_t\}` will be drawn.

This choice is not observed by the worker, who puts prior probability :math:`\pi_0` on :math:`f` being chosen.

Update rule: worker's time :math:`t` estimate of the distribution is :math:`\pi_t f + (1 - \pi_t) g`, where :math:`\pi_t` updates via

.. math::
    :label: odu_pi_rec

    \pi_{t+1}
    = \frac{\pi_t f(w_{t+1})}{\pi_t f(w_{t+1}) + (1 - \pi_t) g(w_{t+1})}


This last expression follows from Bayes' rule, which tells us that

.. math::

    \mathbb{P}\{q = f \,|\, W = w\}
    = \frac{\mathbb{P}\{W = w \,|\, q = f\}\mathbb{P}\{q = f\}}
    {\mathbb{P}\{W = w\}}
    \quad \text{and} \quad
    \mathbb{P}\{W = w\} = \sum_{\omega \in \{f, g\}} \mathbb{P}\{W = w \,|\, q = \omega\} \mathbb{P}\{q = \omega\}


The fact that :eq:`odu_pi_rec` is recursive allows us to progress to a recursive solution method.


Letting

.. math::

    q_{\pi}(w) := \pi f(w) + (1 - \pi) g(w)
    \quad \text{and} \quad
    \kappa(w, \pi) := \frac{\pi f(w)}{\pi f(w) + (1 - \pi) g(w)}


we can express the value function for the unemployed worker recursively as
follows

.. math::
    :label: odu_mvf

    v(w, \pi)
    = \max \left\{
    \frac{w}{1 - \beta}, \, c + \beta \int v(w', \pi') \, q_{\pi}(w') \, dw'
    \right\}
    \quad \text{where} \quad
    \pi' = \kappa(w', \pi)


Notice that the current guess :math:`\pi` is a state variable, since it affects the worker's perception of probabilities for future rewards.

Parameterization
----------------

Following  section 6.6 of :cite:`Ljungqvist2012`, our baseline parameterization will be

* :math:`f` is :math:`\operatorname{Beta}(1, 1)`

* :math:`g` is :math:`\operatorname{Beta}(3, 1.2)`

* :math:`\beta = 0.95` and :math:`c = 0.3`

The densities :math:`f` and :math:`g` have the following shape



.. code-block:: python3

  def beta_function_factory(a, b):

      @vectorize
      def p(x):
          r = gamma(a + b) / (gamma(a) * gamma(b))
          return r * x**(a-1) * (1 - x)**(b-1)

      return p


  x_grid = np.linspace(0, 1, 100)
  f = beta_function_factory(1, 1)
  g = beta_function_factory(3, 1.2)

  fig, ax = plt.subplots(figsize=(10, 8))
  ax.plot(x_grid, f(x_grid), label='$f$', lw=2)
  ax.plot(x_grid, g(x_grid), label='$g$', lw=2)

  ax.legend()
  plt.show()


.. _looking_forward:

Looking Forward
---------------

What kind of optimal policy might result from :eq:`odu_mvf` and the parameterization specified above?

Intuitively, if we accept at :math:`w_a` and :math:`w_a \leq w_b`, then --- all other things being given --- we should also accept at :math:`w_b`.

This suggests a policy of accepting whenever :math:`w` exceeds some threshold value :math:`\bar w`.

But :math:`\bar w` should depend on :math:`\pi` --- in fact, it should be decreasing in :math:`\pi` because

* :math:`f` is a less attractive offer distribution than :math:`g`
* larger :math:`\pi` means more weight on :math:`f` and less on :math:`g`

Thus larger :math:`\pi` depresses the worker's assessment of her future prospects, and relatively low current offers become more attractive.

**Summary:**  We conjecture that the optimal policy is of the form
:math:`\mathbb 1\{w \geq \bar w(\pi) \}` for some decreasing function
:math:`\bar w`.



Take 1: Solution by VFI
=======================

Let's set about solving the model and see how our results match with our intuition.

We begin by solving via value function iteration (VFI), which is natural but ultimately turns out to be second best.

The class ``SearchProblem`` is used to store parameters and methods needed to compute optimal actions.

.. _odu_vfi_code:

.. code-block:: python3

    class SearchProblem:
        """
        A class to store a given parameterization of the "offer distribution
        unknown" model.

        """

        def __init__(self,
                     β=0.95,            # Discount factor
                     c=0.3,             # Unemployment compensation
                     F_a=1,
                     F_b=1,
                     G_a=3,
                     G_b=1.2,
                     w_max=1,           # Maximum wage possible
                     w_grid_size=100,
                     π_grid_size=100,
                     mc_size=500):

            self.β, self.c, self.w_max = β, c, w_max

            self.f = beta_function_factory(F_a, F_b)
            self.g = beta_function_factory(G_a, G_b)

            self.π_min, self.π_max = 1e-3, 1-1e-3    # Avoids instability
            self.w_grid = np.linspace(0, w_max, w_grid_size)
            self.π_grid = np.linspace(self.π_min, self.π_max, π_grid_size)

            self.mc_size = mc_size

            self.w_f = np.random.beta(F_a, F_b, mc_size)
            self.w_g = np.random.beta(G_a, G_b, mc_size)


The following function takes an instance of this class and returns jitted versions
of the Bellman operator ``T``, and a ``get_greedy()`` function to compute the approximate
optimal policy from a guess ``v`` of the value function

.. code-block:: python3

    def operator_factory(sp, parallel_flag=True):

        f, g = sp.f, sp.g
        w_f, w_g = sp.w_f, sp.w_g
        β, c = sp.β, sp.c
        mc_size = sp.mc_size
        w_grid, π_grid = sp.w_grid, sp.π_grid

        @njit
        def κ(w, π):
            """
            Updates π using Bayes' rule and the current wage observation w.
            """
            pf, pg = π * f(w), (1 - π) * g(w)
            π_new = pf / (pf + pg)

            return π_new

        @njit(parallel=parallel_flag)
        def T(v):
            """
            The Bellman operator.

            """
            v_func = lambda x, y: mlinterp((w_grid, π_grid), v, (x, y))
            v_new = np.empty_like(v)

            for i in prange(len(w_grid)):
                for j in prange(len(π_grid)):
                    w = w_grid[i]
                    π = π_grid[j]

                    v_1 = w / (1 - β)

                    integral_f, integral_g = 0.0, 0.0
                    for m in prange(mc_size):
                        integral_f += v_func(w_f[m], κ(w_f[m], π))
                        integral_g += v_func(w_g[m], κ(w_g[m], π))
                    integral = (π * integral_f + (1 - π) * integral_g) / mc_size

                    v_2 = c + β * integral
                    v_new[i, j] = max(v_1, v_2)

            return v_new

        @njit(parallel=parallel_flag)
        def get_greedy(v):
            """"
            Compute optimal actions taking v as the value function.

            """

            v_func = lambda x, y: mlinterp((w_grid, π_grid), v, (x, y))
            σ = np.empty_like(v)

            for i in prange(len(w_grid)):
                for j in prange(len(π_grid)):
                    w = w_grid[i]
                    π = π_grid[j]

                    v_1 = w / (1 - β)

                    integral_f, integral_g = 0.0, 0.0
                    for m in prange(mc_size):
                        integral_f += v_func(w_f[m], κ(w_f[m], π))
                        integral_g += v_func(w_g[m], κ(w_g[m], π))
                    integral = (π * integral_f + (1 - π) * integral_g) / mc_size

                    v_2 = c + β * integral

                    σ[i, j] = v_1 > v_2  # Evaluates to 1 or 0

            return σ

        return T, get_greedy

We will omit a detailed discussion of the code because there is a
more efficient solution method that we will use later.

To solve the model we will use the following function that iterates using
`T` to find a fixed point

.. code-block:: python3

    def solve_model(sp,
                    use_parallel=True,
                    tol=1e-4,
                    max_iter=1000,
                    verbose=True,
                    print_skip=5):

        """
        Solves for the value function

        * sp is an instance of SearchProblem
        """

        T, _ = operator_factory(sp, use_parallel)

        # Set up loop
        i = 0
        error = tol + 1
        m, n = len(sp.w_grid), len(sp.π_grid)

        # Initialize v
        v = np.zeros((m, n)) + sp.c / (1 - sp.β)

        while i < max_iter and error > tol:
            v_new = T(v)
            error = np.max(np.abs(v - v_new))
            i += 1
            if verbose and i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")
            v = v_new

        if i == max_iter:
            print("Failed to converge!")

        if verbose and i < max_iter:
            print(f"\nConverged in {i} iterations.")


        return v_new

Let's look at solutions computed from value function iteration

.. code-block:: python3

    sp = SearchProblem()
    v_star = solve_model(sp)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(sp.π_grid, sp.w_grid, v_star, 12, alpha=0.6, cmap=cm.jet)
    cs = ax.contour(sp.π_grid, sp.w_grid, v_star, 12, colors="black")
    ax.clabel(cs, inline=1, fontsize=10)
    ax.set(xlabel='$\pi$', ylabel='$w$')

    plt.show()

.. _odu_pol_vfi:

We will also plot the optimal policy

.. code-block:: python3

    T, get_greedy = operator_factory(sp)
    σ_star = get_greedy(v_star)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(sp.π_grid, sp.w_grid, σ_star, 1, alpha=0.6, cmap=cm.jet)
    ax.contour(sp.π_grid, sp.w_grid, σ_star, 1, colors="black")
    ax.set(xlabel='$\pi$', ylabel='$w$')

    ax.text(0.5, 0.6, 'reject')
    ax.text(0.7, 0.9, 'accept')

    plt.show()


The results fit well with our intuition from section :ref:`looking forward <looking_forward>`.

* The black line in the figure above corresponds to the function :math:`\bar w(\pi)` introduced there.

* It is decreasing as expected.


Take 2: A More Efficient Method
===============================

Let's consider another method to solve for the optimal policy.

We will use iteration with an operator that has the same contraction rate as the Bellman operator, but

* one dimensional rather than two dimensional
* no maximization step

As a consequence, the algorithm is orders of magnitude faster than VFI.

This section illustrates the point that when it comes to programming, a bit of
mathematical analysis goes a long way.


Another Functional Equation
---------------------------

To begin, note that when :math:`w = \bar w(\pi)`, the worker is indifferent
between accepting and rejecting.

Hence the two choices on the right-hand side of :eq:`odu_mvf` have equal value:

.. math::
    :label: odu_mvf2

    \frac{\bar w(\pi)}{1 - \beta}
    = c + \beta \int v(w', \pi') \, q_{\pi}(w') \, dw'


Together, :eq:`odu_mvf` and :eq:`odu_mvf2` give

.. math::
    :label: odu_mvf3

    v(w, \pi) =
    \max
    \left\{
        \frac{w}{1 - \beta} ,\, \frac{\bar w(\pi)}{1 - \beta}
    \right\}


Combining :eq:`odu_mvf2` and :eq:`odu_mvf3`, we obtain

.. math::

    \frac{\bar w(\pi)}{1 - \beta}
    = c + \beta \int \max \left\{
        \frac{w'}{1 - \beta} ,\, \frac{\bar w(\pi')}{1 - \beta}
    \right\}
    \, q_{\pi}(w') \, dw'


Multiplying by :math:`1 - \beta`, substituting in :math:`\pi' = \kappa(w', \pi)`
and using :math:`\circ` for composition of functions yields

.. math::
    :label: odu_mvf4

    \bar w(\pi)
    = (1 - \beta) c +
    \beta \int \max \left\{ w', \bar w \circ \kappa(w', \pi) \right\} \, q_{\pi}(w') \, dw'


Equation :eq:`odu_mvf4` can be understood as a functional equation, where :math:`\bar w` is the unknown function.

* Let's call it the *reservation wage functional equation* (RWFE).
* The solution :math:`\bar w` to the RWFE is the object that we wish to compute.


Solving the RWFE
----------------

To solve the RWFE, we will first show that its solution is the
fixed point of a `contraction mapping <https://en.wikipedia.org/wiki/Contraction_mapping>`_.

To this end, let

* :math:`b[0,1]` be the bounded real-valued functions on :math:`[0,1]`
* :math:`\| \omega \| := \sup_{x \in [0,1]} | \omega(x) |`

Consider the operator :math:`Q` mapping :math:`\omega \in b[0,1]` into :math:`Q\omega \in b[0,1]` via


.. math::
    :label: odu_dq

    (Q \omega)(\pi)
    = (1 - \beta) c +
    \beta \int \max \left\{ w', \omega \circ \kappa(w', \pi) \right\} \, q_{\pi}(w') \, dw'


Comparing :eq:`odu_mvf4` and :eq:`odu_dq`, we see that the set of fixed points of :math:`Q` exactly coincides with the set of solutions to the RWFE.

* If :math:`Q \bar w = \bar w` then :math:`\bar w` solves :eq:`odu_mvf4` and vice versa.

Moreover, for any :math:`\omega, \omega' \in b[0,1]`, basic algebra and the
triangle inequality for integrals tells us that

.. math::
    :label: odu_nt

    |(Q \omega)(\pi) - (Q \omega')(\pi)|
    \leq \beta \int
    \left|
    \max \left\{w', \omega \circ \kappa(w', \pi) \right\} -
    \max \left\{w', \omega' \circ \kappa(w', \pi) \right\}
    \right|
    \, q_{\pi}(w') \, dw'


Working case by case, it is easy to check that for real numbers :math:`a, b, c` we always have

.. math::
    :label: odu_nt2

    | \max\{a, b\} - \max\{a, c\}| \leq | b - c|


Combining :eq:`odu_nt` and :eq:`odu_nt2` yields

.. math::
    :label: odu_nt3

    |(Q \omega)(\pi) - (Q \omega')(\pi)|
    \leq \beta \int
    \left| \omega \circ \kappa(w', \pi) -  \omega' \circ \kappa(w', \pi) \right|
    \, q_{\pi}(w') \, dw'
    \leq \beta \| \omega - \omega' \|


Taking the supremum over :math:`\pi` now gives us

.. math::
    :label: odu_rwc

    \|Q \omega - Q \omega'\|
    \leq \beta \| \omega - \omega' \|


In other words, :math:`Q` is a contraction of modulus :math:`\beta` on the
complete metric space :math:`(b[0,1], \| \cdot \|)`.

Hence

* A unique solution :math:`\bar w` to the RWFE exists in :math:`b[0,1]`.
* :math:`Q^k \omega \to \bar w` uniformly as :math:`k \to \infty`, for any :math:`\omega \in b[0,1]`.

Implementation
^^^^^^^^^^^^^^


The following function takes an instance of ``SearchProblem`` and returns the
operator ``Q``

.. code-block:: python3

    def Q_factory(sp, parallel_flag=True):

        f, g = sp.f, sp.g
        w_f, w_g = sp.w_f, sp.w_g
        β, c = sp.β, sp.c
        mc_size = sp.mc_size
        w_grid, π_grid = sp.w_grid, sp.π_grid

        @njit
        def κ(w, π):
            """
            Updates π using Bayes' rule and the current wage observation w.
            """
            pf, pg = π * f(w), (1 - π) * g(w)
            π_new = pf / (pf + pg)

            return π_new

        @njit(parallel=parallel_flag)
        def Q(ω):
            """
            Updates the reservation wage function guess ω via the operator Q.
            """
            ω_func = lambda p: interp(π_grid, ω, p)
            ω_new = np.empty_like(ω)

            for i in prange(len(π_grid)):
                π = π_grid[i]
                integral_f, integral_g = 0.0, 0.0

                for m in prange(mc_size):
                    integral_f += max(w_f[m], ω_func(κ(w_f[m], π)))
                    integral_g += max(w_g[m], ω_func(κ(w_g[m], π)))
                integral = (π * integral_f + (1 - π) * integral_g) / mc_size

                ω_new[i] = (1 - β) * c + β * integral

            return ω_new

        return Q

In the next exercise, you are asked to compute an approximation to :math:`\bar w`.


Exercises
=========

.. _odu_ex1:

Exercise 1
----------

Use the default parameters and ``Q_factory`` to compute an optimal policy.

Your result should coincide closely with the figure for the optimal policy :ref:`shown above<odu_pol_vfi>`.

Try experimenting with different parameters, and confirm that the change in
the optimal policy coincides with your intuition.



Solutions
=========



Exercise 1
----------

This code solves the "Offer Distribution Unknown" model by iterating on
a guess of the reservation wage function.

You should find that the run time is shorter than that of the value
function approach.

Similar to above, we set up a function to iterate with ``Q`` to find the fixed point

.. code-block:: python3

    def solve_wbar(sp,
                   use_parallel=True,
                   tol=1e-4,
                   max_iter=1000,
                   verbose=True,
                   print_skip=5):

        Q = Q_factory(sp, use_parallel)

        # Set up loop
        i = 0
        error = tol + 1
        m, n = len(sp.w_grid), len(sp.π_grid)

        # Initialize w
        w = np.ones_like(sp.π_grid)

        while i < max_iter and error > tol:
            w_new = Q(w)
            error = np.max(np.abs(w - w_new))
            i += 1
            if verbose and i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")
            w = w_new

        if i == max_iter:
            print("Failed to converge!")

        if verbose and i < max_iter:
            print(f"\nConverged in {i} iterations.")

        return w_new

The solution can be plotted as follows

.. code-block:: python3

    sp = SearchProblem()
    w_bar = solve_wbar(sp)

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(sp.π_grid, w_bar, color='k')
    ax.fill_between(sp.π_grid, 0, w_bar, color='blue', alpha=0.15)
    ax.fill_between(sp.π_grid, w_bar, sp.w_max, color='green', alpha=0.15)
    ax.text(0.5, 0.6, 'reject')
    ax.text(0.7, 0.9, 'accept')
    ax.set(xlabel='$\pi$', ylabel='$w$')
    ax.grid()
    plt.show()


Appendix
========

The next piece of code is just a fun simulation to see what the effect of a change in the
underlying distribution on the unemployment rate is.

At a point in the simulation, the distribution becomes significantly worse.

It takes a while for agents to learn this, and in the meantime, they are too optimistic
and turn down too many jobs.

As a result, the unemployment rate spikes

.. code-block:: python3

    F_a, F_b, G_a, G_b = 1, 1, 3, 1.2

    sp = SearchProblem(F_a=F_a, F_b=F_b, G_a=G_a, G_b=G_b)
    f, g = sp.f, sp.g

    # Solve for reservation wage
    w_bar = solve_wbar(sp, verbose=False)

    # Interpolate reservation wage function
    π_grid = sp.π_grid
    w_func = njit(lambda x: interp(π_grid, w_bar, x))

    @njit
    def update(a, b, e, π):
        """
        Update e and π by drawing wage offer from beta distribution with 
        parameters a and b
        """

        if e == False:
            w = np.random.beta(a, b)       # Draw random wage
            if w >= w_func(π):
                e = True                   # Take new job
            else:
                π = 1 / (1 + ((1 - π) * g(w)) / (π * f(w)))

        return e, π

    @njit
    def simulate_path(F_a=F_a,
                      F_b=F_b,
                      G_a=G_a,
                      G_b=G_b,
                      N=5000,       # Number of agents
                      T=600,        # Simulation length
                      d=200,        # Change date
                      s=0.025):     # Separation rate

        """
        Simulates path of employment for N number of works over T periods
        """

        e = np.ones((N, T+1))
        π = np.ones((N, T+1)) * 1e-3

        a, b = G_a, G_b   # Initial distribution parameters

        for t in range(T+1):

            if t == d:
                a, b = F_a, F_b  # Change distribution parameters

            # Update each agent
            for n in range(N):
                # If agent is currently employed
                if e[n, t] == 1:
                    p = np.random.uniform(0, 1)
                    # Randomly separate with probability s
                    if p <= s:
                        e[n, t] = 0

                new_e, new_π = update(a, b, e[n, t], π[n, t])
                e[n, t+1] = new_e
                π[n, t+1] = new_π

        return e[:, 1:]

    d = 200  # Change distribution at time d
    unemployment_rate = 1 - simulate_path(d=d).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(unemployment_rate)
    ax.axvline(d, color='r', alpha=0.6, label='Change date')
    ax.set_xlabel('Time')
    ax.set_title('Unemployment rate')
    ax.legend()
    plt.show()
