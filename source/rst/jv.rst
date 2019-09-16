.. _jv:

.. include:: /_static/includes/header.raw

.. highlight:: python3

****************************************
:index:`Job Search V: On-the-Job Search`
****************************************

.. index::
    single: Models; On-the-Job Search

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon
  !pip install interpolation

Overview
========

In this section, we solve a simple on-the-job search model

* based on :cite:`Ljungqvist2012`, exercise 6.18, and :cite:`Jovanovic1979`

Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import scipy.stats as stats
    from interpolation import interp
    from numba import njit, prange
    import matplotlib.pyplot as plt
    %matplotlib inline
    from math import gamma

Model Features
--------------

.. index::
    single: On-the-Job Search; Model Features

* job-specific human capital accumulation combined with on-the-job search

* infinite-horizon dynamic programming with one state variable and two controls


Model
========

.. index::
    single: On-the-Job Search; Model

Let :math:`x_t` denote the time-:math:`t` job-specific human capital of a worker employed at a given firm and let  :math:`w_t` denote current wages.

Let :math:`w_t = x_t(1 - s_t - \phi_t)`, where

* :math:`\phi_t` is investment in job-specific human capital for the current role and

* :math:`s_t` is search effort, devoted to obtaining new offers from other firms.

For as long as the worker remains in the current job, evolution of :math:`\{x_t\}` is given by :math:`x_{t+1} = g(x_t, \phi_t)`.

When search effort at :math:`t` is :math:`s_t`, the worker receives a new job offer with probability :math:`\pi(s_t) \in [0, 1]`.

The value of the offer, measured in job-specific human capital,  is :math:`u_{t+1}`, where :math:`\{u_t\}` is IID with common distribution :math:`f`.

The worker can reject the current offer and continue with existing job.

Hence :math:`x_{t+1} = u_{t+1}` if he/she accepts and :math:`x_{t+1} = g(x_t, \phi_t)` otherwise.

Let :math:`b_{t+1} \in \{0,1\}` be a binary random variable, where :math:`b_{t+1} = 1` indicates that the worker receives an offer at the end of time :math:`t`. 

We can write

.. math::
    :label: jd

    x_{t+1}
    = (1 - b_{t+1}) g(x_t, \phi_t) + b_{t+1}
        \max \{ g(x_t, \phi_t), u_{t+1}\}


Agent's objective: maximize expected discounted sum of wages via controls :math:`\{s_t\}` and :math:`\{\phi_t\}`.

Taking the expectation of :math:`v(x_{t+1})` and using :eq:`jd`,
the Bellman equation for this problem can be written as

.. math::
    :label: jvbell

    v(x)
    = \max_{s + \phi \leq 1}
        \left\{
            x (1 - s - \phi) + \beta (1 - \pi(s)) v[g(x, \phi)] +
            \beta \pi(s) \int v[g(x, \phi) \vee u] f(du)
         \right\}


Here nonnegativity of :math:`s` and :math:`\phi` is understood, while
:math:`a \vee b := \max\{a, b\}`.


Parameterization
----------------

.. index::
    single: On-the-Job Search; Parameterization

In the implementation below, we will focus on the parameterization

.. math::

    g(x, \phi) = A (x \phi)^{\alpha},
    \quad
    \pi(s) = \sqrt s
    \quad \text{and} \quad
    f = \text{Beta}(2, 2)


with default parameter values

* :math:`A = 1.4`
* :math:`\alpha = 0.6`
* :math:`\beta = 0.96`


The :math:`\text{Beta}(2,2)` distribution is supported on :math:`(0,1)` - it has a unimodal, symmetric density peaked at 0.5.


.. _jvboecalc:

Back-of-the-Envelope Calculations
---------------------------------


Before we solve the model, let's make some quick calculations that
provide intuition on what the solution should look like.

To begin, observe that the worker has two instruments to build
capital and hence wages:

#. invest in capital specific to the current job via :math:`\phi`
#. search for a new job with better job-specific capital match via :math:`s`

Since wages are :math:`x (1 - s - \phi)`, marginal cost of investment via either :math:`\phi` or :math:`s` is identical.

Our risk-neutral worker should focus on whatever instrument has the highest expected return.

The relative expected return will depend on :math:`x`.

For example, suppose first that :math:`x = 0.05`

* If :math:`s=1` and :math:`\phi = 0`, then since :math:`g(x,\phi) = 0`,
  taking expectations of :eq:`jd` gives expected next period capital equal to :math:`\pi(s) \mathbb{E} u
  = \mathbb{E} u = 0.5`.
* If :math:`s=0` and :math:`\phi=1`, then next period capital is :math:`g(x, \phi) = g(0.05, 1) \approx 0.23`.

Both rates of return are good, but the return from search is better.

Next, suppose that :math:`x = 0.4`

* If :math:`s=1` and :math:`\phi = 0`, then expected next period capital is again :math:`0.5`
* If :math:`s=0` and :math:`\phi = 1`, then :math:`g(x, \phi) = g(0.4, 1) \approx 0.8`

Return from investment via :math:`\phi` dominates expected return from search.

Combining these observations gives us two informal predictions:

#. At any given state :math:`x`, the two controls :math:`\phi` and :math:`s` will
   function primarily as substitutes --- worker will focus on whichever instrument has the higher expected return.
#. For sufficiently small :math:`x`, search will be preferable to investment in
   job-specific human capital.  For larger :math:`x`, the reverse will be true.

Now let's turn to implementation, and see if we can match our predictions.


Implementation
==============

.. index::
    single: On-the-Job Search; Programming Implementation

We will set up a class ``JVWorker`` that holds the parameters of the model described above

.. code-block:: python3

    class JVWorker:
        r"""
        A Jovanovic-type model of employment with on-the-job search.

        """

        def __init__(self,
                     A=1.4,
                     α=0.6,
                     β=0.96,         # Discount factor
                     π=np.sqrt,      # Search effort function
                     a=2,            # Parameter of f
                     b=2,            # Parameter of f
                     grid_size=50,
                     mc_size=100,
                     ɛ=1e-4):

            self.A, self.α, self.β, self.π = A, α, β, π
            self.mc_size, self.ɛ = mc_size, ɛ

            self.g = njit(lambda x, ϕ: A * (x * ϕ)**α)    # Transition function
            self.f_rvs = np.random.beta(a, b, mc_size)

            # Max of grid is the max of a large quantile value for f and the
            # fixed point y = g(y, 1)
            ɛ = 1e-4
            grid_max = max(A**(1 / (1 - α)), stats.beta(a, b).ppf(1 - ɛ))

            # Human capital
            self.x_grid = np.linspace(ɛ, grid_max, grid_size)


The function ``operator_factory`` takes an instance of this class and returns a
jitted version of the Bellman operator ``T``, ie.

.. math::

    Tv(x)
    = \max_{s + \phi \leq 1} w(s, \phi)


where

.. math::
    :label: defw

     w(s, \phi)
     := x (1 - s - \phi) + \beta (1 - \pi(s)) v[g(x, \phi)] +
             \beta \pi(s) \int v[g(x, \phi) \vee u] f(du)



When we represent :math:`v`, it will be with a NumPy array ``v`` giving values on grid ``x_grid``.

But to evaluate the right-hand side of :eq:`defw`, we need a function, so
we replace the arrays ``v`` and ``x_grid`` with a function ``v_func`` that gives linear
interpolation of ``v`` on ``x_grid``.

Inside the ``for`` loop, for each ``x`` in the grid over the state space, we
set up the function :math:`w(z) = w(s, \phi)` defined in :eq:`defw`.

The function is maximized over all feasible :math:`(s, \phi)` pairs.

Another function, ``get_greedy`` returns the optimal choice of :math:`s` and :math:`\phi`
at each :math:`x`, given a value function.

.. code-block:: python3

    def operator_factory(jv, parallel_flag=True):

        """
        Returns a jitted version of the Bellman operator T

        jv is an instance of JVWorker

        """

        π, β = jv.π, jv.β
        x_grid, ɛ, mc_size = jv.x_grid, jv.ɛ, jv.mc_size
        f_rvs, g = jv.f_rvs, jv.g

        @njit
        def objective(z, x, v):
            s, ϕ = z
            v_func = lambda x: interp(x_grid, v, x)

            integral = 0
            for m in range(mc_size):
                u = f_rvs[m]
                integral += v_func(max(g(x, ϕ), u))
            integral = integral / mc_size

            q = π(s) * integral + (1 - π(s)) * v_func(g(x, ϕ))
            return x * (1 - ϕ - s) + β * q

        @njit(parallel=parallel_flag)
        def T(v):
            """
            The Bellman operator
            """

            v_new = np.empty_like(v)
            for i in prange(len(x_grid)):
                x = x_grid[i]

                # Search on a grid
                search_grid = np.linspace(ɛ, 1, 15)
                max_val = -1
                for s in search_grid:
                    for ϕ in search_grid:
                        current_val = objective((s, ϕ), x, v) if s + ϕ <= 1 else -1
                        if current_val > max_val:
                            max_val = current_val
                v_new[i] = max_val

            return v_new

        @njit
        def get_greedy(v):
            """
            Computes the v-greedy policy of a given function v
            """
            s_policy, ϕ_policy = np.empty_like(v), np.empty_like(v)

            for i in range(len(x_grid)):
                x = x_grid[i]
                # Search on a grid
                search_grid = np.linspace(ɛ, 1, 15)
                max_val = -1
                for s in search_grid:
                    for ϕ in search_grid:
                        current_val = objective((s, ϕ), x, v) if s + ϕ <= 1 else -1
                        if current_val > max_val:
                            max_val = current_val
                            max_s, max_ϕ = s, ϕ
                            s_policy[i], ϕ_policy[i] = max_s, max_ϕ
            return s_policy, ϕ_policy

        return T, get_greedy

To solve the model, we will write a function that uses the Bellman operator
and iterates to find a fixed point.

.. code-block:: python3

    def solve_model(jv,
                    use_parallel=True,
                    tol=1e-4,
                    max_iter=1000,
                    verbose=True,
                    print_skip=25):

        """
        Solves the model by value function iteration

        * jv is an instance of JVWorker

        """

        T, _ = operator_factory(jv, parallel_flag=use_parallel)

        # Set up loop
        v = jv.x_grid * 0.5  # Initial condition
        i = 0
        error = tol + 1

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


Solving for Policies
====================

.. index::
    single: On-the-Job Search; Solving for Policies

Let's generate the optimal policies and see what they look like.

.. _jv_policies:

.. code-block:: python3

    jv = JVWorker()
    T, get_greedy = operator_factory(jv)
    v_star = solve_model(jv)
    s_star, ϕ_star = get_greedy(v_star)


Here's the plots:

.. code-block:: python3

    plots = [s_star, ϕ_star, v_star]
    titles = ["s policy", "ϕ policy",  "value function"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for ax, plot, title in zip(axes, plots, titles):
        ax.plot(jv.x_grid, plot)
        ax.set(title=title)
        ax.grid()

    axes[-1].set_xlabel("x")
    plt.show()


The horizontal axis is the state :math:`x`, while the vertical axis gives :math:`s(x)` and :math:`\phi(x)`.

Overall, the policies match well with our predictions from :ref:`above <jvboecalc>`

* Worker switches from one investment strategy to the other depending on relative return.

* For low values of :math:`x`, the best option is to search for a new job.

* Once :math:`x` is larger, worker does better by investing in human capital specific to the current position.


Exercises
=========

.. _jv_ex1:

Exercise 1
----------

Let's look at the dynamics for the state process :math:`\{x_t\}` associated with these policies.

The dynamics are given by :eq:`jd` when :math:`\phi_t` and :math:`s_t` are
chosen according to the optimal policies, and :math:`\mathbb{P}\{b_{t+1} = 1\}
= \pi(s_t)`.

Since the dynamics are random, analysis is a bit subtle.

One way to do it is to plot, for each :math:`x` in a relatively fine grid
called ``plot_grid``, a
large number :math:`K` of realizations of :math:`x_{t+1}` given :math:`x_t =
x`.

Plot this with one dot for each realization, in the form of a 45 degree
diagram, setting


.. code-block:: python3
    :class: no-execute

    jv = JVWorker(grid_size=25, mc_size=50)
    plot_grid_max, plot_grid_size = 1.2, 100
    plot_grid = np.linspace(0, plot_grid_max, plot_grid_size)
    fig, ax = plt.subplots()
    ax.set_xlim(0, plot_grid_max)
    ax.set_ylim(0, plot_grid_max)




By examining the plot, argue that under the optimal policies, the state
:math:`x_t` will converge to a constant value :math:`\bar x` close to unity.

Argue that at the steady state, :math:`s_t \approx 0` and :math:`\phi_t \approx 0.6`.







.. _jv_ex2:

Exercise 2
----------

In the preceding exercise, we found that :math:`s_t` converges to zero
and :math:`\phi_t` converges to about 0.6.

Since these results were calculated at a value of :math:`\beta` close to
one, let's compare them to the best choice for an *infinitely* patient worker.

Intuitively, an infinitely patient worker would like to maximize steady state
wages, which are a function of steady state capital.

You can take it as given---it's certainly true---that the infinitely patient worker does not
search in the long run (i.e., :math:`s_t = 0` for large :math:`t`).

Thus, given :math:`\phi`, steady state capital is the positive fixed point
:math:`x^*(\phi)` of the map :math:`x \mapsto g(x, \phi)`.

Steady state wages can be written as :math:`w^*(\phi) = x^*(\phi) (1 - \phi)`.

Graph :math:`w^*(\phi)` with respect to :math:`\phi`, and examine the best
choice of :math:`\phi`.

Can you give a rough interpretation for the value that you see?




Solutions
=========



Exercise 1
----------

Here’s code to produce the 45 degree diagram

.. code-block:: python3

    jv = JVWorker(grid_size=25, mc_size=50)
    π, g, f_rvs, x_grid = jv.π, jv.g, jv.f_rvs, jv.x_grid
    T, get_greedy = operator_factory(jv)
    v_star = solve_model(jv, verbose=False)
    s_policy, ϕ_policy = get_greedy(v_star)

    # Turn the policy function arrays into actual functions
    s = lambda y: interp(x_grid, s_policy, y)
    ϕ = lambda y: interp(x_grid, ϕ_policy, y)

    def h(x, b, u):
        return (1 - b) * g(x, ϕ(x)) + b * max(g(x, ϕ(x)), u)


    plot_grid_max, plot_grid_size = 1.2, 100
    plot_grid = np.linspace(0, plot_grid_max, plot_grid_size)
    fig, ax = plt.subplots(figsize=(8, 8))
    ticks = (0.25, 0.5, 0.75, 1.0)
    ax.set(xticks=ticks, yticks=ticks,
           xlim=(0, plot_grid_max),
           ylim=(0, plot_grid_max),
           xlabel='$x_t$', ylabel='$x_{t+1}$')

    ax.plot(plot_grid, plot_grid, 'k--', alpha=0.6)  # 45 degree line
    for x in plot_grid:
        for i in range(jv.mc_size):
            b = 1 if np.random.uniform(0, 1) < π(s(x)) else 0
            u = f_rvs[i]
            y = h(x, b, u)
            ax.plot(x, y, 'go', alpha=0.25)

    plt.show()


Looking at the dynamics, we can see that

-  If :math:`x_t` is below about 0.2 the dynamics are random, but
   :math:`x_{t+1} > x_t` is very likely.
-  As :math:`x_t` increases the dynamics become deterministic, and
   :math:`x_t` converges to a steady state value close to 1.

Referring back to the figure :ref:`here <jv_policies>` we see that :math:`x_t \approx 1` means that
:math:`s_t = s(x_t) \approx 0` and
:math:`\phi_t = \phi(x_t) \approx 0.6`.

Exercise 2
----------

The figure can be produced as follows


.. code-block:: python3

    jv = JVWorker()

    def xbar(ϕ):
        A, α = jv.A, jv.α
        return (A * ϕ**α)**(1 / (1 - α))

    ϕ_grid = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set(xlabel='$\phi$')
    ax.plot(ϕ_grid, [xbar(ϕ) * (1 - ϕ) for ϕ in ϕ_grid], label='$w^*(\phi)$')
    ax.legend()

    plt.show()


Observe that the maximizer is around 0.6.

This is similar to the long-run value for :math:`\phi` obtained in
exercise 1.

Hence the behavior of the infinitely patent worker is similar to that
of the worker with :math:`\beta = 0.96`.

This seems reasonable and helps us confirm that our dynamic programming
solutions are probably correct.
