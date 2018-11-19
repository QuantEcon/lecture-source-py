.. _jv:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3

**********************************************
:index:`Job Search V: On-the-Job Search`
**********************************************

.. index::
    single: Models; On-the-Job Search

.. contents:: :depth: 2

Overview
============

In this section we solve a simple on-the-job search model

* based on :cite:`Ljungqvist2012`, exercise 6.18, and :cite:`Jovanovic1979`



Model features
----------------

.. index::
    single: On-the-Job Search; Model Features

* job-specific human capital accumulation combined with on-the-job search

* infinite horizon dynamic programming with one state variable and two controls


Model
========

.. index::
    single: On-the-Job Search; Model

Let

* :math:`x_t` denote the time-:math:`t` job-specific human capital of a worker employed at a given firm
* :math:`w_t` denote current wages

Let :math:`w_t = x_t(1 - s_t - \phi_t)`, where

* :math:`\phi_t` is investment in job-specific human capital for the current role
* :math:`s_t` is search effort, devoted to obtaining new offers from other firms.

For as long as the worker remains in the current job, evolution of
:math:`\{x_t\}` is given by :math:`x_{t+1} = G(x_t, \phi_t)`

When search effort at :math:`t` is :math:`s_t`, the worker receives a new job
offer with probability :math:`\pi(s_t) \in [0, 1]`

Value of offer is :math:`U_{t+1}`, where :math:`\{U_t\}` is iid with common distribution :math:`F`

Worker has the right to reject the current offer and continue with existing job

In particular, :math:`x_{t+1} = U_{t+1}` if accepts and :math:`x_{t+1} = G(x_t, \phi_t)` if rejects

Letting :math:`b_{t+1} \in \{0,1\}` be binary with :math:`b_{t+1} = 1` indicating an offer, we can write

.. math::
    :label: jd

    x_{t+1}
    = (1 - b_{t+1}) G(x_t, \phi_t) + b_{t+1}
        \max \{ G(x_t, \phi_t), U_{t+1}\}


Agent's objective: maximize expected discounted sum of wages via controls :math:`\{s_t\}` and :math:`\{\phi_t\}`

Taking the expectation of :math:`V(x_{t+1})` and using :eq:`jd`,
the Bellman equation for this problem can be written as

.. math::
    :label: jvbell

    V(x)
    = \max_{s + \phi \leq 1}
        \left\{
            x (1 - s - \phi) + \beta (1 - \pi(s)) V[G(x, \phi)] +
            \beta \pi(s) \int V[G(x, \phi) \vee u] F(du)
         \right\}.


Here nonnegativity of :math:`s` and :math:`\phi` is understood, while
:math:`a \vee b := \max\{a, b\}`


Parameterization
------------------

.. index::
    single: On-the-Job Search; Parameterization

In the implementation below, we will focus on the parameterization

.. math::

    G(x, \phi) = A (x \phi)^{\alpha},
    \quad
    \pi(s) = \sqrt s
    \quad \text{and} \quad
    F = \text{Beta}(2, 2)


with default parameter values

* :math:`A = 1.4`
* :math:`\alpha = 0.6`
* :math:`\beta = 0.96`


The Beta(2,2) distribution is supported on :math:`(0,1)`.  It has a unimodal, symmetric density peaked at 0.5


.. _jvboecalc:

Back-of-the-Envelope Calculations
-----------------------------------


Before we solve the model, let's make some quick calculations that
provide intuition on what the solution should look like

To begin, observe that the worker has two instruments to build
capital and hence wages:

#. invest in capital specific to the current job via :math:`\phi`
#. search for a new job with better job-specific capital match via :math:`s`

Since wages are :math:`x (1 - s - \phi)`, marginal cost of investment via either :math:`\phi` or :math:`s` is identical

Our risk neutral worker should focus on whatever instrument has the highest expected return

The relative expected return will depend on :math:`x`

For example, suppose first that :math:`x = 0.05`

* If :math:`s=1` and :math:`\phi = 0`, then since :math:`G(x,\phi) = 0`,
  taking expectations of :eq:`jd` gives expected next period capital equal to :math:`\pi(s) \mathbb{E} U
  = \mathbb{E} U = 0.5`
* If :math:`s=0` and :math:`\phi=1`, then next period capital is :math:`G(x, \phi) = G(0.05, 1) \approx 0.23`

Both rates of return are good, but the return from search is better

Next suppose that :math:`x = 0.4`

* If :math:`s=1` and :math:`\phi = 0`, then expected next period capital is again :math:`0.5`
* If :math:`s=0` and :math:`\phi = 1`, then :math:`G(x, \phi) = G(0.4, 1) \approx 0.8`

Return from investment via :math:`\phi` dominates expected return from search

Combining these observations gives us two informal predictions:

#. At any given state :math:`x`, the two controls :math:`\phi` and :math:`s` will function primarily as substitutes --- worker will focus on whichever instrument has the higher expected return
#. For sufficiently small :math:`x`, search will be preferable to investment in job-specific human capital.  For larger :math:`x`, the reverse will be true

Now let's turn to implementation, and see if we can match our predictions


Implementation
=======================

.. index::
    single: On-the-Job Search; Programming Implementation

The following code solves the DP problem described above


.. literalinclude:: /_static/code/jv/jv.py


The code is written to be relatively generic---and hence reusable

* For example, we use generic :math:`G(x,\phi)` instead of specific :math:`A (x \phi)^{\alpha}`


Regarding the imports

* ``fixed_quad`` is a simple non-adaptive integration routine

* ``fmin_slsqp`` is a minimization routine that permits inequality constraints



Next we build a class called ``JvWorker`` that

* packages all the parameters and other basic attributes of a given model

* implements the method ``bellman_operator`` for value function iteration

The ``bellman_operator`` method
takes a candidate value function :math:`V` and updates it to :math:`TV` via


.. math::

    TV(x)
    = - \min_{s + \phi \leq 1} w(s, \phi)


where

.. math::
    :label: defw

     w(s, \phi)
     := - \left\{
             x (1 - s - \phi) + \beta (1 - \pi(s)) V[G(x, \phi)] +
             \beta \pi(s) \int V[G(x, \phi) \vee u] F(du)
    \right\}


Here we are minimizing instead of maximizing to fit with SciPy's optimization routines

When we represent :math:`V`, it will be with a NumPy array ``V`` giving values on grid ``x_grid``

But to evaluate the right-hand side of :eq:`defw`, we need a function, so
we replace the arrays ``V`` and ``x_grid`` with a function ``Vf`` that gives linear
interpolation of ``V`` on ``x_grid``

Hence in the preliminaries of ``bellman_operator``

* from the array ``V`` we define a linear interpolation ``Vf`` of its values

    * ``c1`` is used to implement the constraint :math:`s + \phi \leq 1`

    * ``c2`` is used to implement :math:`s \geq \epsilon`, a numerically stable

      alternative to the true constraint :math:`s \geq 0`
    * ``c3`` does the same for :math:`\phi`

Inside the ``for`` loop, for each ``x`` in the grid over the state space, we
set up the function :math:`w(z) = w(s, \phi)` defined in :eq:`defw`.

The function is minimized over all feasible :math:`(s, \phi)` pairs, either by

* a relatively sophisticated solver from SciPy called ``fmin_slsqp``, or
* brute force search over a grid

The former is much faster, but convergence to the global optimum is not
guaranteed.  Grid search is a simple way to check results


Solving for Policies
=====================

.. index::
    single: On-the-Job Search; Solving for Policies

Let's plot the optimal policies and see what they look like

The code is as follows

.. _jv_policies:

.. literalinclude:: /_static/code/jv/jv_test.py



The horizontal axis is the state :math:`x`, while the vertical axis gives :math:`s(x)` and :math:`\phi(x)`

Overall, the policies match well with our predictions from :ref:`section <jvboecalc>`

* Worker switches from one investment strategy to the other depending on relative return

* For low values of :math:`x`, the best option is to search for a new job

* Once :math:`x` is larger, worker does better by investing in human capital specific to the current position


Exercises
=============

.. _jv_ex1:

Exercise 1
------------

Let's look at the dynamics for the state process :math:`\{x_t\}` associated with these policies

The dynamics are given by :eq:`jd` when :math:`\phi_t` and :math:`s_t` are
chosen according to the optimal policies, and :math:`\mathbb{P}\{b_{t+1} = 1\}
= \pi(s_t)`

Since the dynamics are random, analysis is a bit subtle

One way to do it is to plot, for each :math:`x` in a relatively fine grid
called ``plot_grid``, a
large number :math:`K` of realizations of :math:`x_{t+1}` given :math:`x_t =
x`.  Plot this with one dot for each realization, in the form of a 45 degree
diagram.  Set




.. code-block:: python3
    :class: no-execute

    K = 50
    plot_grid_max, plot_grid_size = 1.2, 100
    plot_grid = np.linspace(0, plot_grid_max, plot_grid_size)
    fig, ax = plt.subplots()
    ax.set_xlim(0, plot_grid_max)
    ax.set_ylim(0, plot_grid_max)




By examining the plot, argue that under the optimal policies, the state
:math:`x_t` will converge to a constant value :math:`\bar x` close to unity

Argue that at the steady state, :math:`s_t \approx 0` and :math:`\phi_t \approx 0.6`







.. _jv_ex2:

Exercise 2
----------------

In the preceding exercise we found that :math:`s_t` converges to zero
and :math:`\phi_t` converges to about 0.6

Since these results were calculated at a value of :math:`\beta` close to
one, let's compare them to the best choice for an *infinitely* patient worker

Intuitively, an infinitely patient worker would like to maximize steady state
wages, which are a function of steady state capital

You can take it as given---it's certainly true---that the infinitely patient worker does not
search in the long run (i.e., :math:`s_t = 0` for large :math:`t`)

Thus, given :math:`\phi`, steady state capital is the positive fixed point
:math:`x^*(\phi)` of the map :math:`x \mapsto G(x, \phi)`

Steady state wages can be written as :math:`w^*(\phi) = x^*(\phi) (1 - \phi)`

Graph :math:`w^*(\phi)` with respect to :math:`\phi`, and examine the best
choice of :math:`\phi`

Can you give a rough interpretation for the value that you see?




Solutions
==========



Exercise 1
----------

Here’s code to produce the 45 degree diagram

.. code-block:: python3

    import random
    
    wp = JvWorker(grid_size=25)
    G, π, F = wp.G, wp.π, wp.F       # Simplify names
    
    v_init = wp.x_grid * 0.5
    print("Computing value function")
    V = compute_fixed_point(wp.bellman_operator, v_init, max_iter=40, verbose=False)
    print("Computing policy functions")
    s_policy, ϕ_policy = wp.bellman_operator(V, return_policies=True)
    
    # Turn the policy function arrays into actual functions
    s = lambda y: np.interp(y, wp.x_grid, s_policy)
    ϕ = lambda y: np.interp(y, wp.x_grid, ϕ_policy)
    
    def h(x, b, U):
        return (1 - b) * G(x, ϕ(x)) + b * max(G(x, ϕ(x)), U)
    
    plot_grid_max, plot_grid_size = 1.2, 100
    plot_grid = np.linspace(0, plot_grid_max, plot_grid_size)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, plot_grid_max)
    ax.set_ylim(0, plot_grid_max)
    ticks = (0.25, 0.5, 0.75, 1.0)
    ax.set(xticks=ticks, yticks=ticks)
    ax.set_xlabel('$x_t$', fontsize=16)
    ax.set_ylabel('$x_{t+1}$', fontsize=16, rotation='horizontal')
    
    ax.plot(plot_grid, plot_grid, 'k--')  # 45 degree line
    for x in plot_grid:
        for i in range(50):
            b = 1 if random.uniform(0, 1) < π(s(x)) else 0
            U = wp.F.rvs(1)
            y = h(x, b, U)
            ax.plot(x, y, 'go', alpha=0.25)
    
    plt.show()


Looking at the dynamics, we can see that

-  If :math:`x_t` is below about 0.2 the dynamics are random, but
   :math:`x_{t+1} > x_t` is very likely
-  As :math:`x_t` increases the dynamics become deterministic, and
   :math:`x_t` converges to a steady state value close to 1

Referring back to the figure :ref:`here <jv_policies>` we see that :math:`x_t \approx 1` means that
:math:`s_t = s(x_t) \approx 0` and
:math:`\phi_t = \phi(x_t) \approx 0.6`

Exercise 2
----------

The figure can be produced as follows


.. code-block:: python3
    
    wp = JvWorker(grid_size=25)
    
    def xbar(ϕ):
        return (wp.A * ϕ**wp.α)**(1 / (1 - wp.α))
    
    ϕ_grid = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlabel('$\phi$', fontsize=16)
    ax.plot(ϕ_grid, [xbar(ϕ) * (1 - ϕ) for ϕ in ϕ_grid], 'b-', label='$w^*(\phi)$')
    ax.legend(loc='upper left')
    
    plt.show()


Observe that the maximizer is around 0.6

This this is similar to the long run value for :math:`\phi` obtained in
exercise 1

Hence the behaviour of the infinitely patent worker is similar to that
of the worker with :math:`\beta = 0.96`

This seems reasonable, and helps us confirm that our dynamic programming
solutions are probably correct


