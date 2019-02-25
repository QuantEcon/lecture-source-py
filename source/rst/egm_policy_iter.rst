.. _egm_policy_iter:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3

********************************************************
:index:`Optimal Growth III: The Endogenous Grid Method`
********************************************************

.. contents:: :depth: 2


Overview
============

We solved the stochastic optimal growth model using 

#. :doc:`value function iteration <optgrowth>` 
#. :doc:`Euler equation based time iteration <coleman_policy_iter>` 

We found time iteration to be significantly more accurate at each step

In this lecture we'll look at an ingenious twist on the time iteration technique called the **endogenous grid method** (EGM) 

EGM is a numerical method for implementing policy iteration invented by `Chris Carroll <http://www.econ2.jhu.edu/people/ccarroll/>`__

It is a good example of how a clever algorithm can save a massive amount of computer time

(Massive when we multiply saved CPU cycles on each implementation times the number of implementations worldwide)

The original reference is :cite:`Carroll2006`




Key Idea
==========================

Let's start by reminding ourselves of the theory and then see how the numerics fit in



Theory
------

Take the model set out in :doc:`the time iteration lecture <coleman_policy_iter>`, following the same terminology and notation 

The Euler equation is

.. math::
    :label: egm_euler

    (u'\circ c^*)(y) 
    = \beta \int (u'\circ c^*)(f(y - c^*(y)) z) f'(y - c^*(y)) z \phi(dz)


As we saw, the Coleman operator is a nonlinear operator :math:`K` engineered so that :math:`c^*` is a fixed point of :math:`K`

It takes as its argument a continuous strictly increasing consumption policy :math:`g \in \Sigma` 

It returns a new function :math:`Kg`,  where :math:`(Kg)(y)` is the :math:`c \in (0, \infty)` that solves

.. math::
    :label: egm_coledef

    u'(c) 
    = \beta \int (u' \circ g) (f(y - c) z ) f'(y - c) z \phi(dz)




Exogenous Grid
-------------------

As discussed in :doc:`the lecture on time iteration <coleman_policy_iter>`, to implement the method on a computer we need numerical approximation

In particular, we represent a policy function by a set of values on a finite grid

The function itself is reconstructed from this representation when necessary, using interpolation or some other method

:doc:`Previously <coleman_policy_iter>`, to obtain a finite representation of an updated consumption policy we

* fixed a grid of income points :math:`\{y_i\}`

* calculated the consumption value :math:`c_i` corresponding to each
  :math:`y_i` using :eq:`egm_coledef` and a root finding routine

Each :math:`c_i` is then interpreted as the value of the function :math:`K g` at :math:`y_i` 

Thus, with the points :math:`\{y_i, c_i\}` in hand, we can reconstruct :math:`Kg` via approximation 

Iteration then continues...




Endogenous Grid
--------------------

The method discussed above requires a root finding routine to find the
:math:`c_i` corresponding to a given income value :math:`y_i`

Root finding is costly because it typically involves a significant number of
function evaluations

As pointed out by Carroll :cite:`Carroll2006`, we can avoid this if
:math:`y_i` is chosen endogenously

The only assumption required is that :math:`u'` is invertible on :math:`(0, \infty)`

The idea is this:

First we fix an *exogenous* grid :math:`\{k_i\}` for capital (:math:`k = y - c`)

Then we obtain  :math:`c_i` via

.. math::
    :label: egm_getc

    c_i = 
    (u')^{-1}
    \left\{
        \beta \int (u' \circ g) (f(k_i) z ) \, f'(k_i) \, z \, \phi(dz)
    \right\}

where :math:`(u')^{-1}` is the inverse function of :math:`u'`

Finally, for each :math:`c_i` we set :math:`y_i = c_i + k_i`

It is clear that each :math:`(y_i, c_i)` pair constructed in this manner satisfies :eq:`egm_coledef`

With the points :math:`\{y_i, c_i\}` in hand, we can reconstruct :math:`Kg` via approximation as before

The name EGM comes from the fact that the grid :math:`\{y_i\}` is  determined **endogenously**


Implementation
================

Let's implement this version of the Coleman operator and see how it performs


The Operator
----------------


Here's an implementation of :math:`K` using EGM as described above

.. literalinclude:: /_static/code/egm_policy_iter/coleman_egm.py

Note the lack of any root finding algorithm 

We'll also run our original implementation, which uses an exogenous grid and requires root finding, so we can perform some comparisons


.. literalinclude:: /_static/code/coleman_policy_iter/coleman.py
    :class: collapse


Let's test out the code above on some example parameterizations, after the following imports


   
.. code-block:: ipython 

    import matplotlib.pyplot as plt
    %matplotlib inline 
    import quantecon as qe





Testing on the Log / Cobb--Douglas case
------------------------------------------


As we :doc:`did for value function iteration <optgrowth>` and :doc:`time iteration <coleman_policy_iter>`, let's start by testing our method with the log-linear benchmark




The first step is to bring in the log-linear growth model that we used in the :doc:`value function iteration lecture <optgrowth>`

.. literalinclude:: /_static/code/optgrowth/loglinear_og.py




Next we generate an instance


.. code-block:: python3 

    lg = LogLinearOG()

    # == Unpack parameters / functions for convenience == #
    α, β, μ, s = lg.α, lg.β, lg.μ, lg.s
    v_star, c_star = lg.v_star, lg.c_star
    u, u_prime, f, f_prime = lg.u, lg.u_prime, lg.f, lg.f_prime 







We also need a grid over capital and some shock draws for Monte Carlo integration

.. code-block:: python3 

    grid_max = 4         # Largest grid point, exogenous grid
    grid_size = 200      # Number of grid points
    shock_size = 250     # Number of shock draws in Monte Carlo integral

    k_grid = np.linspace(1e-5, grid_max, grid_size)
    shocks = np.exp(μ + s * np.random.randn(shock_size))




As a preliminary test, let's see if :math:`K c^* = c^*`, as implied by the theory


.. code-block:: python3 

    c_star_new = coleman_egm(c_star,
                k_grid, β, u_prime, u_prime, f, f_prime, shocks)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(k_grid, c_star(k_grid), label="optimal policy $c^*$")
    ax.plot(k_grid, c_star_new(k_grid), label="$Kc^*$")

    ax.legend(loc='upper left')
    plt.show()




Notice that we're passing `u_prime` to `coleman_egm` twice

The reason is that, in the case of log utility, :math:`u'(c) = (u')^{-1}(c) = 1/c`

Hence `u_prime` and `u_prime_inv` are the same

We can't really distinguish the two plots

In fact it's easy to see that the difference is essentially zero:



.. code-block:: python3 

    max(abs(c_star_new(k_grid) - c_star(k_grid)))





Next let's try iterating from an arbitrary initial condition and see if we
converge towards :math:`c^*`


Let's start from the consumption policy that eats the whole pie: :math:`c(y) = y`



.. code-block:: python3 

    g = lambda x: x
    n = 15
    fig, ax = plt.subplots(figsize=(9, 6))
    lb = 'initial condition $c(y) = y$'

    ax.plot(k_grid, g(k_grid), color=plt.cm.jet(0), lw=2, alpha=0.6, label=lb)

    for i in range(n):
        new_g = coleman_egm(g, k_grid, β, u_prime, u_prime, f, f_prime, shocks)
        g = new_g
        ax.plot(k_grid, g(k_grid), color=plt.cm.jet(i / n), lw=2, alpha=0.6)

    lb = 'true policy function $c^*$'
    ax.plot(k_grid, c_star(k_grid), 'k-', lw=2, alpha=0.8, label=lb)
    ax.legend(loc='upper left')

    plt.show()



We see that the policy has converged nicely, in only a few steps


Speed
========

Now let's compare the clock times per iteration for the standard Coleman
operator (with exogenous grid) and the EGM version

We'll do so using the CRRA model adopted in the exercises of the :doc:`Euler equation time iteration lecture <coleman_policy_iter>`

Here's the model and some convenient functions



.. code-block:: python3 

    ## Define the model

    α = 0.65
    β = 0.95
    μ = 0
    s = 0.1
    grid_min = 1e-6
    grid_max = 4
    grid_size = 200
    shock_size = 250

    γ = 1.5   # Preference parameter
    γ_inv = 1 / γ

    def f(k):
        return k**α

    def f_prime(k):
        return α * k**(α - 1)

    def u(c):
        return (c**(1 - γ) - 1) / (1 - γ)

    def u_prime(c):
        return c**(-γ)

    def u_prime_inv(c):
        return c**(-γ_inv)

    k_grid = np.linspace(grid_min, grid_max, grid_size)
    shocks = np.exp(μ + s * np.random.randn(shock_size))

    ## Let's make convenience functions based around these primitives

    def crra_coleman(g):
        return coleman_operator(g, k_grid, β, u_prime, f, f_prime, shocks)

    def crra_coleman_egm(g):
        return coleman_egm(g, k_grid, β, u_prime, u_prime_inv, f, f_prime, shocks)




Here's the result



.. code-block:: python3 

    ## Iterate, compare policies

    sim_length = 20

    print("Timing standard Coleman policy function iteration")
    g_init = k_grid
    g = g_init
    qe.util.tic()
    for i in range(sim_length):
        new_g = crra_coleman(g)
        g = new_g
    qe.util.toc()


    print("Timing policy function iteration with endogenous grid")
    g_init_egm = lambda x: x
    g = g_init_egm
    qe.util.tic()
    for i in range(sim_length):
        new_g = crra_coleman_egm(g)
        g = new_g
    qe.util.toc()


We see that the EGM version is more than 6 times faster



At the same time, the absence of numerical root finding means that it is
typically more accurate at each step as well