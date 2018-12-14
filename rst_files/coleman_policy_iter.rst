.. _coleman_policy_iter:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3

*********************************************
:index:`Optimal Growth II: Time Iteration`
*********************************************

.. contents:: :depth: 2


Overview
============

In this lecture we'll continue our :doc:`earlier study <optgrowth>` of the stochastic optimal growth model

In that lecture we solved the associated discounted dynamic programming problem using value function iteration

The beauty of this technique is its broad applicability

With numerical problems, however, we can often attain higher efficiency in specific applications by deriving methods that are carefully tailored to the application at hand

The stochastic optimal growth model has plenty of structure to exploit for this purpose, especially when we adopt some concavity and smoothness assumptions over primitives

We'll use this structure to obtain an **Euler equation**  based method that's more efficient than value function iteration for this and some other closely related applications

In a :doc:`subsequent lecture <egm_policy_iter>` we'll see that the numerical implementation part of the Euler equation method can be further adjusted to obtain even more efficiency



The Euler Equation
==========================

Let's take the model set out in :doc:`the stochastic growth model lecture <optgrowth>` and add the assumptions that 

#. :math:`u` and :math:`f` are continuously differentiable and strictly concave

#. :math:`f(0) = 0`

#. :math:`\lim_{c \to 0} u'(c) = \infty` and :math:`\lim_{c \to \infty} u'(c) = 0`

#. :math:`\lim_{k \to 0} f'(k) = \infty` and :math:`\lim_{k \to \infty} f'(k) = 0`

The last two conditions are usually called **Inada conditions** 


Recall the Bellman equation 

.. math::
    :label: cpi_fpb30

    v^*(y) = \max_{0 \leq c \leq y}
        \left\{
            u(c) + \beta \int v^*(f(y - c) z) \phi(dz)
        \right\}
    \quad \text{for all} \quad
    y \in \mathbb R_+


Let the optimal consumption policy be denoted by :math:`c^*`

We know that :math:`c^*` is a :math:`v^*` greedy policy, so that :math:`c^*(y)` is the maximizer in :eq:`cpi_fpb30`

The conditions above imply that 

* :math:`c^*` is the unique optimal policy for the stochastic optimal growth model

* the optimal policy is continuous, strictly increasing and also **interior**, in the sense that :math:`0 < c^*(y) < y` for all strictly positive :math:`y`, and

* the value function is strictly concave and continuously differentiable, with

.. math::
    :label: cpi_env

    (v^*)'(y) = u' (c^*(y) ) := (u' \circ c^*)(y)

The last result is called the **envelope condition** due to its relationship with the `envelope theorem <https://en.wikipedia.org/wiki/Envelope_theorem>`_ 

To see why :eq:`cpi_env` might be valid, write the Bellman equation in the equivalent
form

.. math::

    v^*(y) = \max_{0 \leq k \leq y}
        \left\{
            u(y-k) + \beta \int v^*(f(k) z) \phi(dz)
        \right\},


differentiate naively with respect to :math:`y`,  and then  evaluate at the optimum 

Section 12.1 of `EDTC <http://johnstachurski.net/edtc.html>`_ contains full proofs of these results, and closely related discussions can be found in many other texts


Differentiability of the value function and iteriority of the optimal policy
imply that optimal consumption satisfies the first order condition associated
with :eq:`cpi_fpb30`, which is

.. math::
    :label: cpi_foc

    u'(c^*(y)) = \beta \int (v^*)'(f(y - c^*(y)) z) f'(y - c^*(y)) z \phi(dz)


Combining :eq:`cpi_env` and the first-order condition :eq:`cpi_foc` gives the famous **Euler equation**

.. math::
    :label: cpi_euler

    (u'\circ c^*)(y) 
    = \beta \int (u'\circ c^*)(f(y - c^*(y)) z) f'(y - c^*(y)) z \phi(dz)


We can think of the Euler equation as a functional equation

.. math::
    :label: cpi_euler_func

    (u'\circ \sigma)(y) 
    = \beta \int (u'\circ \sigma)(f(y - \sigma(y)) z) f'(y - \sigma(y)) z \phi(dz)


over interior consumption policies :math:`\sigma`, one solution of which is the optimal policy :math:`c^*`

Our aim is to solve the functional equation :eq:`cpi_euler_func` and hence obtain :math:`c^*`



The Coleman Operator
-------------------------------

Recall the Bellman operator

.. math::
    :label: fcbell20_coleman

    Tw(y) := \max_{0 \leq c \leq y}
    \left\{
        u(c) + \beta \int w(f(y - c) z) \phi(dz)
    \right\}


Just as we introduced the Bellman operator to solve the Bellman equation, we
will now introduce an operator over policies to help us solve the Euler
equation

This operator :math:`K` will act on the set of all :math:`\sigma \in \Sigma`
that are continuous, strictly increasing and interior (i.e., :math:`0 < \sigma(y) < y` for all strictly positive :math:`y`) 

Henceforth we denote this set of policies by :math:`\mathscr P`

#. The operator :math:`K` takes as its argument a :math:`\sigma \in \mathscr P` and

#. returns a new function :math:`K\sigma`,  where :math:`K\sigma(y)` is the :math:`c \in (0, y)` that solves

.. math::
    :label: cpi_coledef

    u'(c) 
    = \beta \int (u' \circ \sigma) (f(y - c) z ) f'(y - c) z \phi(dz)


We call this operator the **Coleman operator** to acknowledge the work of :cite:`Coleman1990`  (although many people have studied this and other closely related iterative techniques)

In essence, :math:`K\sigma` is the consumption policy that the Euler equation tells
you to choose today when your future consumption policy is :math:`\sigma`

The  important thing to note about :math:`K` is that, by
construction, its fixed points coincide with solutions to the functional
equation :eq:`cpi_euler_func`

In particular, the optimal policy :math:`c^*` is a fixed point

Indeed, for fixed :math:`y`, the value :math:`Kc^*(y)` is the :math:`c` that
solves

.. math::

    u'(c) 
    = \beta \int (u' \circ c^*) (f(y - c) z ) f'(y - c) z \phi(dz)


In view of the Euler equation, this is exactly :math:`c^*(y)`




Is the Coleman Operator Well Defined?
--------------------------------------

In particular, is there always a unique :math:`c \in (0, y)` that solves
:eq:`cpi_coledef`?

The answer is yes, under our assumptions

For any :math:`\sigma \in \mathscr P`, the right side of :eq:`cpi_coledef`  

* is continuous and strictly increasing in :math:`c` on :math:`(0, y)`

* diverges to :math:`+\infty` as :math:`c \uparrow y`


The left side of :eq:`cpi_coledef`  

* is continuous and strictly decreasing in :math:`c` on :math:`(0, y)`

* diverges to :math:`+\infty` as :math:`c \downarrow 0`


Sketching these curves and using the information above will convince you that they cross exactly once as :math:`c` ranges over :math:`(0, y)`

With a bit more analysis, one can show in addition that :math:`K \sigma \in \mathscr P`
whenever :math:`\sigma \in \mathscr P`



Comparison with Value Function Iteration
=========================================

How does Euler equation time iteration compare with value function iteration?

Both can be used to compute the optimal policy, but is one faster or more
accurate?

There are two parts to this story

First, on a theoretical level, the two methods are essentially isomorphic

In particular, they converge at  the same rate

We'll prove this in just a moment

The other side to the story is the speed of the numerical implementation

It turns out that, once we actually implement these two routines, time iteration is faster and more accurate than value function iteration

More on this below


Equivalent Dynamics
---------------------

Let's talk about the theory first

To explain the connection between the two algorithms, it helps to understand
the notion of equivalent dynamics

(This concept is very helpful in many other contexts as well)

Suppose that we have a function :math:`g \colon X \to X` where :math:`X` is a given set

The pair :math:`(X, g)` is sometimes called a **dynamical system** and we
associate it with trajectories of the form

.. math::

    x_{t+1} = g(x_t), \qquad x_0 \text{ given}


Equivalently, :math:`x_t = g^t(x_0)`, where :math:`g` is the :math:`t`-th
composition of :math:`g` with itself

Here's the picture

.. figure:: /_static/figures/col_pol_composition.png
    :scale: 40%

Now let another function :math:`h \colon Y \to Y` where :math:`Y` is another set



Suppose further that 

* there exists a bijection :math:`\tau` from :math:`X` to :math:`Y`

* the two functions **commute** under :math:`\tau`, which is to say that
  :math:`\tau(g(x)) = h (\tau(x))` for all :math:`x \in X`

The last statement can be written more simply as 

.. math::

    \tau \circ g = h \circ \tau


or, by applying :math:`\tau^{-1}` to both sides

.. math::
    :label: cpi_ghcom

    g = \tau^{-1} \circ h \circ \tau


Here's a commutative diagram that illustrates

.. figure:: /_static/figures/col_pol_bij1.png
    :scale: 20%


Here's a similar figure that traces out the action of the maps on a point
:math:`x \in X`

.. figure:: /_static/figures/col_pol_bij2.png
    :scale: 20%

Now, it's easy to check from :eq:`cpi_ghcom` that :math:`g^2 = \tau^{-1} \circ h^2 \circ \tau` holds

In fact, if you like proofs by induction, you won't have trouble showing that

.. math::

    g^n = \tau^{-1} \circ h^n \circ \tau


is valid for all :math:`n`

What does this tell us?

It tells us that the following are equivalent

* iterate :math:`n` times with :math:`g`, starting at :math:`x`

* shift :math:`x` to :math:`Y` using :math:`\tau`,  iterate :math:`n` times with :math:`h` starting at :math:`\tau(x)`, and shift the result :math:`h^n(\tau(x))` back to :math:`X` using :math:`\tau^{-1}`

We end up with exactly the same object


Back to Economics
--------------------

Have you guessed where this is leading?

What we're going to show now is that the operators :math:`T` and :math:`K`
commute under a certain bijection

The implication is that they have exactly the same rate of convergence

To make life a little easier, we'll assume in the following analysis (although not
always in our applications) that :math:`u(0) = 0`


A Bijection
^^^^^^^^^^^^^

Let :math:`\mathscr V` be all strictly concave, continuously differentiable functions :math:`v` mapping :math:`\mathbb R_+` to itself and satisfying :math:`v(0) = 0` and :math:`v'(y) > u'(y)` for all positive :math:`y` 

For :math:`v \in \mathscr V` let

.. math::
    M v := h \circ v' \qquad \text{where } h := (u')^{-1}


Although we omit details, :math:`\sigma := M v` is actually the unique
:math:`v`-greedy policy

* See proposition 12.1.18 of `EDTC <http://johnstachurski.net/edtc.html>`__

It turns out that :math:`M` is a bijection from :math:`\mathscr V` to :math:`\mathscr P` 

A (solved) exercise below asks you to confirm this


Commutative Operators
^^^^^^^^^^^^^^^^^^^^^^

It is an additional solved exercise (see below) to show that :math:`T` and :math:`K` commute under :math:`M`, in the sense that

.. math::
    :label: cpi_ed_tk

    M \circ T = K \circ M

In view of the preceding discussion, this implies that 

.. math::

    T^n = M^{-1} \circ K^n \circ M


Hence, :math:`T` and :math:`K` converge at exactly the same rate!




Implementation
================

We've just shown that the operators :math:`T` and :math:`K` have the same rate of convergence

However, it turns out that, once numerical approximation is taken into account, significant differences arises

In particular, the image of policy functions under :math:`K` can be calculated faster and with greater accuracy than the image of value functions under :math:`T`


Our intuition for this result is that

* the Coleman operator exploits more information because it uses first order and envelope conditions

* policy functions generally have less curvature than value functions, and hence admit more accurate approximations based on grid point information


The Operator
----------------


Here's some code that implements the Coleman operator

.. literalinclude:: /_static/code/coleman_policy_iter/coleman.py

It has some similarities to the code for the Bellman operator in our :doc:`optimal growth lecture <optgrowth>` 

For example, it evaluates integrals by Monte Carlo and approximates functions using linear interpolation

Here's that Bellman operator code again, which needs to be executed because we'll use it in some tests below

.. literalinclude:: /_static/code/optgrowth/optgrowth.py
    :class: collapse




Testing on the Log / Cobb--Douglas case
------------------------------------------


As we :doc:`did for value function iteration <optgrowth>`, let's start by
testing our method in the presence of a model that does have an analytical
solution



We assume the following imports
   
.. code-block:: ipython 

    import matplotlib.pyplot as plt
    %matplotlib inline
    import quantecon as qe







Now let's bring in the log-linear growth model we used in the :doc:`value function iteration lecture <optgrowth>`

.. literalinclude:: /_static/code/optgrowth/loglinear_og.py
    :class: collapse



Next we generate an instance


.. code-block:: python3 

    lg = LogLinearOG()

    # == Unpack parameters / functions for convenience == #
    α, β, μ, s = lg.α, lg.β, lg.μ, lg.s
    v_star, c_star = lg.v_star, lg.c_star
    u, u_prime, f, f_prime = lg.u, lg.u_prime, lg.f, lg.f_prime 






We also need a grid and some shock draws for Monte Carlo integration

.. code-block:: python3 

    grid_max = 4         # Largest grid point
    grid_size = 200      # Number of grid points
    shock_size = 250     # Number of shock draws in Monte Carlo integral
    
    grid = np.linspace(1e-5, grid_max, grid_size)
    shocks = np.exp(μ + s * np.random.randn(shock_size))



As a preliminary test, let's see if :math:`K c^* = c^*`, as implied by the
theory


.. code-block:: python3 

    c_star_new = coleman_operator(c_star(grid), 
                                  grid, β, u_prime, 
                                  f, f_prime, shocks)

    fig, ax = plt.subplots()
    ax.plot(grid, c_star(grid), label="optimal policy $c^*$")
    ax.plot(grid, c_star_new, label="$Kc^*$")

    ax.legend(loc='upper left')
    plt.show()



We can't really distinguish the two plots, so we are looking good, at least
for this test

Next let's try iterating from an arbitrary initial condition and see if we
converge towards :math:`c^*`


The initial condition we'll use is the one that eats the whole pie: :math:`c(y) = y`



.. code-block:: python3 

    g = grid
    n = 15
    fig, ax = plt.subplots(figsize=(9, 6))
    lb = 'initial condition $c(y) = y$'
    ax.plot(grid, g, color=plt.cm.jet(0), lw=2, alpha=0.6, label=lb)
    for i in range(n):
        new_g = coleman_operator(g, grid, β, u_prime, f, f_prime, shocks)
        g = new_g
        ax.plot(grid, g, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

    lb = 'true policy function $c^*$'
    ax.plot(grid, c_star(grid), 'k-', lw=2, alpha=0.8, label=lb)
    ax.legend(loc='upper left')

    plt.show()


We see that the policy has converged nicely, in only a few steps

Now let's compare the accuracy of iteration using the Coleman and Bellman operators


We'll generate 

#. :math:`K^n c` where :math:`c(y) = y`

#. :math:`(M \circ T^n \circ M^{-1}) c` where :math:`c(y) = y`

In each case we'll compare the resulting policy to :math:`c^*`

The theory on equivalent dynamics says we will get the same policy function
and hence the same errors

But in fact we expect the first method to be more accurate for reasons
discussed above



.. code-block:: python3 

    α, β, μ, s = lg.α, lg.β, lg.μ, lg.s
    v_star, c_star = lg.v_star, lg.c_star
    u, u_prime, f, f_prime = lg.u, lg.u_prime, lg.f, lg.f_prime

    g_init = grid
    w_init = u(grid)
    sim_length = 20

    g, w = g_init, w_init
    for i in range(sim_length):
        new_g = coleman_operator(g, grid, β, u_prime, f, f_prime, shocks)
        new_w = bellman_operator(w, grid, β, u, f, shocks)
        g, w = new_g, new_w

    new_w, vf_g = bellman_operator(w, grid, β, u, f, shocks, compute_policy=True)

    fig, ax = plt.subplots()

    pf_error = c_star(grid) - g
    vf_error = c_star(grid) - vf_g

    ax.plot(grid, 0 * grid, 'k-', lw=1)
    ax.plot(grid, pf_error, lw=2, alpha=0.6, label="policy iteration error")
    ax.plot(grid, vf_error, lw=2, alpha=0.6, label="value iteration error")

    ax.legend(loc='lower left')
    plt.show()



As you can see, time iteration is much more accurate for a given
number of iterations



Exercises
===========


Exercise 1
-----------

Show that :eq:`cpi_ed_tk` is valid.  In particular, 

* Let :math:`v` be strictly concave and continuously differentiable on :math:`(0, \infty)`

* Fix :math:`y \in (0, \infty)` and show that :math:`MTv(y) = KMv(y)`


Exercise 2
-----------

Show that :math:`M` is a bijection from :math:`\mathscr V` to :math:`\mathscr P` 


Exercise 3
------------

Consider the same model as above but with the CRRA utility function

.. math::

    u(c) = \frac{c^{1 - \gamma} - 1}{1 - \gamma}


Iterate 20 times with Bellman iteration and Euler equation time iteration

* start time iteration from :math:`c(y) = y`

* start value function iteration from :math:`v(y) = u(y)`

* set :math:`\gamma = 1.5`

Compare the resulting policies and check that they are close


Exercise 4
-----------


Do the same exercise, but now, rather than plotting results, time how long 20
iterations takes in each case



Solutions
===========


Solution to Exercise 1
-------------------------

Let :math:`T, K, M, v` and :math:`y` be as stated in the exercise

Using the envelope theorem, one can show that :math:`(Tv)'(y) = u'(c(y))`
where :math:`c(y)` solves

.. math::
    :label: cpi_foo

    u'(c(y)) 
    = \beta \int v' (f(y - c(y)) z ) f'(y - c(y)) z \phi(dz)


Hence :math:`MTv(y) = (u')^{-1} (u'(c(y))) = c(y)`

On the other hand, :math:`KMv(y)` is the :math:`c(y)` that solves


.. math::

    \begin{aligned}
        u'(c(y)) 
        & = \beta \int (u' \circ (Mv)) (f(y - c(y)) z ) f'(y - c(y)) z \phi(dz)
        \\
        & = \beta \int (u' \circ ((u')^{-1} \circ v')) 
            (f(y - c(y)) z ) f'(y - c(y)) z \phi(dz)
        \\
        & = \beta \int v'(f(y - c(y)) z ) f'(y - c(y)) z \phi(dz)
    \end{aligned}


We see that :math:`c(y)` is the same in each case


Solution to Exercise 2
-------------------------

We need to show that :math:`M` is a bijection from :math:`\mathscr V` to :math:`\mathscr P` 

To see this, first observe that, in view of our assumptions above, :math:`u'` is a strictly decreasing continuous bijection from :math:`(0,\infty)` to itself

It `follows <https://math.stackexchange.com/questions/672174/continuity-of-an-inverse-function>`__ that :math:`h` has the same properties

Moreover, for fixed :math:`v \in \mathscr V`, the derivative :math:`v'` is
a continuous, strictly decreasing function 

Hence, for fixed :math:`v \in \mathscr V`, the map :math:`M v = h \circ v'`
is strictly increasing and continuous, taking values in :math:`(0, \infty)`

Moreover, interiority holds because :math:`v'` strictly dominates :math:`u'`, implying that

.. math:: (M v)(y) = h(v'(y)) < h(u'(y)) = y

In particular, :math:`\sigma(y) := (Mv)(y)` is an element of :math:`\mathscr
P` 

To see that each :math:`\sigma \in \mathscr P` has a preimage :math:`v \in \mathscr V` with :math:`Mv = \sigma`, fix any :math:`\sigma \in \mathscr P`

Let :math:`v(y) := \int_0^y u'(\sigma(x)) dx` with :math:`v(0) = 0`

With a small amount of effort you will be able to show that :math:`v \in \mathscr V` and :math:`Mv = \sigma`

It's also true that :math:`M` is one-to-one on :math:`\mathscr V`

To see this, suppose that  :math:`v` and :math:`w` are elements of :math:`\mathscr V`
satisfying :math:`Mv = Mw` 

Then :math:`v(0) = w(0) = 0` and :math:`v' = w'` on :math:`(0, \infty)`

The fundamental theorem of calculus then implies that :math:`v = w` on :math:`\mathbb R_+`


Solution to Exercise 3
-------------------------



Here's the code, which will execute if you've run all the code above



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

    def f(k):
        return k**α

    def f_prime(k):
        return α * k**(α - 1)

    def u(c):
        return (c**(1 - γ) - 1) / (1 - γ)

    def u_prime(c):
        return c**(-γ)

    grid = np.linspace(grid_min, grid_max, grid_size)
    shocks = np.exp(μ + s * np.random.randn(shock_size))

    ## Let's make convenience functions based around these primitives

    def crra_bellman(w):
        return bellman_operator(w, grid, β, u, f, shocks)

    def crra_coleman(g):
        return coleman_operator(g, grid, β, u_prime, f, f_prime, shocks)

    ## Iterate with K and T, compare policies

    g_init = grid
    w_init = u(grid)
    sim_length = 20

    g, w = g_init, w_init
    for i in range(sim_length):
        new_g = crra_coleman(g)
        new_w = crra_bellman(w)
        g, w = new_g, new_w

    new_w, vf_g = bellman_operator(w, grid, β, u, f, shocks, compute_policy=True)

    fig, ax = plt.subplots()

    ax.plot(grid, g, lw=2, alpha=0.6, label="policy iteration")
    ax.plot(grid, vf_g, lw=2, alpha=0.6, label="value iteration")

    ax.legend(loc="upper left")
    plt.show()



The policies are indeed close




Solution to Exercise 4
-------------------------


Here's the code

It assumes that you've just run the code from the previous exercise




.. code-block:: python3 

    g_init = grid
    w_init = u(grid)
    sim_length = 100

    print("Timing value function iteration")

    w = w_init
    qe.util.tic()
    for i in range(sim_length):
        new_w = crra_bellman(w)
        w = new_w
    qe.util.toc()


    print("Timing Euler equation time iteration")

    g = g_init
    qe.util.tic()
    for i in range(sim_length):
        new_g = crra_coleman(g)
        g = new_g
    qe.util.toc()

If you run this you'll find that the two operators execute at about the same speed

However, as we saw above, time iteration is numerically far more accurate for a given number of iterations



