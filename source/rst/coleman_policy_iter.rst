.. _coleman_policy_iter:

.. include:: /_static/includes/header.raw

.. highlight:: python3

*********************************************
:index:`Optimal Growth II: Time Iteration`
*********************************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon
  !pip install interpolation


Overview
============

In this lecture, we'll continue our :doc:`earlier study <optgrowth>` of the stochastic optimal growth model.

In that lecture, we solved the associated discounted dynamic programming problem using value function iteration.

The beauty of this technique is its broad applicability.

With numerical problems, however, we can often attain higher efficiency in specific
applications by deriving methods that are carefully tailored to the application at hand.

The stochastic optimal growth model has plenty of structure to exploit for this purpose,
especially when we adopt some concavity and smoothness assumptions over primitives.

We'll use this structure to obtain an **Euler equation**  based method that's more efficient
than value function iteration for this and some other closely related applications.

In a :doc:`subsequent lecture <egm_policy_iter>`, we'll see that the numerical implementation
part of the Euler equation method can be further adjusted to obtain even more efficiency.

Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt
    %matplotlib inline
    from quantecon.optimize import brentq


The Euler Equation
==========================

Let's take the model set out in :doc:`the stochastic growth model lecture <optgrowth>` and add the assumptions that

#. :math:`u` and :math:`f` are continuously differentiable and strictly concave

#. :math:`f(0) = 0`

#. :math:`\lim_{c \to 0} u'(c) = \infty` and :math:`\lim_{c \to \infty} u'(c) = 0`

#. :math:`\lim_{k \to 0} f'(k) = \infty` and :math:`\lim_{k \to \infty} f'(k) = 0`

The last two conditions are usually called **Inada conditions**.


Recall the Bellman equation

.. math::
    :label: cpi_fpb30

    v^*(y) = \max_{0 \leq c \leq y}
        \left\{
            u(c) + \beta \int v^*(f(y - c) z) \phi(dz)
        \right\}
    \quad \text{for all} \quad
    y \in \mathbb R_+


Let the optimal consumption policy be denoted by :math:`\sigma^*`.

We know that :math:`\sigma^*` is a :math:`v^*` greedy policy so that :math:`\sigma^*(y)` is the maximizer in :eq:`cpi_fpb30`.

The conditions above imply that

* :math:`\sigma^*` is the unique optimal policy for the stochastic optimal growth model

* the optimal policy is continuous, strictly increasing and also **interior**, in the sense that :math:`0 < \sigma^*(y) < y` for all strictly positive :math:`y`, and

* the value function is strictly concave and continuously differentiable, with

.. math::
    :label: cpi_env

    (v^*)'(y) = u' (\sigma^*(y) ) := (u' \circ \sigma^*)(y)

The last result is called the **envelope condition** due to its relationship with the `envelope theorem <https://en.wikipedia.org/wiki/Envelope_theorem>`_.

To see why :eq:`cpi_env` might be valid, write the Bellman equation in the equivalent
form

.. math::

    v^*(y) = \max_{0 \leq k \leq y}
        \left\{
            u(y-k) + \beta \int v^*(f(k) z) \phi(dz)
        \right\},


differentiate naively with respect to :math:`y`,  and then  evaluate at the optimum.

Section 12.1 of `EDTC <http://johnstachurski.net/edtc.html>`_ contains full proofs of these results, and closely related discussions can be found in many other texts.


Differentiability of the value function and interiority of the optimal policy
imply that optimal consumption satisfies the first order condition associated
with :eq:`cpi_fpb30`, which is

.. math::
    :label: cpi_foc

    u'(\sigma^*(y)) = \beta \int (v^*)'(f(y - \sigma^*(y)) z) f'(y - \sigma^*(y)) z \phi(dz)


Combining :eq:`cpi_env` and the first-order condition :eq:`cpi_foc` gives the famous **Euler equation**

.. math::
    :label: cpi_euler

    (u'\circ \sigma^*)(y)
    = \beta \int (u'\circ \sigma^*)(f(y - \sigma^*(y)) z) f'(y - \sigma^*(y)) z \phi(dz)


We can think of the Euler equation as a functional equation

.. math::
    :label: cpi_euler_func

    (u'\circ \sigma)(y)
    = \beta \int (u'\circ \sigma)(f(y - \sigma(y)) z) f'(y - \sigma(y)) z \phi(dz)


over interior consumption policies :math:`\sigma`, one solution of which is the optimal policy :math:`\sigma^*`.

Our aim is to solve the functional equation :eq:`cpi_euler_func` and hence obtain :math:`\sigma^*`.



The Coleman-Reffett Operator
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
equation.

This operator :math:`K` will act on the set of all :math:`\sigma \in \Sigma`
that are continuous, strictly increasing and interior (i.e., :math:`0 < \sigma(y) < y` for all strictly positive :math:`y`).

Henceforth we denote this set of policies by :math:`\mathscr P`

#. The operator :math:`K` takes as its argument a :math:`\sigma \in \mathscr P` and

#. returns a new function :math:`K\sigma`,  where :math:`K\sigma(y)` is the :math:`c \in (0, y)` that solves.

.. math::
    :label: cpi_coledef

    u'(c)
    = \beta \int (u' \circ \sigma) (f(y - c) z ) f'(y - c) z \phi(dz)


We call this operator the **Coleman-Reffett operator** to acknowledge the work of
:cite:`Coleman1990` and :cite:`Reffett1996`.

In essence, :math:`K\sigma` is the consumption policy that the Euler equation tells
you to choose today when your future consumption policy is :math:`\sigma`.

The  important thing to note about :math:`K` is that, by
construction, its fixed points coincide with solutions to the functional
equation :eq:`cpi_euler_func`.

In particular, the optimal policy :math:`\sigma^*` is a fixed point.

Indeed, for fixed :math:`y`, the value :math:`K\sigma^*(y)` is the :math:`c` that
solves

.. math::

    u'(c)
    = \beta \int (u' \circ \sigma^*) (f(y - c) z ) f'(y - c) z \phi(dz)


In view of the Euler equation, this is exactly :math:`\sigma^*(y)`.




Is the Coleman-Reffett Operator Well Defined?
-----------------------------------------------

In particular, is there always a unique :math:`c \in (0, y)` that solves
:eq:`cpi_coledef`?

The answer is yes, under our assumptions.

For any :math:`\sigma \in \mathscr P`, the right side of :eq:`cpi_coledef`

* is continuous and strictly increasing in :math:`c` on :math:`(0, y)`

* diverges to :math:`+\infty` as :math:`c \uparrow y`


The left side of :eq:`cpi_coledef`

* is continuous and strictly decreasing in :math:`c` on :math:`(0, y)`

* diverges to :math:`+\infty` as :math:`c \downarrow 0`


Sketching these curves and using the information above will convince you that they cross exactly once as :math:`c` ranges over :math:`(0, y)`.

With a bit more analysis, one can show in addition that :math:`K \sigma \in \mathscr P`
whenever :math:`\sigma \in \mathscr P`.



Comparison with Value Function Iteration
=========================================

How does Euler equation time iteration compare with value function iteration?

Both can be used to compute the optimal policy, but is one faster or more
accurate?

There are two parts to this story.

First, on a theoretical level, the two methods are essentially isomorphic.

In particular, they converge at  the same rate.

We'll prove this in just a moment.

The other side of the story is the accuracy of the numerical implementation.

It turns out that, once we actually implement these two routines, time iteration is more accurate than value function iteration.

More on this below.


Equivalent Dynamics
---------------------

Let's talk about the theory first.

To explain the connection between the two algorithms, it helps to understand
the notion of equivalent dynamics.

(This concept is very helpful in many other contexts as well)

Suppose that we have a function :math:`g \colon X \to X` where :math:`X` is a given set.

The pair :math:`(X, g)` is sometimes called a **dynamical system** and we
associate it with trajectories of the form

.. math::

    x_{t+1} = g(x_t), \qquad x_0 \text{ given}


Equivalently, :math:`x_t = g^t(x_0)`, where :math:`g` is the :math:`t`-th
composition of :math:`g` with itself.

Here's the picture

.. figure:: /_static/lecture_specific/coleman_policy_iter/col_pol_composition.png

Now let another function :math:`h \colon Y \to Y` where :math:`Y` is another set.



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

.. figure:: /_static/lecture_specific/coleman_policy_iter/col_pol_bij1.png


Here's a similar figure that traces out the action of the maps on a point
:math:`x \in X`

.. figure:: /_static/lecture_specific/coleman_policy_iter/col_pol_bij2.png

Now, it's easy to check from :eq:`cpi_ghcom` that :math:`g^2 = \tau^{-1} \circ h^2 \circ \tau` holds.

In fact, if you like proofs by induction, you won't have trouble showing that

.. math::

    g^n = \tau^{-1} \circ h^n \circ \tau


is valid for all :math:`n`.

What does this tell us?

It tells us that the following are equivalent

* iterate :math:`n` times with :math:`g`, starting at :math:`x`

* shift :math:`x` to :math:`Y` using :math:`\tau`,  iterate :math:`n` times with :math:`h` starting at :math:`\tau(x)` and shift the result :math:`h^n(\tau(x))` back to :math:`X` using :math:`\tau^{-1}`

We end up with exactly the same object.


Back to Economics
--------------------

Have you guessed where this is leading?

What we're going to show now is that the operators :math:`T` and :math:`K`
commute under a certain bijection.

The implication is that they have exactly the same rate of convergence.

To make life a little easier, we'll assume in the following analysis (although not
always in our applications) that :math:`u(0) = 0`.


A Bijection
^^^^^^^^^^^^^

Let :math:`\mathscr V` be all strictly concave, continuously differentiable functions :math:`v` mapping :math:`\mathbb R_+` to itself and satisfying :math:`v(0) = 0` and :math:`v'(y) > u'(y)` for all positive :math:`y`.

For :math:`v \in \mathscr V` let

.. math::
    M v := h \circ v' \qquad \text{where } h := (u')^{-1}


Although we omit details, :math:`\sigma := M v` is actually the unique
:math:`v`-greedy policy.

* See proposition 12.1.18 of `EDTC <http://johnstachurski.net/edtc.html>`__.

It turns out that :math:`M` is a bijection from :math:`\mathscr V` to :math:`\mathscr P`.

A (solved) exercise below asks you to confirm this.


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

We've just shown that the operators :math:`T` and :math:`K` have the same rate of convergence.

However, it turns out that, once numerical approximation is taken into account, significant differences arise.

In particular, the image of policy functions under :math:`K` can be calculated faster and with greater accuracy than the image of value functions under :math:`T`.


Our intuition for this result is that

* the Coleman-Reffett operator exploits more information because it uses first order and envelope conditions

* policy functions generally have less curvature than value functions, and hence admit more accurate approximations based on grid point information

First, we'll store the parameters of the model in a class ``OptimalGrowthModel``

.. code-block:: python3

    class OptimalGrowthModel:

        def __init__(self,
                     f,
                     f_prime,
                     u,
                     u_prime,
                     β=0.96,
                     μ=0,
                     s=0.1,
                     grid_max=4,
                     grid_size=200,
                     shock_size=250):

            self.β, self.μ, self.s = β, μ, s
            self.f, self.u = f, u
            self.f_prime, self.u_prime = f_prime, u_prime

            self.grid = np.linspace(1e-5, grid_max, grid_size)  # Set up grid
            # Store shocks
            self.shocks = np.exp(μ + s * np.random.randn(shock_size))


Here's some code that returns the Coleman-Reffett operator, :math:`K`.

.. literalinclude:: /_static/lecture_specific/coleman_policy_iter/coleman_operator.py

It has some similarities to the code for the Bellman operator in our :doc:`optimal growth lecture <optgrowth>`.

For example, it evaluates integrals by Monte Carlo and approximates functions using linear interpolation.

Here's that Bellman operator code again, which needs to be executed because we'll use it in some tests below.

.. literalinclude:: /_static/lecture_specific/optgrowth/bellman_operator.py
    :class: collapse




Testing on the Log / Cobb--Douglas Case
------------------------------------------


As we :doc:`did for value function iteration <optgrowth>`, let's start by
testing our method in the presence of a model that does have an analytical
solution.


First, we generate an instance of ``OptimalGrowthModel`` and return the corresponding
Coleman-Reffett operator.


.. code-block:: python3

    α = 0.3

    @njit
    def f(k):
        "Deterministic part of production function"
        return k**α

    @njit
    def f_prime(k):
        return α * k**(α - 1)

    og = OptimalGrowthModel(f=f, f_prime=f_prime,
                            u=np.log, u_prime=njit(lambda x: 1/x))

    K = time_operator_factory(og)


As a preliminary test, let's see if :math:`K \sigma^* = \sigma^*`, as implied by the
theory.


.. code-block:: python3

    @njit
    def σ_star(y, α, β):
        "True optimal policy"
        return (1 - α * β) * y

    grid, β = og.grid, og.β
    σ_star_new = K(σ_star(grid, α, β))

    fig, ax = plt.subplots()
    ax.plot(grid, σ_star(grid, α, β), label="optimal policy $\sigma^*$")
    ax.plot(grid, σ_star_new, label="$K\sigma^*$")

    ax.legend()
    plt.show()



We can't really distinguish the two plots, so we are looking good, at least
for this test.

Next, let's try iterating from an arbitrary initial condition and see if we
converge towards :math:`\sigma^*`.


The initial condition we'll use is the one that eats the whole pie: :math:`\sigma(y) = y`.



.. code-block:: python3

    n = 15
    σ = grid.copy()  # Set initial condition
    fig, ax = plt.subplots(figsize=(9, 6))
    lb = 'initial condition $\sigma(y) = y$'
    ax.plot(grid, σ, color=plt.cm.jet(0), alpha=0.6, label=lb)

    for i in range(n):
        σ = K(σ)
        ax.plot(grid, σ, color=plt.cm.jet(i / n), alpha=0.6)

    lb = 'true policy function $\sigma^*$'
    ax.plot(grid, σ_star(grid, α, β), 'k-', alpha=0.8, label=lb)
    ax.legend()

    plt.show()


We see that the policy has converged nicely, in only a few steps.

Now let's compare the accuracy of iteration between the operators.

We'll generate

#. :math:`K^n \sigma` where :math:`\sigma(y) = y`

#. :math:`(M \circ T^n \circ M^{-1}) \sigma` where :math:`\sigma(y) = y`

In each case, we'll compare the resulting policy to :math:`\sigma^*`.

The theory on equivalent dynamics says we will get the same policy function
and hence the same errors.

But in fact we expect the first method to be more accurate for reasons
discussed above



.. code-block:: python3

    T, get_greedy = operator_factory(og)  # Return the Bellman operator

    σ = grid          # Set initial condition for σ
    v = og.u(grid)    # Set initial condition for v
    sim_length = 20

    for i in range(sim_length):
        σ = K(σ)  # Time iteration
        v = T(v)  # Value function iteration

    # Calculate difference with actual solution
    σ_error = σ_star(grid, α, β) - σ
    v_error = σ_star(grid, α, β) - get_greedy(v)

    plt.plot(grid, σ_error, alpha=0.6, label="policy iteration error")
    plt.plot(grid, v_error, alpha=0.6, label="value iteration error")
    plt.legend()
    plt.show()


As you can see, time iteration is much more accurate for a given
number of iterations.



Exercises
===========


Exercise 1
-----------

Show that :eq:`cpi_ed_tk` is valid.  In particular,

* Let :math:`v` be strictly concave and continuously differentiable on :math:`(0, \infty)`.

* Fix :math:`y \in (0, \infty)` and show that :math:`MTv(y) = KMv(y)`.


Exercise 2
-----------

Show that :math:`M` is a bijection from :math:`\mathscr V` to :math:`\mathscr P`.


Exercise 3
------------

Consider the same model as above but with the CRRA utility function

.. math::

    u(c) = \frac{c^{1 - \gamma} - 1}{1 - \gamma}


Iterate 20 times with Bellman iteration and Euler equation time iteration

* start time iteration from :math:`\sigma(y) = y`

* start value function iteration from :math:`v(y) = u(y)`

* set :math:`\gamma = 1.5`

Compare the resulting policies and check that they are close.

Exercise 4
-----------


Solve the above model as we did in :doc:`the previous lecture <optgrowth>` using
the operators :math:`T` and :math:`K`, and check the solutions are similiar by plotting.



Solutions
===========


Exercise 1
-------------------------

Let :math:`T, K, M, v` and :math:`y` be as stated in the exercise.

Using the envelope theorem, one can show that :math:`(Tv)'(y) = u'(\sigma(y))`
where :math:`\sigma(y)` solves

.. math::
    :label: cpi_foo

    u'(\sigma(y))
    = \beta \int v' (f(y - \sigma(y)) z ) f'(y - \sigma(y)) z \phi(dz)


Hence :math:`MTv(y) = (u')^{-1} (u'(\sigma(y))) = \sigma(y)`.

On the other hand, :math:`KMv(y)` is the :math:`\sigma(y)` that solves


.. math::

    \begin{aligned}
        u'(\sigma(y))
        & = \beta \int (u' \circ (Mv)) (f(y - \sigma(y)) z ) f'(y - \sigma(y)) z \phi(dz)
        \\
        & = \beta \int (u' \circ ((u')^{-1} \circ v'))
            (f(y - \sigma(y)) z ) f'(y - \sigma(y)) z \phi(dz)
        \\
        & = \beta \int v'(f(y - \sigma(y)) z ) f'(y - \sigma(y)) z \phi(dz)
    \end{aligned}


We see that :math:`\sigma(y)` is the same in each case.


Exercise 2
-------------------------

We need to show that :math:`M` is a bijection from :math:`\mathscr V` to :math:`\mathscr P`.

To see this, first observe that, in view of our assumptions above, :math:`u'` is a strictly decreasing continuous bijection from :math:`(0,\infty)` to itself.

It `follows <https://math.stackexchange.com/questions/672174/continuity-of-an-inverse-function>`__ that :math:`h` has the same properties.

Moreover, for fixed :math:`v \in \mathscr V`, the derivative :math:`v'` is
a continuous, strictly decreasing function.

Hence, for fixed :math:`v \in \mathscr V`, the map :math:`M v = h \circ v'`
is strictly increasing and continuous, taking values in :math:`(0, \infty)`.

Moreover, interiority holds because :math:`v'` strictly dominates :math:`u'`, implying that

.. math:: (M v)(y) = h(v'(y)) < h(u'(y)) = y

In particular, :math:`\sigma(y) := (Mv)(y)` is an element of :math:`\mathscr
P`.

To see that each :math:`\sigma \in \mathscr P` has a preimage :math:`v \in \mathscr V` with :math:`Mv = \sigma`, fix any :math:`\sigma \in \mathscr P`.

Let :math:`v(y) := \int_0^y u'(\sigma(x)) dx` with :math:`v(0) = 0`.

With a small amount of effort, you will be able to show that :math:`v \in \mathscr V` and :math:`Mv = \sigma`.

It's also true that :math:`M` is one-to-one on :math:`\mathscr V`.

To see this, suppose that  :math:`v` and :math:`w` are elements of :math:`\mathscr V`
satisfying :math:`Mv = Mw`.

Then :math:`v(0) = w(0) = 0` and :math:`v' = w'` on :math:`(0, \infty)`.

The fundamental theorem of calculus then implies that :math:`v = w` on :math:`\mathbb R_+`.


Exercise 3
-------------------------

Here's the code, which will execute if you've run all the code above



.. code-block:: python3

    γ = 1.5   # Preference parameter

    @njit
    def u(c):
        return (c**(1 - γ) - 1) / (1 - γ)

    @njit
    def u_prime(c):
        return c**(-γ)

    og = OptimalGrowthModel(f=f, f_prime=f_prime, u=u, u_prime=u_prime)

    T, get_greedy = operator_factory(og)
    K = time_operator_factory(og)

    σ = grid        # Initial condition for σ
    v = u(grid)     # Initial condition for v
    sim_length = 20

    for i in range(sim_length):
        σ = K(σ)  # Time iteration
        v = T(v)  # Value function iteration


    plt.plot(grid, σ, alpha=0.6, label="policy iteration")
    plt.plot(grid, get_greedy(v), alpha=0.6, label="value iteration")
    plt.legend()
    plt.show()

The policies are indeed close.


Exercise 4
-------------------------


Here's is the function we need to solve the model using value function iteration,
copied from the previous lecture

.. literalinclude:: /_static/lecture_specific/optgrowth/solve_model.py

Similarly, we can write a function that uses ``K`` to solve the model.

.. code-block:: python3

    def solve_model_time(og,
                         use_parallel=True,
                         tol=1e-4,
                         max_iter=1000,
                         verbose=True,
                         print_skip=25):

        K = time_operator_factory(og, parallel_flag=use_parallel)

        # Set up loop
        σ = og.grid  # Initial condition
        i = 0
        error = tol + 1

        while i < max_iter and error > tol:
            σ_new = K(σ)
            error = np.max(np.abs(σ - σ_new))
            i += 1
            if verbose and i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")
            σ = σ_new

        if i == max_iter:
            print("Failed to converge!")

        if verbose and i < max_iter:
            print(f"\nConverged in {i} iterations.")

        return σ_new

Solving both models and plotting

.. code-block:: ipython

    v_star = solve_model(og)
    σ_star = solve_model_time(og)

    plt.plot(grid, get_greedy(v_star), alpha=0.6, label='Bellman operator')
    plt.plot(grid, σ_star, alpha=0.6, label='Coleman-Reffett operator')
    plt.legend()
    plt.show()



Time iteration is numerically far more accurate for a given number of iterations.
