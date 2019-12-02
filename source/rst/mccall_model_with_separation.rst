.. _mccall_with_sep:

.. include:: /_static/includes/header.raw

.. highlight:: python3


************************************
Job Search II: Search and Separation
************************************

.. index::
    single: An Introduction to Job Search

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
========


Previously :doc:`we looked <mccall_model>` at the McCall job search model :cite:`McCall1970` as a way of understanding unemployment and worker decisions.

One unrealistic feature of the model is that every job is permanent.

In this lecture, we extend the McCall model by introducing job separation.

Once separation enters the picture, the agent comes to view

*  the loss of a job as a capital loss, and

*  a spell of unemployment as an *investment* in searching for an acceptable job

The other minor addition is that a utility function will be included to make
worker preferences slightly more sophisticated.

We'll need the following imports

.. code-block:: ipython

    import numpy as np
    from quantecon.distributions import BetaBinomial
    from numba import njit
    import matplotlib.pyplot as plt
    %matplotlib inline

The Model
=========

The model is similar to that described in :doc:`our lecture <mccall_model>` on
the McCall job search model.

It concerns the life of an infinitely lived worker and

* the opportunities he or she (let's say he to save one character) has to work at different wages

* exogenous events that destroy his current job

* his decision making process while unemployed

The worker can be in one of two states: employed or unemployed.

He wants to maximize

.. math::
    :label: objective

    {\mathbb E} \sum_{t=0}^\infty \beta^t u(y_t)

At this stage the only difference from the :doc:`baseline model
<mccall_model>` is that we've added some flexibility to preferences by
introducing a utility function :math:`u`.

It satisfies :math:`u'> 0` and :math:`u'' < 0`.


The Wage Process
----------------

For now we will drop the separation of state process and wage process that we
maintained for the :doc:`baseline model <mccall_model>`. 

In particular, we simply suppose that wage offers :math:`\{ w_t \}` are IID with common distribution :math:`q`.

The set of possible wage values is denoted by :math:`\mathbb W`.

(Later we will go back to having a separate state process :math:`\{s_t\}`
driving random outcomes, since this formulation is usually convenient in more sophisticated
models.)



Timing and Decisions
--------------------

At the start of each period, the agent can be either

* unemployed or
* employed at some existing wage level :math:`w_e`.

At the start of a given period, the current wage offer :math:`w = w_t` is observed.

If currently *employed*, the worker 

1. consumes his wage :math:`w_e`, receiving utility :math:`u(w_e)`, and then
2. is fired with probability :math:`\alpha`.

If currently *unemployed*, the worker either accepts or rejects the current offer 

If he accepts, then he begins work immediately at wage :math:`w_t`.

If he rejects, then he receives unemployment compensation :math:`c`.

(Note: we do not allow for job search while employed---this topic is taken up in a :doc:`later lecture <jv>`)





Solving the Model 
=================

Let

* :math:`v(w_e)` be total lifetime value accruing to a worker who enters the current period *employed* with existing wage :math:`w_e`

* :math:`h(w)` be total lifetime value accruing to a worker who is *unemployed* this period and observes current wage :math:`w`.

Here *value* means the value of the objective function :eq:`objective` when the worker makes optimal decisions at all future points in time.

Our first aim is to obtain these functions.

The Bellman Equations
---------------------

Suppose for now that the worker can calculate the functions :math:`v` and :math:`h` and use them in his decision making.

Then :math:`v` and :math:`h`  should satisfy

.. math::
    :label: bell1_mccall

    v(w_e) = u(w_e) + \beta 
        \left[
            (1-\alpha)v(w_e) + \alpha \sum_{w' \in \mathbb W} h(w') q(w') 
        \right]

and

.. math::
    :label: bell2_mccall

    h(w) = \max \left\{ v(w), \,  u(c) + \beta \sum_{w' \in \mathbb W} h(w') q(w') \right\} 


Equation :eq:`bell1_mccall` expresses the value of being employed at wage :math:`w_e` in terms of 

* current reward :math:`u(w_e)` plus
* discounted expected reward tomorrow, given the :math:`\alpha` probability of being fired 

Equation :eq:`bell2_mccall` expresses the value of being unemployed at current
wage :math:`w` as a maximum over the value of two options: accept or reject
the current offer :math:`w(s)`.

Accepting transitions the worker to employment and hence yields current reward :math:`v(w)`.

Rejecting leads to unemployment compensation and unemployment tomorrow.

Equations :eq:`bell1_mccall` and :eq:`bell2_mccall` are the Bellman equations for this model.

They provide enough information to solve out for both :math:`v` and :math:`h`.


A Simplifying Transformation
----------------------------

Rather than jumping straight into solving these equations, let's see if we can
simplify them somewhat.

(This process will be analogous to our :ref:`second pass <mm_op2>` at the plain vanila
McCall model, where we simplified the Bellman equation.)

First, let 

.. math::
    :label: defd_mm

    d := \sum_{w' \in \mathbb W} h(w') q(w')

be the expected value of unemployment tomorrow.

We can now write :eq:`bell2_mccall` as 

.. math::

    h(w) = \max \left\{ v(w), \,  u(c) + \beta d \right\} 

or, shifting time forward one period 

.. math::

     \sum_{w' \in \mathbb W} h(w') q(w')
      = \sum_{w' \in \mathbb W} \max \left\{ v(w'), \,  u(c) + \beta d \right\} q(w')


Using :eq:`defd_mm` again now gives

.. math::
    :label: bell02_mccall

      d = \sum_{w' \in \mathbb W} \max \left\{ v(w'), \,  u(c) + \beta d \right\} q(w')

Finally, :eq:`bell1_mccall` can now be rewritten as

.. math::
    :label: bell01_mccall

    v(w) = u(w) + \beta 
        \left[
            (1-\alpha)v(w) + \alpha d  
        \right]

In the last expression, we wrote :math:`w_e` as :math:`w` to make the notation
simpler.


The Reservation Wage
--------------------

Suppose we can use :eq:`bell02_mccall` and :math:`bell01_mccall` to solve for
:math:`d` and :math:`v`.

(We will see how to do this soon.)

We can then determine optimal behavior for the worker.

From :eq:`bell2_mccall`, we see that an unemployed agent accepts current offer
:math:`w` if `v(w) \geq  u(c) + \beta d`.

This means precisely that the value of accepting is higher than the expected
value of rejecting.

It is clear that :math:`v` is (at least weakly) increasing in :math:`w`, since the agent is
never made worse off by a higher wage offer.

Hence, we can express the optimal choice as accepting wage offer :math:`w` if
and only if

.. math::

     w \geq \bar w 
     \quad \text{where} \quad
     \bar w \text{ solves } v(\bar w) =  u(c) + \beta d 


Solving the Bellman Equations
-----------------------------


We'll use the same iterative approach to solving the Bellman equations that we
adopted in the :doc:`first job search lecture <mccall_model>`.

Here this amounts to

#. make guesses for :math:`d` and :math:`v`

#. plug these guesses into the right-hand sides of :eq:`bell02_mccall` and :eq:`bell01_mccall`

#. update the left-hand sides from this rule and then repeat

In other words, we are iterating using the rules


.. math::
    :label: bell1001

      d_{n+1} = \sum_{w' \in \mathbb W} 
        \max \left\{ v(w'), \,  u(c) + \beta d_n \right\} q(w')

.. math::
    :label: bell2001

    v_{n+1}(w) = u(w) + \beta 
        \left[
            (1-\alpha)v_n(w) + \alpha d_n  
        \right]

starting from some initial conditions :math:`d_0, v_0`.

As before, the system always converges to the true solutions---in this case,
the :math:`v` and :math:`d` that solve :eq:`bell02_mccall` and :eq:`bell01_mccall`.

A proof can be obtained via the Banach contraction mapping theorem.




Implementation
==============

Let's implement this iterative process.

In the code, you'll see that we use a class to store the various parameters and other
objects associated with a given model.

This helps to tidy up the code and provides an object that's easy to pass to functions.

The default utility function is a CRRA utility function

.. code-block:: python3

    # A default utility function

    @njit
    def u(c, σ):
        if c > 0:
            return (c**(1 - σ) - 1) / (1 - σ)
        else:
            return -10e6

    class McCallModel:
        """
        Stores the parameters and functions associated with a given model.
        """

        def __init__(self,
                     α=0.2,        # Job separation rate
                     β=0.98,       # Discount factor
                     γ=0.7,        # Job offer rate
                     c=6.0,        # Unemployment compensation
                     σ=2.0,        # Utility parameter
                     w_vals=None,  # Possible wage values
                     ϕ_vals=None): # Probabilities over w_vals

            self.α, self.β, self.γ, self.c = α, β, γ, c
            self.σ = σ

            # Add a default wage vector and probabilities over the vector using
            # the beta-binomial distribution
            if w_vals is None:
                n = 60  # number of possible outcomes for wage
                self.w_vals = np.linspace(10, 20, n)  # Wages between 10 and 20
                a, b = 600, 400  # shape parameters
                dist = BetaBinomial(n-1, a, b)
                self.ϕ_vals = dist.pdf()
            else:
                self.w_vals = w_vals
                self.ϕ_vals = ϕ_vals


The following defines jitted versions of the Bellman operators :math:`h` and :math:`v`

.. code-block:: python3

    @njit
    def Q(v, h, paras):
        """
        A jitted function to update the Bellman equations

        """

        α, β, γ, c, σ, w_vals, ϕ_vals = paras

        v_new = np.empty_like(v)

        for i in range(len(w_vals)):
            w = w_vals[i]
            v_new[i] = u(w, σ) + β * ((1 - α) * v[i] + α * h)

        h_new = u(c, σ) + β * (1 - γ) * h + \
                        β * γ * np.sum(np.maximum(h, v) * ϕ_vals)

        return v_new, h_new


The approach is to iterate until successive iterates are closer together than some small tolerance level.

We then return the current iterate as an approximate solution

.. code-block:: python3

    def solve_model(mcm, tol=1e-5, max_iter=2000):
        """
        Iterates to convergence on the Bellman equations

        mcm is an instance of McCallModel
        """

        v = np.ones_like(mcm.w_vals)   # Initial guess of v
        h = 1                          # Initial guess of h
        i = 0
        error = tol + 1

        while error > tol and i < max_iter:
            v_new, h_new = Q(v, h, (mcm.α, mcm.β, mcm.γ, mcm.c, mcm.σ, \
                                    mcm.w_vals, mcm.ϕ_vals)
                            )
            error_1 = np.max(np.abs(v_new - v))
            error_2 = np.abs(h_new - h)
            error = max(error_1, error_2)
            v = v_new
            h = h_new
            i += 1

        return v, h

Let's plot the approximate solutions :math:`v` and :math:`h` to see what they look like.

We'll use the default parameterizations found in the code above


.. code-block:: python3

    mcm = McCallModel()
    v, h = solve_model(mcm)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(mcm.w_vals, v, 'b-', lw=2, alpha=0.7, label='$v$')
    ax.plot(mcm.w_vals, [h] * len(mcm.w_vals), 
            'g-', lw=2, alpha=0.7, label='$h$')
    ax.set_xlim(min(mcm.w_vals), max(mcm.w_vals))
    ax.legend()
    ax.grid()

    plt.show()


The value :math:`v` is increasing because higher :math:`w` generates a higher wage flow conditional on staying employed.



Here's a function ``compute_reservation_wage`` that takes an instance of ``McCallModel``
and returns the reservation wage associated with a given model.

It uses `np.searchsorted <https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html>`__
to obtain the first :math:`w` in the set of possible wages such that :math:`v(w) > h`.

If :math:`v(w) < h` for all :math:`w`, then the function returns `np.inf`

.. code-block:: python3

    def compute_reservation_wage(mcm, return_values=False):
        """
        Computes the reservation wage of an instance of the McCall model
        by finding the smallest w such that v(w) > h.

        If v(w) > h for all w, then the reservation wage w_bar is set to
        the lowest wage in mcm.w_vals.

        If v(w) < h for all w, then w_bar is set to np.inf.
        """

        v, h = solve_model(mcm)
        w_idx = np.searchsorted(v - h, 0)

        if w_idx == len(v):
            w_bar = np.inf
        else:
            w_bar = mcm.w_vals[w_idx]

        if not return_values:
            return w_bar
        else:
            return w_bar, v, h

Let's use it to look at how the reservation wage varies with parameters.

In each instance below, we'll show you a figure and then ask you to reproduce it in the exercises.




The Reservation Wage and Unemployment Compensation
--------------------------------------------------

First, let's look at how :math:`\bar w` varies with unemployment compensation.

In the figure below, we use the default parameters in the ``McCallModel`` class, apart from
`c` (which takes the values given on the horizontal axis)

.. figure:: /_static/lecture_specific/mccall_model_with_separation/mccall_resw_c.png

As expected, higher unemployment compensation causes the worker to hold out for higher wages.

In effect, the cost of continuing job search is reduced.



The Reservation Wage and Discounting
------------------------------------

Next, let's investigate how :math:`\bar w` varies with the discount factor.

The next figure plots the reservation wage associated with different values of
:math:`\beta`

.. figure:: /_static/lecture_specific/mccall_model_with_separation/mccall_resw_beta.png

Again, the results are intuitive: More patient workers will hold out for higher wages.



The Reservation Wage and Job Destruction
----------------------------------------

Finally, let's look at how :math:`\bar w` varies with the job separation rate :math:`\alpha`.

Higher :math:`\alpha` translates to a greater chance that a worker will face termination in each period once employed.


.. figure:: /_static/lecture_specific/mccall_model_with_separation/mccall_resw_alpha.png

Once more, the results are in line with our intuition.

If the separation rate is high, then the benefit of holding out for a higher wage falls.

Hence the reservation wage is lower.



Exercises
=========



Exercise 1
----------

Reproduce all the reservation wage figures shown above.

Exercise 2
----------

Plot the reservation wage against the job offer rate :math:`\gamma`.

Use



.. code-block:: python3

    grid_size = 25
    γ_vals = np.linspace(0.05, 0.95, grid_size)



Interpret your results.



Solutions
=========

Exercise 1
----------

Using the ``compute_reservation_wage`` function mentioned earlier in the lecture,
we can create an array for reservation wages for different values of :math:`c`,
:math:`\beta` and :math:`\alpha` and plot the results like so

.. code-block:: python3

    grid_size = 25
    c_vals = np.linspace(2, 12, grid_size)  # Unemployment compensation
    w_bar_vals = np.empty_like(c_vals)

    mcm = McCallModel()

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, c in enumerate(c_vals):
        mcm.c = c
        w_bar = compute_reservation_wage(mcm)
        w_bar_vals[i] = w_bar

    ax.set(xlabel='unemployment compensation',
           ylabel='reservation wage')
    ax.plot(c_vals, w_bar_vals, label=r'$\bar w$ as a function of $c$')
    ax.grid()

    plt.show()

Exercise 2
----------

Similar to above, we can plot :math:`\bar w` against :math:`\gamma` as follows

.. code-block:: python3

    grid_size = 25
    γ_vals = np.linspace(0.05, 0.95, grid_size)
    w_bar_vals = np.empty_like(γ_vals)

    mcm = McCallModel()

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, γ in enumerate(γ_vals):
        mcm.γ = γ
        w_bar = compute_reservation_wage(mcm)
        w_bar_vals[i] = w_bar

    ax.plot(γ_vals, w_bar_vals, label=r'$\bar w$ as a function of $\gamma$')
    ax.set(xlabel='job offer rate', ylabel='reservation wage')
    ax.grid()

    plt.show()

As expected, the reservation wage increases in :math:`\gamma`.

This is because higher :math:`\gamma` translates to a more favorable job
search environment.

Hence workers are less willing to accept lower offers.
