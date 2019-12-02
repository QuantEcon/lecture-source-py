
.. include:: /_static/includes/header.raw

.. highlight:: python3

***********************************************
Job Search III: Fitted Value Function Iteration
***********************************************

.. contents:: :depth: 2


In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon
  !pip install --upgrade interpolation


Overview
========

Moreover, the topic is nontrivial because working with continuous distributions leads to an uncountably infinite state space.

Infinite state spaces produce their own challenges, particularly when we seek
to apply value function iteration (VFI).

This will lead us to modify VFI by adding an interpolation step.

The combination of VFI and this interpolation step is called **fitted value function iteration**.

We will use the following imports:

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

    import quantecon as qe
    from interpolation import interp
    from numpy.random import randn
    from numba import njit, jitclass, prange, float64




The Model
=========

The model is the same as :doc:`before <mccall_model>` except for the assumption on the state space.

The wage offer takes the form

.. math::

    w_t = w(s_t) \quad \text{where } \; s_t \in \mathbb{S}


where :math:`\{s_t\}` is IID and :math:`s_t` is observable at the start of :math:`t`.
  :math:`w_t = w(s_t)`,


We will assume that :math:`s_t` takes values in :math:`\mathbb R` according to
density :math:`q`.

The Bellman equation is now

.. math::
    :label: odu_pv

    v^*(s)
    = \max \left\{
            \frac{w(s)}{1 - \beta}, \, c + \beta 
            \int v^*(w(s')) q (s') ds'
        \right\}

The optimal policy is 

.. math::
    \sigma(s) := \mathbf{1}
        \left\{
            \frac{w(s)}{1 - \beta} \geq c + \beta \int
            v^*(w(s')) q (s') ds'
        \right\}

We can also write this as

.. math::
    \sigma(s) := \mathbf{1} \{ w(s) \geq \bar w \}

where

.. math::
    \bar w := (1 - \beta) \left\{ c + \beta \int v^*(w(s')) q (s') ds' \right\}


is the reservation wage.




Computing the Optimal Policy: Take 1
====================================

To put the above ideas into action, we need to compute the value function at
each possible state :math:`s \in \mathbb S`.

Let's suppose that :math:`\mathbb S = \{1, \ldots, n\}`.

The value function is then represented by the vector :math:`v^* =
(v^*(i))_{i=1}^n`.

In view of :eq:`odu_pv`, this vector satisfies the nonlinear system of equations


.. math::
    :label: odu_pv2

    v^*(i)
    = \max \left\{
            \frac{w(i)}{1 - \beta}, \, c + \beta \sum_{1 \leq j \leq n} 
                v^*(w(j)) q (j)
        \right\}
    \quad
    \text{for } i = 1, \ldots, n


The Algorithm
-------------

To compute this vector, we proceed as follows:

Step 1: pick an arbitrary initial guess :math:`v \in \mathbb R^n`.

Step 2: compute a new vector :math:`v' \in \mathbb R^n` via


.. math::
    :label: odu_pv2p

    v'(i)
    = \max \left\{
            \frac{w(i)}{1 - \beta}, \, c + \beta \sum_{1 \leq j \leq n} 
                v^*(w(j)) q (j)
        \right\}
    \quad
    \text{for } i = 1, \ldots, n

Step 3: calculate a measure of the deviation between :math:`v` and :math:`v'`, such as :math:`\max_i |v(i)- v(i')|`.

Step 4: if the deviation is larger than some fixed tolerance, set :math:`v = v'` and go to step 2, else continue.

Step 5: return :math:`v`.

This algorithm returns an arbitrarily good approximation to the true solution
to :eq:`odu_pv2`, which represents the value function.

(Arbitrarily good means here that the approximation converges to the true
solution as the tolerance goes to zero.)


The Fixed Point Theory
----------------------

What's the math behind these ideas?

First, one defines a mapping :math:`T` from :math:`\mathbb R^n` to
itself via

.. math::
    :label: odu_pv3

    (Tv)(i)
    = \max \left\{
            \frac{w(i)}{1 - \beta}, \, c + \beta \sum_{1 \leq j \leq n} 
                v^*(w(j)) q (j)
        \right\}
    \quad
    \text{for } i = 1, \ldots, n


(A new vector :math:`Tv` is obtained from given vector :math:`v` by evaluating
the r.h.s. at each :math:`i`)

One can show that the conditions of the Banach contraction mapping theorem are
satisfied by :math:`T` as a self-mapping on :math:`\mathbb R^n`.

One implication is that :math:`T` has a unique fixed point in :math:`\mathbb R^n`.

Moreover, it's immediate from the definition of :math:`T` that this fixed
point is precisely the value function.

The iterative algorithm presented above corresponds to iterating with
:math:`T` from some initial guess :math:`v`.

The Banach contraction mapping theorem tells us that this iterative process
generates a sequence that converges to the fixed point.



Implementation
--------------

Our default state process will be Beta-Binomial:


.. code-block:: python3

    n, a, b = 50, 200, 100
    dist = BetaBinomial(n, a, b)
    q_default = dist.pdf()

Our default set of values for wages will be

.. code-block:: python3

    w_min, w_max = 10, 60
    w_default = np.linspace(w_min, w_max, n+1)

Here's a plot of wages vs probabilities:

.. code-block:: python3

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(w_default, q_default, 'o-', label='$q(w(i))$')
    ax.set_xlabel('wages')
    ax.set_ylabel('probabilities')

    plt.show()

For the dynamic programming component, 
we are going to use Numba to accelerate our code (see, in particular, the
discussion of ``@jitclass`` in :doc:`our lecture on Numba <numba>`).

The following helps Numba by providing some type information about the data we need.

.. code-block:: python3

    mccall_data = [
        ('c', float64),      # unemployment compensation
        ('β', float64),      # discount factor
        ('w', float64[:]),   # w[i] = w(i) = wage at state i
        ('q', float64[:])    # q[i] = probability of state i
    ]

Here's a class that stores the data and the right hand side of the Bellman
equation.

Default parameter values are embedded in the class.


.. code-block:: python3

    @jitclass(mccall_data)
    class McCallModel:

        def __init__(self, c=25, β=0.99, w=w_default, q=q_default):

            self.c, self.β = c, β
            self.w, self.q = w_default, q_default

        def bellman(self, i, v):
            """
            The r.h.s. of the Bellman equation at state i.
            """
            stopping_value = self.w[i] / (1 - self.β)
            continuation_value = self.c + self.β * np.sum(v * self.q)
            max_value = max(stopping_value, continuation_value)
            return(max_value)



Based on these defaults, let's try plotting the first few approximate value functions
in the sequence :math:`\{ T^n v \}`.

We will start from guess :math:`v` given by :math:`v(i) = w(i) / (1 - β)`, which is the value of accepting at every given wage.

Here's a function to implement this:


.. code-block:: python3

    def plot_value_function_seq(mcm, ax, num_plots=6):
        """
        Plot a sequence of value functions.

            * mcm is an instance of McCallModel
            * ax is an axes object that implements a plot method.

        """

        n = len(mcm.w)
        v = mcm.w / (1 - mcm.β)
        v_next = np.empty_like(v)
        for i in range(num_plots):
            ax.plot(mcm.w, v, label=f"iterate {i}")
            # Update guess
            for i in range(n):
                v_next[i] = mcm.bellman(i, v)
            v[:] = v_next  # copy contents into v

        ax.legend(loc='lower right')

Now let's create an instance of ``McCallModel`` and call the function:

.. code-block:: python3

    mcm = McCallModel()

    fig, ax = plt.subplots(figsize=(9, 6.5))
    plot_value_function_seq(mcm, ax)
    plt.show()

Here's a more serious iteration effort, that continues until measured deviation
between successive iterates is below `tol`.

We'll be using JIT compilation via Numba to turbocharge our loops

.. code-block:: python3

    @jit(nopython=True)
    def compute_reservation_wage(mcm,
                                 max_iter=500,
                                 tol=1e-6):

        # == First compute the value function == #

        n = len(mcm.w)
        v = mcm.w / (1 - mcm.β)          # initial guess
        v_next = np.empty_like(v)    # storage
        i = 0
        error = tol + 1
        while i < max_iter and error > tol:

            for i in range(n):
                v_next[i] = mcm.bellman(i, v)

            error = np.max(np.abs(v_next - v))
            i += 1

            v[:] = v_next  # copy contents into v

        # == Now compute the reservation wage == #

        return (1 - mcm.β) * (mcm.c + mcm.β * np.sum(v * mcm.q))



Let's compute the reservation wage at the default parameters


.. code-block:: python3

    compute_reservation_wage(mcm)



Comparative Statics
-------------------


Now we know how to compute the reservation wage, let's see how it varies with
parameters.

In particular, let's look at what happens when we change :math:`\beta` and
:math:`c`.



.. code-block:: python3

    grid_size = 25
    R = np.empty((grid_size, grid_size))

    c_vals = np.linspace(10.0, 30.0, grid_size)
    β_vals = np.linspace(0.9, 0.99, grid_size)

    for i, c in enumerate(c_vals):
        for j, β in enumerate(β_vals):
            mcm = McCallModel(c=c, β=β)
            R[i, j] = compute_reservation_wage(mcm)



.. code-block:: python3

    fig, ax = plt.subplots(figsize=(10, 5.7))

    cs1 = ax.contourf(c_vals, β_vals, R.T, alpha=0.75)
    ctr1 = ax.contour(c_vals, β_vals, R.T)

    plt.clabel(ctr1, inline=1, fontsize=13)
    plt.colorbar(cs1, ax=ax)


    ax.set_title("reservation wage")
    ax.set_xlabel("$c$", fontsize=16)
    ax.set_ylabel("$β$", fontsize=16)

    ax.ticklabel_format(useOffset=False)

    plt.show()






As expected, the reservation wage increases both with patience and with
unemployment compensation.


Here's an implementation:



.. code-block:: python3

    @jit(nopython=True)
    def compute_reservation_wage_two(mcm,
                                     max_iter=500,
                                     tol=1e-5):

        # == First compute q == #

        h = np.sum(mcm.w * mcm.q) / (1 - mcm.β)
        i = 0
        error = tol + 1
        while i < max_iter and error > tol:

            s = np.maximum(mcm.w / (1 - mcm.β), h)
            h_next = mcm.c + mcm.β * np.sum(s * mcm.q)

            error = np.abs(h_next - h)
            i += 1

            h = h_next

        # == Now compute the reservation wage == #

        return (1 - mcm.β) * h


You can use this code to solve the exercise below.



Exercises
=========



Exercise 1
----------

Compute the average duration of unemployment when :math:`\beta=0.99` and
:math:`c` takes the following values

    ``c_vals = np.linspace(10, 40, 25)``

That is, start the agent off as unemployed, compute their reservation wage
given the parameters, and then simulate to see how long it takes to accept.

Repeat a large number of times and take the average.

Plot mean unemployment duration as a function of :math:`c` in ``c_vals``.


Solutions
=========


Exercise 1
----------

Here's one solution



.. code-block:: python3


    cdf = np.cumsum(q_default)

    @jit(nopython=True)
    def compute_stopping_time(w_bar, seed=1234):

        np.random.seed(seed)
        t = 1
        while True:
            # Generate a wage draw
            w = w_default[qe.random.draw(cdf)]
            if w >= w_bar:
                stopping_time = t
                break
            else:
                t += 1
        return stopping_time

    @jit(nopython=True)
    def compute_mean_stopping_time(w_bar, num_reps=100000):
        obs = np.empty(num_reps)
        for i in range(num_reps):
            obs[i] = compute_stopping_time(w_bar, seed=i)
        return obs.mean()

    c_vals = np.linspace(10, 40, 25)
    stop_times = np.empty_like(c_vals)
    for i, c in enumerate(c_vals):
        mcm = McCallModel(c=c)
        w_bar = compute_reservation_wage_two(mcm)
        stop_times[i] = compute_mean_stopping_time(w_bar)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.plot(c_vals, stop_times, label="mean unemployment duration")
    ax.set(xlabel="unemployment compensation", ylabel="months")
    ax.legend()

    plt.show()
