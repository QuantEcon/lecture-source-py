
.. _optgrowth:

.. include:: /_static/includes/header.raw

.. highlight:: python3

**************************************************************
:index:`Optimal Growth II: Accelerating the Code with Numba`
**************************************************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon
  !pip install interpolation

Overview
========

In a :doc:`previous lecture <optgrowth>`, we studied a stochastic optimal
growth model with one representative agent.

We solved the model using dynamic programming.

In writing our code, we focused on clarity and flexibility.

These are good things but there's often a trade-off between flexibility and
speed.

The reason is that, when code is less flexible, we can exploit structure more
easily.

So, in this lecture, we are going to accept less flexibility while gaining
speed, using just-in-time compilation to 
accelerate our code.

Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    from interpolation import interp
    from numba import jit, njit, jitclass, prange, float64, int32
    from quantecon.optimize.scalar_maximization import brent_max

    %matplotlib inline


We are using an interpolation function from
`interpolation.py <https://github.com/EconForge/interpolation.py>`__ because it
helps us JIT-compile our code.

The function `brent_max` is also designed for embedding in JIT-compiled code.

These are alternatives to similar functions in SciPy (which, unfortunately, are not JIT-aware).



The Model
=========

.. index::
    single: Optimal Growth; Model


The model is exactly the same as discussed in :doc:`this lecture <optgrowth>`.



Computation
===========

.. index::
    single: Dynamic Programming; Computation



In terms of primitives, we will assume for now that

* :math:`f(k) = k^{\alpha}`

* :math:`u(c) = \ln c`

* :math:`\phi` is the distribution of :math:`\exp(\mu + s \zeta)` when :math:`\zeta` is standard normal

We will store these primitives of the optimal growth model in a class.

In fact we are going to use :doc:`Numba's <numba>` `@jitclass` decorator to target our class for JIT compilation.

Because we are going to use Numba to compile our class, we need to specify the
types of the data:

.. code-block:: python3

   opt_growth_data = [
       ('α', float64),          # Production parameter
       ('β', float64),          # Discount factor
       ('μ', float64),          # Shock location parameter
       ('s', float64),          # Shock scale parameter
       ('grid', float64[:]),    # Grid (array)
       ('shocks', float64[:])   # Shock draws (array)
   ]

Note the convention for specifying the types of each argument.

Now we're ready to create our class, which will combine parameters and a
method that realizes the right hand side of the Bellman equation :eq:`fpb30`.

.. code-block:: python3

   @jitclass(opt_growth_data)
   class OptimalGrowthModel:

       def __init__(self,
                    α=0.4, 
                    β=0.96, 
                    μ=0,
                    s=0.1,
                    grid_max=4,
                    grid_size=200,
                    shock_size=250):

           self.α, self.β, self.μ, self.s = α, β, μ, s

           # Set up grid
           self.grid = np.linspace(1e-5, grid_max, grid_size)
           # Store shocks
           self.shocks = np.exp(μ + s * np.random.randn(shock_size))
           
       def f(self, k):
           return k**self.α
           
       def u(self, c):
           return np.log(c)

       def objective(self, c, y, v_array):
           """
           Right hand side of the Bellman equation.
           """

           u, f, β, shocks = self.u, self.f, self.β, self.shocks

           v = lambda x: interp(self.grid, v_array, x)

           return u(c) + β * np.mean(v(f(y - c) * shocks))



The Bellman Operator
--------------------

Here's a jitted function that implements the Bellman operator 

.. code-block:: python3

   @jit(nopython=True)
   def T(og, v):
       """
       The Bellman operator.

         * og is an instance of OptimalGrowthModel
         * v is an array representing a guess of the value function
       """
       v_new = np.empty_like(v)
       
       for i in range(len(grid)):
           y = grid[i]
           
           # Maximize RHS of Bellman equation at state y
           v_max = brent_max(og.objective, 1e-10, y, args=(y, v))[1]
           v_new[i] = v_max
           
       return v_new


Here's another function, very similar to the last, that computes a :math:`v`-greedy
policy:


.. code-block:: python3

   @jit(nopython=True)
   def get_greedy(og, v):
       """
       Compute a v-greedy policy.

         * og is an instance of OptimalGrowthModel
         * v is an array representing a guess of the value function
       """
       v_greedy = np.empty_like(v)
       
       for i in range(len(grid)):
           y = grid[i]
           
           # Find maximizer of RHS of Bellman equation at state y
           c_star = brent_max(og.objective, 1e-10, y, args=(y, v))[0]
           v_greedy[i] = c_star
           
       return v_greedy

The last two functions could be merged, as they were in our :doc:`previous implementation <optgrowth>`, but we resisted doing so to increase efficiency.


Here's a function that iterates until the difference is below a particular tolerance level.

.. code-block:: python3

   def solve_model(og,
                   tol=1e-4,
                   max_iter=1000,
                   verbose=True,
                   print_skip=25):

       # Set up loop
       v = np.log(og.grid)  # Initial condition
       i = 0
       error = tol + 1

       while i < max_iter and error > tol:
           v_new = T(og, v)
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



Let's compute the approximate solution at the default parameters.


First we create an instance:

.. code-block:: python3

    og = OptimalGrowthModel()

Now we call ``solve_model``, using the ``%%time`` magic to check how long it
takes.

.. code-block:: python3

    %%time
    v_solution = solve_model(og)

You will notice that this is *much* faster than our :doc:`original implementation <optgrowth>`.

Now let's plot our result.

.. code-block:: python3

    grid = og.grid
    fig, ax = plt.subplots()

    ax.plot(grid, v_solution, lw=2,
            alpha=0.6, label='Approximate value function')

    ax.legend(loc='lower right')
    plt.show()

Let's have a look at the policy too:

.. code-block:: python3

    v_greedy = get_greedy(og, v_solution)

    fig, ax = plt.subplots()

    ax.plot(grid, v_greedy, lw=2,
            alpha=0.6, label='Approximate value function')

    ax.legend(loc='lower right')
    plt.show()

Everything seems in order, so our code acceleration has been successful!

