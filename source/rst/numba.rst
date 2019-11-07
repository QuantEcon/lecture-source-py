.. _speed:

.. include:: /_static/includes/header.raw

*****
Numba
*****

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Please also make sure that you have the latest version of Anaconda, since old
versions are a :doc:`common source of errors <troubleshooting>`.

Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt

    %matplotlib inline


Overview
========

In our lecture on :doc:`NumPy <numpy>`, we learned one method to improve speed
and efficiency in numerical work.

That method, called *vectorization*, involved sending array processing
operations in batch to efficient low-level code.

Unfortunately, as we :ref:`discussed previously <numba-p_c_vectorization>`, vectorization is limited and has several weaknesses.

One is that it is highly memory-intensive.

Another is that only some algorithms can be vectorized.

In the last few years, a new Python library called `Numba
<http://numba.pydata.org/>`__ has appeared that solves many of these problems.

It does so through something called **just in time (JIT) compilation**.

JIT compilation is effective in many numerical settings and can generate extremely fast, efficient code.

It can also do other tricks such as multithreading (a form of parallelization well suited to numerical work).

Numba will be a key part of our lectures --- especially those lectures
involving dynamic programming.

This lecture introduces the main ideas.


A Cautionary Note
------------------

Numba aims to automatically compile functions to native machine code instructions on the fly.

When it succeeds, the compiled code is extremely fast.

But the process isn't flawless, since Numba needs to infer type information on
all variables to generate pure machine instructions.

Inference isn't possible in every setting, so you will need to invest some time in learning how Numba works.

However, you will find that, for simple routines, Numba infers types very well.

Moreover, the "hot loops" that we actually need to speed up are often
relatively simple.

This explains why, despite its imperfections, we are huge fans of Numba at QuantEcon!


.. _numba_link:

:index:`Compiling Functions`
============================

.. index::
    single: Python; Numba

As stated above, Numba's primary use is compiling functions to fast native
machine code during runtime.


.. _quad_map_eg:

An Example
----------

Let's consider some problems that are difficult to vectorize.

One is generating the trajectory of a difference equation given an initial
condition.

Let's take the difference equation to be the quadratic map

.. math::

    x_{t+1} = 4 x_t (1 - x_t)


Here's the plot of a typical trajectory, starting from :math:`x_0 = 0.1`, with :math:`t` on the x-axis

.. code-block:: python3

  def qm(x0, n):
      x = np.empty(n+1)
      x[0] = x0
      for t in range(n):
          x[t+1] = 4 * x[t] * (1 - x[t])
      return x

  x = qm(0.1, 250)
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(x, 'b-', lw=2, alpha=0.8)
  ax.set_xlabel('time', fontsize=16)
  plt.show()

To speed this up using Numba is trivial using Numba's ``jit`` function

.. code-block:: python3

    from numba import jit

    qm_numba = jit(qm)  # qm_numba is now a 'compiled' version of qm

Let's time and compare identical function calls across these two versions:

.. code-block:: python3

    n = 1_000_000
    qe.util.tic()
    qm(0.1, int(n))
    time1 = qe.util.toc()


.. code-block:: python3

    qe.util.tic()
    qm_numba(0.1, int(n))
    time2 = qe.util.toc()


The first execution is relatively slow because of JIT compilation (see below).

Next time and all subsequent times it runs much faster:

.. _qm_numba_result:

.. code-block:: python3

    qe.util.tic()
    qm_numba(0.1, int(n))
    time2 = qe.util.toc()

.. code-block:: python3

    time1 / time2  # Calculate speed gain

On our machines the output here is typically a two orders of magnitude speed gain.

(Your mileage will vary depending on hardware and so on.)

Nonetheless, this kind of speed gain is huge relative to how simple and clear the implementation is.

Decorator Notation
^^^^^^^^^^^^^^^^^^

If you don't need a separate name for the "numbafied" version of ``qm``,
you can just put ``@jit`` before the function

.. code-block:: python3

    @jit
    def qm(x0, n):
        x = np.empty(n+1)
        x[0] = x0
        for t in range(n):
            x[t+1] = 4 * x[t] * (1 - x[t])
        return x


This is equivalent to ``qm = jit(qm)``.


How and When it Works
---------------------

Numba attempts to generate fast machine code using the infrastructure provided by the `LLVM Project <http://llvm.org/>`_.

It does this by inferring type information on the fly.

As you can imagine, this is easier for simple Python objects (simple scalar data types, such as floats, integers, etc.).

Numba also plays well with NumPy arrays, which it treats as typed memory regions.

In an ideal setting, Numba can infer all necessary type information.

This allows it to generate native machine code, without having to call the Python runtime environment.

In such a setting, Numba will be on par with machine code from low-level languages.

When Numba cannot infer all type information, some Python objects are given generic ``object`` status, and some code is generated using the Python runtime.

In this second setting, Numba typically provides only minor speed gains --- or none at all.

Hence, it's prudent when using Numba to focus on speeding up small, time-critical snippets of code.

This will give you much better performance than blanketing your Python programs with ``@jit`` statements.


A Gotcha: Global Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the following example

.. code-block:: python3

    a = 1

    @jit
    def add_x(x):
        return a + x

    print(add_x(10))

.. code-block:: python3

    a = 2

    print(add_x(10))


Notice that changing the global had no effect on the value returned by the
function.

When Numba compiles machine code for functions, it treats global variables as constants to ensure type stability.



Numba for Vectorization
-----------------------

Numba can also be used to create custom :ref:`ufuncs <ufuncs>` with the `@vectorize <http://numba.pydata.org/numba-doc/dev/user/vectorize.html>`__ decorator.

To illustrate the advantage of using Numba to vectorize a function, we
return to a maximization problem :ref:`discussed previously <ufuncs>`

.. code-block:: python3

    def f(x, y):
        return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

    grid = np.linspace(-3, 3, 1000)
    x, y = np.meshgrid(grid, grid)

    qe.tic()
    np.max(f(x, y))
    qe.toc()

This is plain vanilla vectorized code using NumPy.

Here's a Numba version that does the same task, using `@vectorize`

.. code-block:: python3

    from numba import vectorize

    @vectorize
    def f_vec(x, y):
        return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

    grid = np.linspace(-3, 3, 1000)
    x, y = np.meshgrid(grid, grid)

    np.max(f_vec(x, y))  # Run once to compile

    qe.tic()
    np.max(f_vec(x, y))
    qe.toc()

So far there's no real advantage to using the `@vectorize` version.

However, we can gain further speed improvements using Numba's automatic
parallelization feature by specifying ``target='parallel'``.

In this case, we need to specify the types of our inputs and outputs

.. code-block:: python3

    @vectorize('float64(float64, float64)', target='parallel')
    def f_vec(x, y):
        return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

    np.max(f_vec(x, y))  # Run once to compile

    qe.tic()
    np.max(f_vec(x, y))
    qe.toc()

Now our code runs significantly faster than the NumPy version.

We'll discuss parallelization again below.

Compiling Classes
==================

As mentioned above, at present Numba can only compile a subset of Python.

However, that subset is ever expanding.

Numba is now quite effective at compiling classes.

If a class is successfully compiled, then its methods acts as JIT-compiled
functions.

To give one example, let's consider the class for analyzing the Solow growth model we
created in :doc:`this lecture <python_oop>`.

To compile this class we use the `@jitclass` decorator:

.. code-block:: python3

    from numba import jitclass, float64

Notice that we also imported something called `float64`.

This is a data type representing standard floating point numbers.

We are importing it here because Numba needs a bit of extra help with types when it trys to deal with classes.

Here's our code:

.. code-block:: python3

    solow_data = [
        ('n', float64),
        ('s', float64),
        ('δ', float64),
        ('α', float64),
        ('z', float64),
        ('k', float64)
    ]

    @jitclass(solow_data)
    class Solow:
        r"""
        Implements the Solow growth model with the update rule

            k_{t+1} = [(s z k^α_t) + (1 - δ)k_t] /(1 + n)

        """
        def __init__(self, n=0.05,  # population growth rate
                           s=0.25,  # savings rate
                           δ=0.1,   # depreciation rate
                           α=0.3,   # share of labor
                           z=2.0,   # productivity
                           k=1.0):  # current capital stock

            self.n, self.s, self.δ, self.α, self.z = n, s, δ, α, z
            self.k = k

        def h(self):
            "Evaluate the h function"
            # Unpack parameters (get rid of self to simplify notation)
            n, s, δ, α, z = self.n, self.s, self.δ, self.α, self.z
            # Apply the update rule
            return (s * z * self.k**α + (1 - δ) * self.k) / (1 + n)

        def update(self):
            "Update the current state (i.e., the capital stock)."
            self.k =  self.h()

        def steady_state(self):
            "Compute the steady state value of capital."
            # Unpack parameters (get rid of self to simplify notation)
            n, s, δ, α, z = self.n, self.s, self.δ, self.α, self.z
            # Compute and return steady state
            return ((s * z) / (n + δ))**(1 / (1 - α))

        def generate_sequence(self, t):
            "Generate and return a time series of length t"
            path = []
            for i in range(t):
                path.append(self.k)
                self.update()
            return path

First we specified the types of the instance data for the class in
`solow_data`.

After that, targeting the class for JIT compilation only requires adding
`@jitclass(solow_data)` before the class definition.

When we call the methods in the class, the methods are compiled just like functions.


.. code-block:: python3

    s1 = Solow()
    s2 = Solow(k=8.0)

    T = 60
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot the common steady state value of capital
    ax.plot([s1.steady_state()]*T, 'k-', label='steady state')

    # Plot time series for each economy
    for s in s1, s2:
        lb = f'capital series from initial state {s.k}'
        ax.plot(s.generate_sequence(T), 'o-', lw=2, alpha=0.6, label=lb)

    ax.legend()
    plt.show()


