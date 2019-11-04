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

    qe.util.tic()
    qm(0.1, int(10**5))
    time1 = qe.util.toc()


.. code-block:: python3

    qe.util.tic()
    qm_numba(0.1, int(10**5))
    time2 = qe.util.toc()


The first execution is relatively slow because of JIT compilation (see below).

Next time and all subsequent times it runs much faster:

.. _qm_numba_result:

.. code-block:: python3

    qe.util.tic()
    qm_numba(0.1, int(10**5))
    time2 = qe.util.toc()

.. code-block:: python3

    time1 / time2  # Calculate speed gain

That's a speed increase of two orders of magnitude!

Your mileage will of course vary depending on hardware and so on.

Nonetheless, two orders of magnitude is huge relative to how simple and clear the implementation is.

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

