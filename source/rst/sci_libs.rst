.. _speed:

.. include:: /_static/includes/header.raw

***************************
Other Scientific Libraries
***************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
============



In this lecture, we review some other scientific libraries that are useful for
economic research and analysis.

We have, however, already picked most of the low hanging fruit in terms of
economic research.

Hence you should feel free to skip this lecture on first pass.


:index:`Cython`
===============

.. index::
    single: Python; Cython


Like :doc:`Numba <numba>`,  `Cython <http://cython.org/>`__ provides an approach to generating fast compiled code that can be used from Python.

As was the case with Numba, a key problem is the fact that Python is dynamically typed.

As you'll recall, Numba solves this problem (where possible) by inferring type.

Cython's approach is different --- programmers add type definitions directly to their "Python" code.

As such, the Cython language can be thought of as Python with type definitions.

In addition to a language specification, Cython is also a language translator, transforming Cython code into optimized C and C++ code.

Cython also takes care of building language extensions --- the wrapper code that interfaces between the resulting compiled code and Python.



**Important Note:**.

In what follows code is executed in a Jupyter notebook.

This is to take advantage of a Cython `cell magic <http://ipython.readthedocs.org/en/stable/interactive/magics.html#cell-magics>`_ that makes Cython particularly easy to use.

Some modifications are required to run the code outside a notebook

* See the book `Cython <http://shop.oreilly.com/product/0636920033431.do>`__ by Kurt Smith or `the online documentation <http://cython.org/>`_





A First Example
------------------

Let's start with a rather artificial example.

Suppose that we want to compute the sum :math:`\sum_{i=0}^n \alpha^i` for given :math:`\alpha, n`.

Suppose further that we've forgotten the basic formula

.. math::

    \sum_{i=0}^n \alpha^i = \frac{1 - \alpha^{n+1}}{1 - \alpha}


for a geometric progression and hence have resolved to rely on a loop.

Python vs C
^^^^^^^^^^^^^^^^^^

Here's a pure Python function that does the job

.. code-block:: python3

    def geo_prog(alpha, n):
        current = 1.0
        sum = current
        for i in range(n):
            current = current * alpha
            sum = sum + current
        return sum

This works fine but for large :math:`n` it is slow.

Here's a C function that will do the same thing

.. code-block:: c
    :class: no-execute

    double geo_prog(double alpha, int n) {
        double current = 1.0;
        double sum = current;
        int i;
        for (i = 1; i <= n; i++) {
            current = current * alpha;
            sum = sum + current;
        }
        return sum;
    }


If you're not familiar with C, the main thing you should take notice of is the
type definitions

* ``int`` means integer

* ``double`` means double precision floating-point number

* the ``double`` in ``double geo_prog(...`` indicates that the function will
  return a double

Not surprisingly, the C code is faster than the Python code.

A Cython Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cython implementations look like a convex combination of Python and C.

We're going to run our Cython code in the Jupyter notebook, so we'll start by
loading the Cython extension in a notebook cell

.. code-block:: ipython

    %load_ext Cython

In the next cell, we execute the following

.. code-block:: ipython

    %%cython
    def geo_prog_cython(double alpha, int n):
        cdef double current = 1.0
        cdef double sum = current
        cdef int i
        for i in range(n):
            current = current * alpha
            sum = sum + current
        return sum

Here ``cdef`` is a Cython keyword indicating a variable declaration and is followed by a type.

The ``%%cython`` line at the top is not actually Cython code --- it's a Jupyter cell magic indicating the start of Cython code.

After executing the cell, you can now call the function ``geo_prog_cython`` from within Python.

What you are in fact calling is compiled C code with a Python call interface

.. code-block:: python3

    import quantecon as qe
    qe.util.tic()
    geo_prog(0.99, int(10**6))
    qe.util.toc()

.. code-block:: python3

    qe.util.tic()
    geo_prog_cython(0.99, int(10**6))
    qe.util.toc()





Example 2: Cython with NumPy Arrays
------------------------------------------

Let's go back to the first problem that we worked with: generating the iterates of the quadratic map

.. math::

    x_{t+1} = 4 x_t (1 - x_t)


The problem of computing iterates and returning a time series requires us to work with arrays.

The natural array type to work with is NumPy arrays.

Here's a Cython implementation that initializes, populates and returns a NumPy
array

.. code-block:: ipython

    %%cython
    import numpy as np

    def qm_cython_first_pass(double x0, int n):
        cdef int t
        x = np.zeros(n+1, float)
        x[0] = x0
        for t in range(n):
            x[t+1] = 4.0 * x[t] * (1 - x[t])
        return np.asarray(x)


If you run this code and time it, you will see that its performance is disappointing --- nothing like the speed gain we got from Numba

.. code-block:: python3

    qe.util.tic()
    qm_cython_first_pass(0.1, int(10**5))
    qe.util.toc()

This example was also computed in the :ref:`Numba lecture <qm_numba_result>`, and you can see Numba is around 90 times faster.

The reason is that working with NumPy arrays incurs substantial Python overheads.

We can do better by using Cython's `typed memoryviews <http://docs.cython.org/src/userguide/memoryviews.html>`_, which provide more direct access to arrays in memory.

When using them, the first step is to create a NumPy array.

Next, we declare a memoryview and bind it to the NumPy array.

Here's an example:

.. code-block:: ipython

    %%cython
    import numpy as np
    from numpy cimport float_t

    def qm_cython(double x0, int n):
        cdef int t
        x_np_array = np.zeros(n+1, dtype=float)
        cdef float_t [:] x = x_np_array
        x[0] = x0
        for t in range(n):
            x[t+1] = 4.0 * x[t] * (1 - x[t])
        return np.asarray(x)

Here

* ``cimport`` pulls in some compile-time information from NumPy

* ``cdef float_t [:] x = x_np_array`` creates a memoryview on the NumPy array ``x_np_array``

* the return statement uses ``np.asarray(x)`` to convert the memoryview back to a NumPy array

Let's time it:

.. code-block:: python3

    qe.util.tic()
    qm_cython(0.1, int(10**5))
    qe.util.toc()

This is fast, although still slightly slower than ``qm_numba``.



Summary
------------

Cython requires more expertise than Numba, and is a little more fiddly in terms of getting good performance.

In fact, it's surprising how difficult it is to beat the speed improvements provided by Numba.

Nonetheless,

* Cython is a very mature, stable and widely used tool

* Cython can be more useful than Numba when working with larger, more sophisticated applications





Joblib
==========


`Joblib <https://joblib.readthedocs.io/en/latest/>`__ is a popular Python library for
caching and parallelization.

To install it, start Jupyter and type

.. code-block:: ipython

    !pip install joblib


from within a notebook.

Here we review just the basics.




Caching
--------


Perhaps, like us, you sometimes run a long computation that simulates a model
at a given set of parameters --- to generate a figure, say, or a table.

20 minutes later you realize that you want to tweak the figure and now you have to
do it all again.

What caching will do is automatically store results at each parameterization.

With Joblib, results are compressed and stored on file, and automatically served
back up to you when you repeat the calculation.



An Example
-----------

Let's look at a toy example, related to the quadratic map model discussed :ref:`above <quad_map_eg>`.

Let's say we want to generate a long trajectory from a certain initial
condition :math:`x_0` and see what fraction of the sample is below 0.1.

(We'll omit JIT compilation or other speedups for simplicity).

Here's our code

.. code-block:: python3

    from joblib import Memory
    location = './cachedir'
    memory = Memory(location='./joblib_cache')

    @memory.cache
    def qm(x0, n):
        x = np.empty(n+1)
        x[0] = x0
        for t in range(n):
            x[t+1] = 4 * x[t] * (1 - x[t])
        return np.mean(x < 0.1)


We are using `joblib <https://joblib.readthedocs.io/en/latest/>`_ to cache the result of calling `qm` at a given set of parameters.

With the argument `location='./joblib_cache'`, any call to this function results in both the input values and output values being stored a subdirectory `joblib_cache` of the present working directory.

(In UNIX shells, `.` refers to the present working directory).

The first time we call the function with a given set of parameters we see some
extra output that notes information being cached

.. code-block:: python3

    qe.util.tic()
    n = int(1e7)
    qm(0.2, n)
    qe.util.toc()

The next time we call the function with the same set of parameters, the result is returned almost instantaneously

.. code-block:: python3

    qe.util.tic()
    n = int(1e7)
    qm(0.2, n)
    qe.util.toc()







Other Options
=================

There are in fact many other approaches to speeding up your Python code.

One is interfacing with Fortran.

.. index::
    single: Python; Interfacing with Fortran

If you are comfortable writing Fortran you will find it very easy to create
extension modules from Fortran code using `F2Py <https://docs.scipy.org/doc/numpy/f2py/>`_.

F2Py is a Fortran-to-Python interface generator that is particularly simple to
use.

Robert Johansson provides a `very nice introduction <http://nbviewer.jupyter.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-6A-Fortran-and-C.ipynb>`_ to F2Py, among other things.

Recently, `a Jupyter cell magic for Fortran
<http://nbviewer.jupyter.org/github/mgaitan/fortran_magic/blob/master/documentation.ipynb>`_ has been developed --- you might want to give it a try.






Exercises
=============

.. _speed_ex1:

Exercise 1
-------------

Later we'll learn all about :doc:`finite-state Markov chains <finite_markov>`.

For now, let's just concentrate on simulating a very simple example of such a chain.

Suppose that the volatility of returns on an asset can be in one of two regimes --- high or low.

The transition probabilities across states are as follows

.. figure:: /_static/lecture_specific/sci_libs/nfs_ex1.png


For example, let the period length be one month, and suppose the current state is high.

We see from the graph that the state next month will be

* high with probability 0.8

* low with probability 0.2

Your task is to simulate a sequence of monthly volatility states according to this rule.

Set the length of the sequence to ``n = 100000`` and start in the high state.

Implement a pure Python version, a Numba version and a Cython version, and compare speeds.

To test your code, evaluate the fraction of time that the chain spends in the low state.

If your code is correct, it should be about 2/3.



Solutions
==========



Exercise 1
----------

We let

-  0 represent "low"
-  1 represent "high"

.. code-block:: python3

    p, q = 0.1, 0.2  # Prob of leaving low and high state respectively

Here's a pure Python version of the function

.. code-block:: python3

    def compute_series(n):
        x = np.empty(n, dtype=int)
        x[0] = 1  # Start in state 1
        U = np.random.uniform(0, 1, size=n)
        for t in range(1, n):
            current_x = x[t-1]
            if current_x == 0:
                x[t] = U[t] < p
            else:
                x[t] = U[t] > q
        return x

Let's run this code and check that the fraction of time spent in the low
state is about 0.666

.. code-block:: python3

    n = 100000
    x = compute_series(n)
    print(np.mean(x == 0))  # Fraction of time x is in state 0


Now let's time it

.. code-block:: python3

    qe.util.tic()
    compute_series(n)
    qe.util.toc()


Next let's implement a Numba version, which is easy

.. code-block:: python3

    from numba import jit

    compute_series_numba = jit(compute_series)

Let's check we still get the right numbers

.. code-block:: python3

    x = compute_series_numba(n)
    print(np.mean(x == 0))


Let's see the time

.. code-block:: python3

    qe.util.tic()
    compute_series_numba(n)
    qe.util.toc()


This is a nice speed improvement for one line of code.

Now let's implement a Cython version

.. code-block:: ipython

    %load_ext Cython

.. code-block:: ipython

    %%cython
    import numpy as np
    from numpy cimport int_t, float_t

    def compute_series_cy(int n):
        # == Create NumPy arrays first == #
        x_np = np.empty(n, dtype=int)
        U_np = np.random.uniform(0, 1, size=n)
        # == Now create memoryviews of the arrays == #
        cdef int_t [:] x = x_np
        cdef float_t [:] U = U_np
        # == Other variable declarations == #
        cdef float p = 0.1
        cdef float q = 0.2
        cdef int t
        # == Main loop == #
        x[0] = 1
        for t in range(1, n):
            current_x = x[t-1]
            if current_x == 0:
                x[t] = U[t] < p
            else:
                x[t] = U[t] > q
        return np.asarray(x)

.. code-block:: python3

    compute_series_cy(10)

.. code-block:: python3

    x = compute_series_cy(n)
    print(np.mean(x == 0))


.. code-block:: python3

    qe.util.tic()
    compute_series_cy(n)
    qe.util.toc()


The Cython implementation is fast but not as fast as Numba.
