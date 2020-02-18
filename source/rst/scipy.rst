.. _sp:

.. include:: /_static/includes/header.raw

**************
:index:`SciPy`
**************

.. index::
    single: Python; SciPy

.. contents:: :depth: 2


Overview
========

`SciPy <http://www.scipy.org>`_ builds on top of NumPy to provide common tools for scientific programming such as


* `linear algebra <http://docs.scipy.org/doc/scipy/reference/linalg.html>`_
* `numerical integration <http://docs.scipy.org/doc/scipy/reference/integrate.html>`_
* `interpolation <http://docs.scipy.org/doc/scipy/reference/interpolate.html>`_
* `optimization <http://docs.scipy.org/doc/scipy/reference/optimize.html>`_
* `distributions and random number generation <http://docs.scipy.org/doc/scipy/reference/stats.html>`_
* `signal processing <http://docs.scipy.org/doc/scipy/reference/signal.html>`_
* etc., etc


Like NumPy, SciPy is stable, mature and widely used.

Many SciPy routines are thin wrappers around industry-standard Fortran libraries such as `LAPACK <https://en.wikipedia.org/wiki/LAPACK>`_, `BLAS <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_, etc.

It's not really necessary to "learn" SciPy as a whole.

A more common approach is to get some idea of what's in the library and then look up `documentation <http://docs.scipy.org/doc/scipy/reference/index.html>`_ as required.

In this lecture, we aim only to highlight some useful parts of the package.


:index:`SciPy` versus :index:`NumPy`
====================================

SciPy is a package that contains various tools that are built on top of NumPy, using its array data type and related functionality.

In fact, when we import SciPy we also get NumPy, as can be seen from this excerpt the SciPy initialization file:

.. code-block:: python3

    # Import numpy symbols to scipy namespace
    from numpy import *
    from numpy.random import rand, randn
    from numpy.fft import fft, ifft
    from numpy.lib.scimath import *

However, it's more common and better practice to use NumPy functionality explicitly

.. code-block:: python3

    import numpy as np

    a = np.identity(3)

What is useful in SciPy is the functionality in its sub-packages

* ``scipy.optimize``, ``scipy.integrate``, ``scipy.stats``, etc.

Let's explore some of the major sub-packages.

Statistics
==========

.. index::
    single: SciPy; Statistics

The ``scipy.stats`` subpackage supplies

* numerous random variable objects (densities, cumulative distributions, random sampling, etc.)
* some estimation procedures
* some statistical tests

Random Variables and Distributions
----------------------------------

Recall that ``numpy.random`` provides functions for generating random variables

.. code-block:: python3

    np.random.beta(5, 5, size=3)

This generates a draw from the distribution with the density function below when ``a, b = 5, 5``

.. math::
    :label: betadist2

    f(x; a, b) = \frac{x^{(a - 1)} (1 - x)^{(b - 1)}}
        {\int_0^1 u^{(a - 1)} (1 - u)^{(b - 1)} du}
        \qquad (0 \leq x \leq 1)


Sometimes we need access to the density itself, or the cdf, the quantiles, etc.

For this, we can use ``scipy.stats``, which provides all of this functionality as well as random number generation in a single consistent interface.

Here's an example of usage

.. code-block:: ipython

    from scipy.stats import beta
    import matplotlib.pyplot as plt
    %matplotlib inline

    q = beta(5, 5)      # Beta(a, b), with a = b = 5
    obs = q.rvs(2000)   # 2000 observations
    grid = np.linspace(0.01, 0.99, 100)

    fig, ax = plt.subplots()
    ax.hist(obs, bins=40, density=True)
    ax.plot(grid, q.pdf(grid), 'k-', linewidth=2)
    plt.show()


The object ``q`` that represents the distribution has additional useful methods, including

.. code-block:: python3

    q.cdf(0.4)      # Cumulative distribution function

.. code-block:: python3

    q.ppf(0.8)      # Quantile (inverse cdf) function

.. code-block:: python3

    q.mean()

The general syntax for creating these objects that represent distributions (of type ``rv_frozen``) is

    ``name = scipy.stats.distribution_name(shape_parameters, loc=c, scale=d)``

Here ``distribution_name`` is one of the distribution names in `scipy.stats <http://docs.scipy.org/doc/scipy/reference/stats.html>`_.

The ``loc`` and ``scale`` parameters transform the original random variable
:math:`X` into :math:`Y = c + d X`.



Alternative Syntax
------------------

There is an alternative way of calling the methods described above.

For example, the code that generates the figure above can be replaced by

.. code-block:: python3

    obs = beta.rvs(5, 5, size=2000)
    grid = np.linspace(0.01, 0.99, 100)

    fig, ax = plt.subplots()
    ax.hist(obs, bins=40, density=True)
    ax.plot(grid, beta.pdf(grid, 5, 5), 'k-', linewidth=2)
    plt.show()



Other Goodies in scipy.stats
----------------------------

There are a variety of statistical functions in ``scipy.stats``.

For example, ``scipy.stats.linregress`` implements simple linear regression

.. code-block:: python3

    from scipy.stats import linregress

    x = np.random.randn(200)
    y = 2 * x + 0.1 * np.random.randn(200)
    gradient, intercept, r_value, p_value, std_err = linregress(x, y)
    gradient, intercept


To see the full list, consult the `documentation <https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-functions-scipy-stats>`__.



Roots and Fixed Points
======================

A **root** or **zero** of a real function :math:`f` on :math:`[a,b]` is an :math:`x \in [a, b]` such that :math:`f(x)=0`.

For example, if we plot the function

.. math::
    :label: root_f

    f(x) = \sin(4 (x - 1/4)) + x + x^{20} - 1


with :math:`x \in [0,1]` we get

.. code-block:: python3

  f = lambda x: np.sin(4 * (x - 1/4)) + x + x**20 - 1
  x = np.linspace(0, 1, 100)

  fig, ax = plt.subplots()
  ax.plot(x, f(x))
  ax.axhline(ls='--', c='k', label='$f(x)$')
  ax.set_xlabel('$x$', fontsize=12)
  ax.set_ylabel('$f(x)$', fontsize=12)
  ax.legend(fontsize=12)
  plt.show()

The unique root is approximately 0.408.

Let's consider some numerical techniques for finding roots.

:index:`Bisection`
------------------

.. index::
    single: SciPy; Bisection

One of the most common algorithms for numerical root-finding is *bisection*.

To understand the idea, recall the well-known game where

* Player A thinks of a secret number between 1 and 100

* Player B asks if it's less than 50

    * If yes, B asks if it's less than 25

    * If no, B asks if it's less than 75

And so on.

This is bisection.

Here's a simplistic implementation of the algorithm in Python.

It works for all sufficiently well behaved increasing continuous functions with :math:`f(a) < 0 < f(b)`

.. _bisect_func:

.. code-block:: python3

    def bisect(f, a, b, tol=10e-5):
        """
        Implements the bisection root finding algorithm, assuming that f is a
        real-valued function on [a, b] satisfying f(a) < 0 < f(b).
        """
        lower, upper = a, b

        while upper - lower > tol:
            middle = 0.5 * (upper + lower)
            if f(middle) > 0:   # root is between lower and middle 
                lower, upper = lower, middle
            else:               # root is between middle and upper 
                lower, upper = middle, upper

        return 0.5 * (upper + lower)


Let's test it using the function :math:`f` defined in :eq:`root_f`

.. code-block:: python3

    bisect(f, 0, 1)

Not surprisingly, SciPy provides its own bisection function. 

Let's test it using the same function :math:`f` defined in :eq:`root_f`

.. code-block:: python3

    from scipy.optimize import bisect

    bisect(f, 0, 1)



The :index:`Newton-Raphson Method`
----------------------------------

.. index::
    single: SciPy; Newton-Raphson Method

Another very common root-finding algorithm is the `Newton-Raphson method <https://en.wikipedia.org/wiki/Newton%27s_method>`_.

In SciPy this algorithm is implemented by ``scipy.optimize.newton``.

Unlike bisection, the Newton-Raphson method uses local slope information in an attempt to increase the speed of convergence.

Let's investigate this using the same function :math:`f` defined above.

With a suitable initial condition for the search we get convergence:

.. code-block:: python3

    from scipy.optimize import newton

    newton(f, 0.2)   # Start the search at initial condition x = 0.2

But other initial conditions lead to failure of convergence:

.. code-block:: python3

    newton(f, 0.7)   # Start the search at x = 0.7 instead



Hybrid Methods
--------------

A general principle of numerical methods is as follows:

* If you have specific knowledge about a given problem, you might be able to exploit it to generate efficiency.

* If not, then the choice of algorithm involves a trade-off between speed and robustness.

In practice, most default algorithms for root-finding, optimization and fixed points use *hybrid* methods.

These methods typically combine a fast method with a robust method in the following manner:

#. Attempt to use a fast method
#. Check diagnostics
#. If diagnostics are bad, then switch to a more robust algorithm

In ``scipy.optimize``, the function ``brentq`` is such a hybrid method and a good default

.. code-block:: python3

    from scipy.optimize import brentq

    brentq(f, 0, 1)

Here the correct solution is found and the speed is better than bisection:

.. code-block:: ipython

    %timeit brentq(f, 0, 1)

.. code-block:: ipython

    %timeit bisect(f, 0, 1)


Multivariate Root-Finding
-------------------------

.. index::
    single: SciPy; Multivariate Root-Finding

Use ``scipy.optimize.fsolve``, a wrapper for a hybrid method in MINPACK.

See the `documentation <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html>`__ for details.


Fixed Points
------------

A **fixed point** of a real function :math:`f` on :math:`[a,b]` is an :math:`x \in [a, b]` such that :math:`f(x)=x`.



.. index::
    single: SciPy; Fixed Points

SciPy has a function for finding (scalar) fixed points too

.. code-block:: python3

    from scipy.optimize import fixed_point

    fixed_point(lambda x: x**2, 10.0)  # 10.0 is an initial guess


If you don't get good results, you can always switch back to the ``brentq`` root finder, since
the fixed point of a function :math:`f` is the root of :math:`g(x) := x - f(x)`.




:index:`Optimization`
=====================

.. index::
    single: SciPy; Optimization

Most numerical packages provide only functions for *minimization*.

Maximization can be performed by recalling that the maximizer of a function :math:`f` on domain :math:`D` is
the minimizer of :math:`-f` on :math:`D`.

Minimization is closely related to root-finding: For smooth functions, interior optima correspond to roots of the first derivative.

The speed/robustness trade-off described above is present with numerical optimization too.

Unless you have some prior information you can exploit, it's usually best to use hybrid methods.

For constrained, univariate (i.e., scalar) minimization, a good hybrid option is ``fminbound``

.. code-block:: python3

    from scipy.optimize import fminbound

    fminbound(lambda x: x**2, -1, 2)  # Search in [-1, 2]


Multivariate Optimization
-------------------------

.. index::
    single: Optimization; Multivariate

Multivariate local optimizers include ``minimize``, ``fmin``, ``fmin_powell``, ``fmin_cg``, ``fmin_bfgs``, and ``fmin_ncg``.

Constrained multivariate local optimizers include ``fmin_l_bfgs_b``, ``fmin_tnc``, ``fmin_cobyla``.

See the `documentation <http://docs.scipy.org/doc/scipy/reference/optimize.html>`__ for details.



:index:`Integration`
====================

.. index::
    single: SciPy; Integration

Most numerical integration methods work by computing the integral of an approximating polynomial.

The resulting error depends on how well the polynomial fits the integrand, which in turn depends on how "regular" the integrand is.

In SciPy, the relevant module for numerical integration is ``scipy.integrate``.

A good default for univariate integration is ``quad``

.. code-block:: python3

    from scipy.integrate import quad

    integral, error = quad(lambda x: x**2, 0, 1)
    integral


In fact, ``quad`` is an interface to a very standard numerical integration routine in the Fortran library QUADPACK.

It uses `Clenshaw-Curtis quadrature <https://en.wikipedia.org/wiki/Clenshaw-Curtis_quadrature>`_,  based on expansion in terms of Chebychev polynomials.

There are other options for univariate integration---a useful one is ``fixed_quad``, which is fast and hence works well inside ``for`` loops.

There are also functions for multivariate integration.

See the `documentation <http://docs.scipy.org/doc/scipy/reference/integrate.html>`__ for more details.



:index:`Linear Algebra`
=======================

.. index::
    single: SciPy; Linear Algebra

We saw that NumPy provides a module for linear algebra called ``linalg``.

SciPy also provides a module for linear algebra with the same name.

The latter is not an exact superset of the former, but overall it has more functionality.

We leave you to investigate the `set of available routines <http://docs.scipy.org/doc/scipy/reference/linalg.html>`_.


Exercises
=========

.. _sp_ex1:


Exercise 1
----------

Previously we discussed the concept of :ref:`recursive function calls <recursive_functions>`.

Try to write a recursive implementation of homemade bisection function :ref:`described above <bisect_func>`.

Test it on the function :eq:`root_f`.


Solutions
=========


Exercise 1
----------

Here's a reasonable solution:

.. code-block:: python3

    def bisect(f, a, b, tol=10e-5):
        """
        Implements the bisection root-finding algorithm, assuming that f is a
        real-valued function on [a, b] satisfying f(a) < 0 < f(b).
        """
        lower, upper = a, b
        if upper - lower < tol:
            return 0.5 * (upper + lower)
        else:
            middle = 0.5 * (upper + lower)
            print(f'Current mid point = {middle}')
            if f(middle) > 0:   # Implies root is between lower and middle
                return bisect(f, lower, middle)
            else:               # Implies root is between middle and upper
                return bisect(f, middle, upper)


We can test it as follows

.. code-block:: python3

    f = lambda x: np.sin(4 * (x - 0.25)) + x + x**20 - 1
    bisect(f, 0, 1)
