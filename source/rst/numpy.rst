.. _np:

.. include:: /_static/includes/header.raw

**************
:index:`NumPy`
**************

.. index::
    single: Python; NumPy

.. contents:: :depth: 2

.. epigraph::

    "Let's be clear: the work of science has nothing whatever to do with consensus.  Consensus is the business of politics. Science, on the contrary, requires only one investigator who happens to be right, which means that he or she has results that are verifiable by reference to the real world. In science consensus is irrelevant. What is relevant is reproducible results." -- Michael Crichton



Overview
========

`NumPy <https://en.wikipedia.org/wiki/NumPy>`_ is a first-rate library for numerical programming

* Widely used in academia, finance and industry.

* Mature, fast, stable and under continuous development.


We have already seen some code involving NumPy in the preceding lectures.

In this lecture, we will start a more systematic discussion of both

* NumPy arrays and

* the fundamental array processing operations provided by NumPy.



References
----------

* `The official NumPy documentation <http://docs.scipy.org/doc/numpy/reference/>`_.


.. _numpy_array:

NumPy Arrays
============

.. index::
    single: NumPy; Arrays

The essential problem that NumPy solves is fast array processing.

The most important structure that NumPy defines is an array data type formally called a `numpy.ndarray <http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_.

NumPy arrays power a large proportion of the scientific Python ecosystem.

Let's first import the library.

.. code-block:: python3

    import numpy as np

To create a NumPy array containing only zeros we use  `np.zeros <http://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html#numpy.zeros>`_

.. code-block:: python3

    a = np.zeros(3)
    a

.. code-block:: python3

    type(a)



NumPy arrays are somewhat like native Python lists, except that

* Data *must be homogeneous* (all elements of the same type).
* These types must be one of the `data types <https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`_ (``dtypes``) provided by NumPy.

The most important of these dtypes are:

*    float64: 64 bit floating-point number
*    int64: 64 bit integer
*    bool:  8 bit True or False

There are also dtypes to represent complex numbers, unsigned integers, etc.

On modern machines, the default dtype for arrays is ``float64``

.. code-block:: python3

    a = np.zeros(3)
    type(a[0])


If we want to use integers we can specify as follows:

.. code-block:: python3

    a = np.zeros(3, dtype=int)
    type(a[0])


.. _numpy_shape_dim:

Shape and Dimension
-------------------

.. index::
    single: NumPy; Arrays (Shape and Dimension)

Consider the following assignment

.. code-block:: python3

    z = np.zeros(10)


Here ``z`` is a *flat* array with no dimension --- neither row nor column vector.

The dimension is recorded in the ``shape`` attribute, which is a tuple

.. code-block:: python3

    z.shape

Here the shape tuple has only one element, which is the length of the array (tuples with one element end with a comma).

To give it dimension, we can change the ``shape`` attribute

.. code-block:: python3

    z.shape = (10, 1)
    z

.. code-block:: python3

    z = np.zeros(4)
    z.shape = (2, 2)
    z


In the last case, to make the 2 by 2 array, we could also pass a tuple to the ``zeros()`` function, as
in ``z = np.zeros((2, 2))``.

.. _creating_arrays:

Creating Arrays
---------------

.. index::
    single: NumPy; Arrays (Creating)

As we've seen, the ``np.zeros`` function creates an array of zeros.

You can probably guess what ``np.ones`` creates.

Related is ``np.empty``, which creates arrays in memory that can later be populated with data

.. code-block:: python3

    z = np.empty(3)
    z

The numbers you see here are garbage values.

(Python allocates 3 contiguous 64 bit pieces of memory, and the existing contents of those memory slots are interpreted as ``float64`` values)

To set up a grid of evenly spaced numbers use ``np.linspace``

.. code-block:: python3

    z = np.linspace(2, 4, 5)  # From 2 to 4, with 5 elements

To create an identity matrix use either ``np.identity`` or ``np.eye``

.. code-block:: python3

    z = np.identity(2)
    z


In addition, NumPy arrays can be created from Python lists, tuples, etc. using ``np.array``

.. code-block:: python3

    z = np.array([10, 20])                 # ndarray from Python list
    z

.. code-block:: python3

    type(z)

.. code-block:: python3

    z = np.array((10, 20), dtype=float)    # Here 'float' is equivalent to 'np.float64'
    z

.. code-block:: python3

    z = np.array([[1, 2], [3, 4]])         # 2D array from a list of lists
    z

See also ``np.asarray``, which performs a similar function, but does not make
a distinct copy of data already in a NumPy array.

.. code-block:: python3

    na = np.linspace(10, 20, 2)
    na is np.asarray(na)   # Does not copy NumPy arrays

.. code-block:: python3

    na is np.array(na)     # Does make a new copy --- perhaps unnecessarily


To read in the array data from a text file containing numeric data use ``np.loadtxt``
or ``np.genfromtxt``---see `the documentation <http://docs.scipy.org/doc/numpy/reference/routines.io.html>`_ for details.


Array Indexing
--------------

.. index::
    single: NumPy; Arrays (Indexing)

For a flat array, indexing is the same as Python sequences:

.. code-block:: python3

    z = np.linspace(1, 2, 5)
    z

.. code-block:: python3

    z[0]

.. code-block:: python3

    z[0:2]  # Two elements, starting at element 0

.. code-block:: python3

    z[-1]


For 2D arrays the index syntax is as follows:

.. code-block:: python3

    z = np.array([[1, 2], [3, 4]])
    z

.. code-block:: python3

    z[0, 0]

.. code-block:: python3

    z[0, 1]


And so on.

Note that indices are still zero-based, to maintain compatibility with Python sequences.

Columns and rows can be extracted as follows

.. code-block:: python3

    z[0, :]

.. code-block:: python3

    z[:, 1]

NumPy arrays of integers can also be used to extract elements

.. code-block:: python3

    z = np.linspace(2, 4, 5)
    z

.. code-block:: python3

    indices = np.array((0, 2, 3))
    z[indices]

Finally, an array of ``dtype bool`` can be used to extract elements

.. code-block:: python3

    z

.. code-block:: python3

    d = np.array([0, 1, 1, 0, 0], dtype=bool)
    d

.. code-block:: python3

    z[d]

We'll see why this is useful below.

An aside: all elements of an array can be set equal to one number using slice notation

.. code-block:: python3

    z = np.empty(3)
    z

.. code-block:: python3

    z[:] = 42
    z


Array Methods
-------------

.. index::
    single: NumPy; Arrays (Methods)

Arrays have useful methods, all of which are carefully optimized

.. code-block:: python3

    a = np.array((4, 3, 2, 1))
    a

.. code-block:: python3

    a.sort()              # Sorts a in place
    a

.. code-block:: python3

    a.sum()               # Sum

.. code-block:: python3

    a.mean()              # Mean

.. code-block:: python3

    a.max()               # Max

.. code-block:: python3

    a.argmax()            # Returns the index of the maximal element

.. code-block:: python3

    a.cumsum()            # Cumulative sum of the elements of a

.. code-block:: python3

    a.cumprod()           # Cumulative product of the elements of a

.. code-block:: python3

    a.var()               # Variance

.. code-block:: python3

    a.std()               # Standard deviation

.. code-block:: python3

    a.shape = (2, 2)
    a.T                   # Equivalent to a.transpose()


Another method worth knowing is ``searchsorted()``.

If ``z`` is a nondecreasing array, then ``z.searchsorted(a)`` returns the index of the first element of ``z`` that is ``>= a``

.. code-block:: python3

    z = np.linspace(2, 4, 5)
    z

.. code-block:: python3

    z.searchsorted(2.2)

Many of the methods discussed above have equivalent functions in the NumPy namespace

.. code-block:: python3

    a = np.array((4, 3, 2, 1))

.. code-block:: python3

    np.sum(a)

.. code-block:: python3

    np.mean(a)




Operations on Arrays
====================

.. index::
    single: NumPy; Arrays (Operations)

Arithmetic Operations
---------------------

The operators ``+``, ``-``, ``*``, ``/`` and ``**`` all act *elementwise* on arrays

.. code-block:: python3

    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    a + b

.. code-block:: python3

    a * b


We can add a scalar to each element as follows

.. code-block:: python3

    a + 10

Scalar multiplication is similar

.. code-block:: python3

    a * 10

The two-dimensional arrays follow the same general rules

.. code-block:: python3

    A = np.ones((2, 2))
    B = np.ones((2, 2))
    A + B

.. code-block:: python3

    A + 10

.. code-block:: python3

    A * B


.. _numpy_matrix_multiplication:

In particular, ``A * B`` is *not* the matrix product, it is an element-wise product.

Matrix Multiplication
---------------------

.. index::
    single: NumPy; Matrix Multiplication

With Anaconda's scientific Python package based around Python 3.5 and above,
one can use the ``@`` symbol for matrix multiplication, as follows:

.. code-block:: python3

    A = np.ones((2, 2))
    B = np.ones((2, 2))
    A @ B

(For older versions of Python and NumPy you need to use the `np.dot <http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html>`_ function)


We can also use ``@`` to take the inner product of two flat arrays

.. code-block:: python3

    A = np.array((1, 2))
    B = np.array((10, 20))
    A @ B


In fact, we can use ``@`` when one element is a Python list or tuple

.. code-block:: python3

    A = np.array(((1, 2), (3, 4)))
    A

.. code-block:: python3

    A @ (0, 1)

Since we are post-multiplying, the tuple is treated as a column vector.


Mutability and Copying Arrays
-----------------------------

NumPy arrays are mutable data types, like Python lists.

In other words, their contents can be altered (mutated) in memory after initialization.

We already saw examples above.

Here's another example:

.. code-block:: python3

    a = np.array([42, 44])
    a

.. code-block:: python3

    a[-1] = 0  # Change last element to 0
    a

Mutability leads to the following behavior (which can be shocking to MATLAB programmers...)

.. code-block:: python3

    a = np.random.randn(3)
    a

.. code-block:: python3

    b = a
    b[0] = 0.0
    a


What's happened is that we have changed ``a`` by changing ``b``.

The name ``b`` is bound to ``a`` and becomes just another reference to the
array (the Python assignment model is described in more detail :doc:`later in the course <python_advanced_features>`).

Hence, it has equal rights to make changes to that array.

This is in fact the most sensible default behavior!

It means that we pass around only pointers to data, rather than making copies.

Making copies is expensive in terms of both speed and memory.




Making Copies
^^^^^^^^^^^^^

It is of course possible to make ``b`` an independent copy of ``a`` when required.

This can be done using ``np.copy``

.. code-block:: python3

    a = np.random.randn(3)
    a

.. code-block:: python3

    b = np.copy(a)
    b

Now ``b`` is an independent copy (called a *deep copy*)

.. code-block:: python3

    b[:] = 1
    b

.. code-block:: python3

    a

Note that the change to ``b`` has not affected ``a``.


Additional Functionality
========================

Let's look at some other useful things we can do with NumPy.


Vectorized Functions
--------------------

.. index::
    single: NumPy; Vectorized Functions

NumPy provides versions of the standard functions ``log``, ``exp``, ``sin``, etc. that act *element-wise* on arrays

.. code-block:: python3

    z = np.array([1, 2, 3])
    np.sin(z)

This eliminates the need for explicit element-by-element loops such as

.. code-block:: python3

    n = len(z)
    y = np.empty(n)
    for i in range(n):
        y[i] = np.sin(z[i])

Because they act element-wise on arrays, these functions are called *vectorized functions*.

In NumPy-speak, they are also called *ufuncs*, which stands for "universal functions".

As we saw above, the usual arithmetic operations (``+``, ``*``, etc.) also
work element-wise, and combining these with the ufuncs gives a very large set of fast element-wise functions.

.. code-block:: python3

    z

.. code-block:: python3

    (1 / np.sqrt(2 * np.pi)) * np.exp(- 0.5 * z**2)


Not all user-defined functions will act element-wise.

For example, passing the function ``f`` defined below a NumPy array causes a ``ValueError``

.. code-block:: python3

    def f(x):
        return 1 if x > 0 else 0

The NumPy function ``np.where`` provides a vectorized alternative:

.. code-block:: python3

    x = np.random.randn(4)
    x

.. code-block:: python3

    np.where(x > 0, 1, 0)  # Insert 1 if x > 0 true, otherwise 0


You can also use ``np.vectorize`` to vectorize a given function

.. code-block:: python3

    f = np.vectorize(f)
    f(x)                # Passing the same vector x as in the previous example


However, this approach doesn't always obtain the same speed as a more carefully crafted vectorized function.


Comparisons
-----------

.. index::
    single: NumPy; Comparisons

As a rule, comparisons on arrays are done element-wise


.. code-block:: python3

    z = np.array([2, 3])
    y = np.array([2, 3])
    z == y

.. code-block:: python3

    y[0] = 5
    z == y

.. code-block:: python3

    z != y

The situation is similar for ``>``, ``<``, ``>=`` and ``<=``.

We can also do comparisons against scalars

.. code-block:: python3

    z = np.linspace(0, 10, 5)
    z

.. code-block:: python3

    z > 3

This is particularly useful for *conditional extraction*

.. code-block:: python3

    b = z > 3
    b

.. code-block:: python3

    z[b]


Of course we can---and frequently do---perform this in one step

.. code-block:: python3

    z[z > 3]


Sub-packages
------------

NumPy provides some additional functionality related to scientific programming
through its sub-packages.

We've already seen how we can generate random variables using `np.random`

.. code-block:: python3

    z = np.random.randn(10000)  # Generate standard normals
    y = np.random.binomial(10, 0.5, size=1000)    # 1,000 draws from Bin(10, 0.5)
    y.mean()

Another commonly used subpackage is `np.linalg`

.. code-block:: python3

    A = np.array([[1, 2], [3, 4]])

    np.linalg.det(A)           # Compute the determinant

.. code-block:: python3

    np.linalg.inv(A)           # Compute the inverse


.. index::
    single: SciPy

.. index::
    single: Python; SciPy

Much of this functionality is also available in `SciPy <http://www.scipy.org/>`_, a collection of modules that are built on top of NumPy.

We'll cover the SciPy versions in more detail :doc:`soon <scipy>`.

For a comprehensive list of what's available in NumPy see `this documentation <https://docs.scipy.org/doc/numpy/reference/routines.html>`_.


Exercises
=========


.. _np_ex1:

Exercise 1
----------

Consider the polynomial expression

.. math::
    :label: np_polynom

    p(x) = a_0 + a_1 x + a_2 x^2 + \cdots a_N x^N = \sum_{n=0}^N a_n x^n


:ref:`Earlier <pyess_ex2>`, you wrote a simple function ``p(x, coeff)`` to evaluate :eq:`np_polynom` without considering efficiency.

Now write a new function that does the same job, but uses NumPy arrays and array operations for its computations, rather than any form of Python loop.

(Such functionality is already implemented as ``np.poly1d``, but for the sake of the exercise don't use this class)

* Hint: Use ``np.cumprod()``




.. _np_ex2:

Exercise 2
----------

Let ``q`` be a NumPy array of length ``n`` with ``q.sum() == 1``.

Suppose that ``q`` represents a `probability mass function <https://en.wikipedia.org/wiki/Probability_mass_function>`_.

We wish to generate a discrete random variable :math:`x` such that :math:`\mathbb P\{x = i\} = q_i`.

In other words, ``x`` takes values in ``range(len(q))`` and ``x = i`` with probability ``q[i]``.

The standard (inverse transform) algorithm is as follows:

* Divide the unit interval :math:`[0, 1]` into :math:`n` subintervals :math:`I_0, I_1, \ldots, I_{n-1}` such that the length of :math:`I_i` is :math:`q_i`.
* Draw a uniform random variable :math:`U` on :math:`[0, 1]` and return the :math:`i` such that :math:`U \in I_i`.

The probability of drawing :math:`i` is the length of :math:`I_i`, which is equal to :math:`q_i`.

We can implement the algorithm as follows

.. code-block:: python3

    from random import uniform

    def sample(q):
        a = 0.0
        U = uniform(0, 1)
        for i in range(len(q)):
            if a < U <= a + q[i]:
                return i
            a = a + q[i]


If you can't see how this works, try thinking through the flow for a simple example, such as ``q = [0.25, 0.75]``
It helps to sketch the intervals on paper.

Your exercise is to speed it up using NumPy, avoiding explicit loops

* Hint: Use ``np.searchsorted`` and ``np.cumsum``

If you can, implement the functionality as a class called ``DiscreteRV``, where

* the data for an instance of the class is the vector of probabilities ``q``
* the class has a ``draw()`` method, which returns one draw according to the algorithm described above

If you can, write the method so that ``draw(k)`` returns ``k`` draws from ``q``.




.. _np_ex3:

Exercise 3
----------

Recall our :ref:`earlier discussion <oop_ex1>` of the empirical cumulative distribution function.

Your task is to

#. Make the ``__call__`` method more efficient using NumPy.
#. Add a method that plots the ECDF over :math:`[a, b]`, where :math:`a` and :math:`b` are method parameters.


Solutions
=========



.. code-block:: ipython

    import matplotlib.pyplot as plt
    %matplotlib inline

Exercise 1
----------

This code does the job

.. code-block:: python3

    def p(x, coef):
        X = np.ones_like(coef)
        X[1:] = x
        y = np.cumprod(X)   # y = [1, x, x**2,...]
        return coef @ y

Let's test it

.. code-block:: python3

    x = 2
    coef = np.linspace(2, 4, 3)
    print(coef)
    print(p(x, coef))
    # For comparison
    q = np.poly1d(np.flip(coef))
    print(q(x))


Exercise 2
----------

Here's our first pass at a solution:

.. code-block:: python3

    from numpy import cumsum
    from numpy.random import uniform

    class DiscreteRV:
        """
        Generates an array of draws from a discrete random variable with vector of
        probabilities given by q.
        """

        def __init__(self, q):
            """
            The argument q is a NumPy array, or array like, nonnegative and sums
            to 1
            """
            self.q = q
            self.Q = cumsum(q)

        def draw(self, k=1):
            """
            Returns k draws from q. For each such draw, the value i is returned
            with probability q[i].
            """
            return self.Q.searchsorted(uniform(0, 1, size=k))

The logic is not obvious, but if you take your time and read it slowly,
you will understand.

There is a problem here, however.

Suppose that ``q`` is altered after an instance of ``discreteRV`` is
created, for example by

.. code-block:: python3

    q = (0.1, 0.9)
    d = DiscreteRV(q)
    d.q = (0.5, 0.5)

The problem is that ``Q`` does not change accordingly, and ``Q`` is the
data used in the ``draw`` method.

To deal with this, one option is to compute ``Q`` every time the draw
method is called.

But this is inefficient relative to computing ``Q`` once-off.

A better option is to use descriptors.

A solution from the `quantecon
library <https://github.com/QuantEcon/QuantEcon.py/tree/master/quantecon>`__
using descriptors that behaves as we desire can be found
`here <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/discrete_rv.py>`__.

Exercise 3
----------

An example solution is given below.

In essence, we've just taken `this
code <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/ecdf.py>`__
from QuantEcon and added in a plot method

.. code-block:: python3

    """
    Modifies ecdf.py from QuantEcon to add in a plot method

    """

    class ECDF:
        """
        One-dimensional empirical distribution function given a vector of
        observations.

        Parameters
        ----------
        observations : array_like
            An array of observations

        Attributes
        ----------
        observations : array_like
            An array of observations

        """

        def __init__(self, observations):
            self.observations = np.asarray(observations)

        def __call__(self, x):
            """
            Evaluates the ecdf at x

            Parameters
            ----------
            x : scalar(float)
                The x at which the ecdf is evaluated

            Returns
            -------
            scalar(float)
                Fraction of the sample less than x

            """
            return np.mean(self.observations <= x)

        def plot(self, ax, a=None, b=None):
            """
            Plot the ecdf on the interval [a, b].

            Parameters
            ----------
            a : scalar(float), optional(default=None)
                Lower endpoint of the plot interval
            b : scalar(float), optional(default=None)
                Upper endpoint of the plot interval

            """

            # === choose reasonable interval if [a, b] not specified === #
            if a is None:
                a = self.observations.min() - self.observations.std()
            if b is None:
                b = self.observations.max() + self.observations.std()

            # === generate plot === #
            x_vals = np.linspace(a, b, num=100)
            f = np.vectorize(self.__call__)
            ax.plot(x_vals, f(x_vals))
            plt.show()

Here's an example of usage

.. code-block:: python3

    fig, ax = plt.subplots()
    X = np.random.randn(1000)
    F = ECDF(X)
    F.plot(ax)
