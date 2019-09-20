.. _linear_algebra:

.. include:: /_static/includes/header.raw

**************
Linear Algebra
**************

.. index::
    single: Linear Algebra

.. contents:: :depth: 2

Overview
========

Linear algebra is one of the most useful branches of applied mathematics for economists to invest in.

For example, many applied problems in economics and finance require the solution of a linear system of equations, such as

.. math::

    \begin{aligned}
        y_1 = a x_1 + b x_2 \\
        y_2 = c x_1 + d x_2
    \end{aligned}


or, more generally,

.. math::
    :label: la_se

    \begin{aligned}
        y_1 = a_{11} x_1 + a_{12} x_2 + \cdots + a_{1k} x_k \\
        \vdots  \\
        y_n = a_{n1} x_1 + a_{n2} x_2 + \cdots + a_{nk} x_k
    \end{aligned}


The objective here is to solve for the "unknowns" :math:`x_1, \ldots, x_k` given :math:`a_{11}, \ldots, a_{nk}` and :math:`y_1, \ldots, y_n`.

When considering such problems, it is essential that we first consider at least some of the following questions

* Does a solution actually exist?

* Are there in fact many solutions, and if so how should we interpret them?

* If no solution exists, is there a best "approximate" solution?

* If a solution exists, how should we compute it?

These are the kinds of topics addressed by linear algebra.

In this lecture we will cover the basics of linear and matrix algebra, treating both theory and computation.

We admit some overlap with :doc:`this lecture <numpy>`, where operations on NumPy arrays were first explained.

Note that this lecture is more theoretical than most, and contains background
material that will be used in applications as we go along.

Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import interp2d
    from scipy.linalg import inv, solve, det, eig

:index:`Vectors`
================

.. index::
    single: Linear Algebra; Vectors

A *vector* of length :math:`n` is just a sequence (or array, or tuple) of :math:`n` numbers, which we write as :math:`x = (x_1, \ldots, x_n)` or  :math:`x = [x_1, \ldots, x_n]`.

We will write these sequences either horizontally or vertically as we please.

(Later, when we wish to perform certain matrix operations, it will become necessary to distinguish between the two)

The set of all :math:`n`-vectors is denoted by :math:`\mathbb R^n`.

For example, :math:`\mathbb R ^2` is the plane, and a vector in :math:`\mathbb R^2` is just a point in the plane.

Traditionally, vectors are represented visually as arrows from the origin to
the point.

The following figure represents three vectors in this manner



.. code-block:: ipython

  fig, ax = plt.subplots(figsize=(10, 8))
  # Set the axes through the origin
  for spine in ['left', 'bottom']:
      ax.spines[spine].set_position('zero')
  for spine in ['right', 'top']:
      ax.spines[spine].set_color('none')

  ax.set(xlim=(-5, 5), ylim=(-5, 5))
  ax.grid()
  vecs = ((2, 4), (-3, 3), (-4, -3.5))
  for v in vecs:
      ax.annotate('', xy=v, xytext=(0, 0),
                  arrowprops=dict(facecolor='blue',
                  shrink=0,
                  alpha=0.7,
                  width=0.5))
      ax.text(1.1 * v[0], 1.1 * v[1], str(v))
  plt.show()





Vector Operations
-----------------

.. index::
    single: Vectors; Operations

The two most common operators for vectors are addition and scalar multiplication, which we now describe.

As a matter of definition, when we add two vectors, we add them element-by-element

.. math::

    x + y =
    \begin{bmatrix}
        x_1 \\
        x_2 \\
        \vdots \\
        x_n
    \end{bmatrix} +
    \begin{bmatrix}
         y_1 \\
         y_2 \\
        \vdots \\
         y_n
    \end{bmatrix} :=
    \begin{bmatrix}
        x_1 + y_1 \\
        x_2 + y_2 \\
        \vdots \\
        x_n + y_n
    \end{bmatrix}


Scalar multiplication is an operation that takes a number :math:`\gamma` and a
vector :math:`x` and produces

.. math::

    \gamma x :=
    \begin{bmatrix}
        \gamma x_1 \\
        \gamma x_2 \\
        \vdots \\
        \gamma x_n
    \end{bmatrix}


Scalar multiplication is illustrated in the next figure



.. code-block:: python3
  :class: collapse

  fig, ax = plt.subplots(figsize=(10, 8))
  # Set the axes through the origin
  for spine in ['left', 'bottom']:
      ax.spines[spine].set_position('zero')
  for spine in ['right', 'top']:
      ax.spines[spine].set_color('none')

  ax.set(xlim=(-5, 5), ylim=(-5, 5))
  x = (2, 2)
  ax.annotate('', xy=x, xytext=(0, 0),
              arrowprops=dict(facecolor='blue',
              shrink=0,
              alpha=1,
              width=0.5))
  ax.text(x[0] + 0.4, x[1] - 0.2, '$x$', fontsize='16')


  scalars = (-2, 2)
  x = np.array(x)

  for s in scalars:
      v = s * x
      ax.annotate('', xy=v, xytext=(0, 0),
                  arrowprops=dict(facecolor='red',
                  shrink=0,
                  alpha=0.5,
                  width=0.5))
      ax.text(v[0] + 0.4, v[1] - 0.2, f'${s} x$', fontsize='16')
  plt.show()






In Python, a vector can be represented as a list or tuple, such as ``x = (2, 4, 6)``, but is more commonly represented as a :ref:`NumPy array <numpy_array>`.

One advantage of NumPy arrays is that scalar multiplication and addition have very natural syntax

.. code-block:: python3

    x = np.ones(3)            # Vector of three ones
    y = np.array((2, 4, 6))   # Converts tuple (2, 4, 6) into array
    x + y


.. code-block:: python3

    4 * x





Inner Product and Norm
----------------------

.. index::
    single: Vectors; Inner Product

.. index::
    single: Vectors; Norm

The *inner product* of vectors :math:`x,y \in \mathbb R ^n` is defined as

.. math::

    x' y := \sum_{i=1}^n x_i y_i


Two vectors are called *orthogonal* if their inner product is zero.

The *norm* of a vector :math:`x` represents its "length" (i.e., its distance from the zero vector) and is defined as

.. math::

    \| x \| := \sqrt{x' x} := \left( \sum_{i=1}^n x_i^2 \right)^{1/2}


The expression :math:`\| x - y\|` is thought of as the distance between :math:`x` and :math:`y`.

Continuing on from the previous example, the inner product and norm can be computed as
follows




.. code-block:: python3

    np.sum(x * y)          # Inner product of x and y


.. code-block:: python3

    np.sqrt(np.sum(x**2))  # Norm of x, take one


.. code-block:: python3

    np.linalg.norm(x)      # Norm of x, take two





Span
-----

.. index::
    single: Vectors; Span

Given a set of vectors :math:`A := \{a_1, \ldots, a_k\}` in :math:`\mathbb R ^n`, it's natural to think about the new vectors we can create by performing linear operations.

New vectors created in this manner are called *linear combinations* of :math:`A`.

In particular, :math:`y \in \mathbb R ^n` is a linear combination of :math:`A := \{a_1, \ldots, a_k\}` if

.. math::

    y = \beta_1 a_1 + \cdots + \beta_k a_k
    \text{ for some scalars } \beta_1, \ldots, \beta_k


In this context, the values :math:`\beta_1, \ldots, \beta_k` are called the *coefficients* of the linear combination.

The set of linear combinations of :math:`A` is called the *span* of :math:`A`.

The next figure shows the span of :math:`A = \{a_1, a_2\}` in :math:`\mathbb R ^3`.

The span is a two-dimensional plane passing through these two points and the origin.

.. _la_3dvec:

.. code-block:: python3
  :class: collapse

  fig = plt.figure(figsize=(10, 8))
  ax = fig.gca(projection='3d')

  x_min, x_max = -5, 5
  y_min, y_max = -5, 5

  α, β = 0.2, 0.1

  ax.set(xlim=(x_min, x_max), ylim=(x_min, x_max), zlim=(x_min, x_max),
         xticks=(0,), yticks=(0,), zticks=(0,))

  gs = 3
  z = np.linspace(x_min, x_max, gs)
  x = np.zeros(gs)
  y = np.zeros(gs)
  ax.plot(x, y, z, 'k-', lw=2, alpha=0.5)
  ax.plot(z, x, y, 'k-', lw=2, alpha=0.5)
  ax.plot(y, z, x, 'k-', lw=2, alpha=0.5)


  # Fixed linear function, to generate a plane
  def f(x, y):
      return α * x + β * y

  # Vector locations, by coordinate
  x_coords = np.array((3, 3))
  y_coords = np.array((4, -4))
  z = f(x_coords, y_coords)
  for i in (0, 1):
      ax.text(x_coords[i], y_coords[i], z[i], f'$a_{i+1}$', fontsize=14)

  # Lines to vectors
  for i in (0, 1):
      x = (0, x_coords[i])
      y = (0, y_coords[i])
      z = (0, f(x_coords[i], y_coords[i]))
      ax.plot(x, y, z, 'b-', lw=1.5, alpha=0.6)


  # Draw the plane
  grid_size = 20
  xr2 = np.linspace(x_min, x_max, grid_size)
  yr2 = np.linspace(y_min, y_max, grid_size)
  x2, y2 = np.meshgrid(xr2, yr2)
  z2 = f(x2, y2)
  ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.jet,
                  linewidth=0, antialiased=True, alpha=0.2)
  plt.show()





Examples
^^^^^^^^

If :math:`A` contains only one vector :math:`a_1 \in \mathbb R ^2`, then its
span is just the scalar multiples of :math:`a_1`, which is the unique line passing through both :math:`a_1` and the origin.

If :math:`A = \{e_1, e_2, e_3\}` consists  of the *canonical basis vectors* of :math:`\mathbb R ^3`, that is

.. math::

    e_1 :=
    \begin{bmatrix}
         1 \\
         0 \\
         0
    \end{bmatrix}
    , \quad
    e_2 :=
    \begin{bmatrix}
         0 \\
         1 \\
         0
    \end{bmatrix}
    , \quad
    e_3 :=
    \begin{bmatrix}
         0 \\
         0 \\
         1
    \end{bmatrix}


then the span of :math:`A` is all of :math:`\mathbb R ^3`, because, for any
:math:`x = (x_1, x_2, x_3) \in \mathbb R ^3`, we can write

.. math::

    x = x_1 e_1 + x_2 e_2 + x_3 e_3


Now consider :math:`A_0 = \{e_1, e_2, e_1 + e_2\}`.

If :math:`y = (y_1, y_2, y_3)` is any linear combination of these vectors, then :math:`y_3 = 0` (check it).

Hence :math:`A_0` fails to span all of :math:`\mathbb R ^3`.

.. _la_li:

Linear Independence
-------------------

.. index::
    single: Vectors; Linear Independence

As we'll see, it's often desirable to find families of vectors with relatively large span, so that many vectors can be described by linear operators on a few vectors.

The condition we need for a set of vectors to have a large span is what's called linear independence.

In particular, a collection of vectors :math:`A := \{a_1, \ldots, a_k\}` in :math:`\mathbb R ^n` is said to be

* *linearly dependent* if some strict subset of :math:`A` has the same span as :math:`A`.

* *linearly independent* if it is not linearly dependent.

Put differently, a set of vectors is linearly independent if no vector is redundant to the span and linearly dependent otherwise.

To illustrate the idea, recall :ref:`the figure <la_3dvec>` that showed the span of vectors :math:`\{a_1, a_2\}` in :math:`\mathbb R ^3` as a plane through the origin.

If we take a third vector :math:`a_3` and form the set :math:`\{a_1, a_2, a_3\}`, this set will be

* linearly dependent if :math:`a_3` lies in the plane

* linearly independent otherwise

As another illustration of the concept, since :math:`\mathbb R ^n` can be spanned by :math:`n` vectors
(see the discussion of canonical basis vectors above), any collection of
:math:`m > n` vectors in :math:`\mathbb R ^n` must be linearly dependent.

The following statements are equivalent to linear independence of :math:`A := \{a_1, \ldots, a_k\} \subset \mathbb R ^n`

#. No vector in :math:`A` can be formed as a linear combination of the other elements.

#. If :math:`\beta_1 a_1 + \cdots \beta_k a_k = 0` for scalars :math:`\beta_1, \ldots, \beta_k`, then :math:`\beta_1 = \cdots = \beta_k = 0`.

(The zero in the first expression is the origin of :math:`\mathbb R ^n`)


.. _la_unique_reps:

Unique Representations
----------------------

Another nice thing about sets of linearly independent vectors is that each element in the span has a unique representation as a linear combination of these vectors.

In other words, if :math:`A := \{a_1, \ldots, a_k\} \subset \mathbb R ^n` is
linearly independent and

.. math::

    y = \beta_1 a_1 + \cdots \beta_k a_k


then no other coefficient sequence :math:`\gamma_1, \ldots, \gamma_k` will produce
the same vector :math:`y`.

Indeed, if we also have :math:`y = \gamma_1 a_1 + \cdots \gamma_k a_k`,
then

.. math::

    (\beta_1 - \gamma_1) a_1 + \cdots + (\beta_k - \gamma_k) a_k = 0


Linear independence now implies :math:`\gamma_i = \beta_i` for all :math:`i`.



Matrices
========

.. index::
    single: Linear Algebra; Matrices

Matrices are a neat way of organizing data for use in linear operations.

An :math:`n \times k` matrix is a rectangular array :math:`A` of numbers with :math:`n` rows and :math:`k` columns:

.. math::

    A =
    \begin{bmatrix}
        a_{11} & a_{12} & \cdots & a_{1k} \\
        a_{21} & a_{22} & \cdots & a_{2k} \\
        \vdots & \vdots &  & \vdots \\
        a_{n1} & a_{n2} & \cdots & a_{nk}
    \end{bmatrix}


Often, the numbers in the matrix represent coefficients in a system of linear equations, as discussed at the start of this lecture.

For obvious reasons, the matrix :math:`A` is also called a vector if either :math:`n = 1` or :math:`k = 1`.

In the former case, :math:`A` is called a *row vector*, while in the latter it is called a *column vector*.

If :math:`n = k`, then :math:`A` is called *square*.

The matrix formed by replacing :math:`a_{ij}` by :math:`a_{ji}` for every :math:`i` and :math:`j` is called the *transpose* of :math:`A` and denoted :math:`A'` or :math:`A^{\top}`.

If :math:`A = A'`, then :math:`A` is called *symmetric*.

For a square matrix :math:`A`, the :math:`i` elements of the form :math:`a_{ii}` for :math:`i=1,\ldots,n` are called the *principal diagonal*.

:math:`A` is called *diagonal* if the only nonzero entries are on the principal diagonal.

If, in addition to being diagonal, each element along the principal diagonal is equal to 1, then :math:`A` is called the *identity matrix* and denoted by :math:`I`.




Matrix Operations
-----------------

.. index::
    single: Matrix; Operations

Just as was the case for vectors, a number of algebraic operations are defined for matrices.

Scalar multiplication and addition are immediate generalizations of the vector case:

.. math::

    \gamma A =
    \gamma
    \begin{bmatrix}
        a_{11} &  \cdots & a_{1k} \\
        \vdots & \vdots  & \vdots \\
        a_{n1} &  \cdots & a_{nk}
    \end{bmatrix} :=
    \begin{bmatrix}
        \gamma a_{11} & \cdots & \gamma a_{1k} \\
        \vdots & \vdots & \vdots \\
        \gamma a_{n1} & \cdots & \gamma a_{nk}
    \end{bmatrix}


and

.. math::

    A + B =
    \begin{bmatrix}
        a_{11} & \cdots & a_{1k} \\
        \vdots & \vdots & \vdots \\
        a_{n1} & \cdots & a_{nk}
    \end{bmatrix} +
    \begin{bmatrix}
        b_{11} & \cdots & b_{1k} \\
        \vdots & \vdots & \vdots \\
        b_{n1} & \cdots & b_{nk}
    \end{bmatrix} :=
    \begin{bmatrix}
        a_{11} + b_{11} &  \cdots & a_{1k} + b_{1k} \\
        \vdots & \vdots & \vdots \\
        a_{n1} + b_{n1} &  \cdots & a_{nk} + b_{nk}
    \end{bmatrix}


In the latter case, the matrices must have the same shape in order for the definition to make sense.

We also have a convention for *multiplying* two matrices.

The rule for matrix multiplication generalizes the idea of inner products discussed above
and is designed to make multiplication play well with basic linear operations.

If :math:`A` and :math:`B` are two matrices, then their product :math:`A B` is formed by taking as its
:math:`i,j`-th element the inner product of the :math:`i`-th row of :math:`A` and the
:math:`j`-th column of :math:`B`.

There are many tutorials to help you visualize this operation, such as `this one <http://www.mathsisfun.com/algebra/matrix-multiplying.html>`_, or the discussion on the `Wikipedia page <https://en.wikipedia.org/wiki/Matrix_multiplication>`_.

If :math:`A` is :math:`n \times k` and :math:`B` is :math:`j \times m`, then
to multiply :math:`A` and :math:`B` we require :math:`k = j`, and the
resulting matrix :math:`A B` is :math:`n \times m`.

As perhaps the most important special case, consider multiplying :math:`n \times k` matrix :math:`A` and :math:`k \times 1` column vector :math:`x`.

According to the preceding rule, this gives us an :math:`n \times 1` column vector

.. math::
    :label: la_atx

    A x =
    \begin{bmatrix}
        a_{11} &  \cdots & a_{1k} \\
        \vdots & \vdots  & \vdots \\
        a_{n1} &  \cdots & a_{nk}
    \end{bmatrix}
    \begin{bmatrix}
        x_{1}  \\
        \vdots  \\
        x_{k}
    \end{bmatrix} :=
    \begin{bmatrix}
        a_{11} x_1 + \cdots + a_{1k} x_k \\
        \vdots \\
        a_{n1} x_1 + \cdots + a_{nk} x_k
    \end{bmatrix}


.. note::

    :math:`A B` and :math:`B A` are not generally the same thing.

Another important special case is the identity matrix.

You should check that if :math:`A` is :math:`n \times k` and :math:`I` is the :math:`k \times k` identity matrix, then :math:`AI = A`.

If :math:`I` is the :math:`n \times n` identity matrix, then :math:`IA = A`.




Matrices in NumPy
-----------------

.. index::
    single: Matrix; Numpy

NumPy arrays are also used as matrices, and have fast, efficient functions and methods for all the standard matrix operations [#fn_mdt]_.


You can create them manually from tuples of tuples (or lists of lists) as follows

.. code-block:: python3

    A = ((1, 2),
         (3, 4))

    type(A)


.. code-block:: python3

    A = np.array(A)

    type(A)


.. code-block:: python3

    A.shape


The ``shape`` attribute is a tuple giving the number of rows and columns ---
see :ref:`here <numpy_shape_dim>` for more discussion.

To get the transpose of ``A``, use ``A.transpose()`` or, more simply, ``A.T``.



There are many convenient functions for creating common matrices (matrices of zeros, ones, etc.) --- see :ref:`here <creating_arrays>`.



Since operations are performed elementwise by default, scalar multiplication and addition have very natural syntax

.. code-block:: python3

    A = np.identity(3)
    B = np.ones((3, 3))
    2 * A


.. code-block:: python3

    A + B



To multiply matrices we use the ``@`` symbol.

In particular, ``A @ B`` is matrix multiplication, whereas ``A * B`` is element-by-element multiplication.

See :ref:`here <numpy_matrix_multiplication>` for more discussion.





.. _la_linear_map:

Matrices as Maps
----------------

.. index::
    single: Matrix; Maps

Each :math:`n \times k` matrix :math:`A` can be identified with a function :math:`f(x) = Ax` that maps :math:`x \in \mathbb R ^k` into :math:`y = Ax \in \mathbb R ^n`.

These kinds of functions have a special property: they are *linear*.

A function :math:`f \colon \mathbb R ^k \to \mathbb R ^n` is called *linear* if, for all :math:`x, y \in \mathbb R ^k` and all scalars :math:`\alpha, \beta`, we have

.. math::

    f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)


You can check that this holds for the function :math:`f(x) = A x + b` when :math:`b` is the zero vector and fails when :math:`b` is nonzero.

In fact, it's `known <https://en.wikipedia.org/wiki/Linear_map#Matrices>`_ that :math:`f` is linear if and *only if* there exists a matrix :math:`A` such that :math:`f(x) = Ax` for all :math:`x`.




Solving Systems of Equations
============================

.. index::
    single: Matrix; Solving Systems of Equations

Recall again the system of equations :eq:`la_se`.

If we compare :eq:`la_se` and :eq:`la_atx`, we see that :eq:`la_se` can now be
written more conveniently as

.. math::
    :label: la_se2

    y = Ax


The problem we face is to determine a vector :math:`x \in \mathbb R ^k` that solves :eq:`la_se2`, taking :math:`y` and :math:`A` as given.

This is a special case of a more general problem: Find an :math:`x` such that :math:`y = f(x)`.

Given an arbitrary function :math:`f` and a :math:`y`, is there always an :math:`x` such that :math:`y = f(x)`?

If so, is it always unique?

The answer to both these questions is negative, as the next figure shows



.. code-block:: python3
  :class: collapse

  def f(x):
      return 0.6 * np.cos(4 * x) + 1.4


  xmin, xmax = -1, 1
  x = np.linspace(xmin, xmax, 160)
  y = f(x)
  ya, yb = np.min(y), np.max(y)

  fig, axes = plt.subplots(2, 1, figsize=(10, 10))

  for ax in axes:
      # Set the axes through the origin
      for spine in ['left', 'bottom']:
          ax.spines[spine].set_position('zero')
      for spine in ['right', 'top']:
          ax.spines[spine].set_color('none')

      ax.set(ylim=(-0.6, 3.2), xlim=(xmin, xmax),
             yticks=(), xticks=())

      ax.plot(x, y, 'k-', lw=2, label='$f$')
      ax.fill_between(x, ya, yb, facecolor='blue', alpha=0.05)
      ax.vlines([0], ya, yb, lw=3, color='blue', label='range of $f$')
      ax.text(0.04, -0.3, '$0$', fontsize=16)

  ax = axes[0]

  ax.legend(loc='upper right', frameon=False)
  ybar = 1.5
  ax.plot(x, x * 0 + ybar, 'k--', alpha=0.5)
  ax.text(0.05, 0.8 * ybar, '$y$', fontsize=16)
  for i, z in enumerate((-0.35, 0.35)):
      ax.vlines(z, 0, f(z), linestyle='--', alpha=0.5)
      ax.text(z, -0.2, f'$x_{i}$', fontsize=16)

  ax = axes[1]

  ybar = 2.6
  ax.plot(x, x * 0 + ybar, 'k--', alpha=0.5)
  ax.text(0.04, 0.91 * ybar, '$y$', fontsize=16)

  plt.show()





In the first plot, there are multiple solutions, as the function is not one-to-one, while
in the second there are no solutions, since :math:`y` lies outside the range of :math:`f`.

Can we impose conditions on :math:`A` in :eq:`la_se2` that rule out these problems?

In this context, the most important thing to recognize about the expression
:math:`Ax` is that it corresponds to a linear combination of the columns of :math:`A`.

In particular, if :math:`a_1, \ldots, a_k` are the columns of :math:`A`, then

.. math::

    Ax = x_1 a_1 + \cdots + x_k a_k


Hence the range of :math:`f(x) = Ax` is exactly the span of the columns of :math:`A`.

We want the range to be large so that it contains arbitrary :math:`y`.

As you might recall, the condition that we want for the span to be large is :ref:`linear independence <la_li>`.

A happy fact is that linear independence of the columns of :math:`A` also gives us uniqueness.

Indeed, it follows from our :ref:`earlier discussion <la_unique_reps>` that if :math:`\{a_1, \ldots, a_k\}` are linearly independent and :math:`y = Ax = x_1 a_1 + \cdots + x_k a_k`, then no :math:`z \not= x` satisfies :math:`y = Az`.


The Square Matrix Case
----------------------

Let's discuss some more details, starting with the case where :math:`A` is :math:`n \times n`.

This is the familiar case where the number of unknowns equals the number of equations.

For arbitrary :math:`y \in \mathbb R ^n`, we hope to find a unique :math:`x \in \mathbb R ^n` such that :math:`y = Ax`.

In view of the observations immediately above, if the columns of :math:`A` are
linearly independent, then their span, and hence the range of :math:`f(x) =
Ax`, is all of :math:`\mathbb R ^n`.

Hence there always exists an :math:`x` such that :math:`y = Ax`.

Moreover, the solution is unique.

In particular, the following are equivalent

#. The columns of :math:`A` are linearly independent.

#. For any :math:`y \in \mathbb R ^n`, the equation :math:`y = Ax` has a unique solution.

The property of having linearly independent columns is sometimes expressed as having *full column rank*.

Inverse Matrices
^^^^^^^^^^^^^^^^

.. index::
    single: Matrix; Inverse

Can we give some sort of expression for the solution?

If :math:`y` and :math:`A` are scalar with :math:`A \not= 0`, then the
solution is :math:`x = A^{-1} y`.

A similar expression is available in the matrix case.

In particular, if square matrix :math:`A` has full column rank, then it possesses a multiplicative
*inverse matrix* :math:`A^{-1}`, with the property that :math:`A A^{-1} = A^{-1} A = I`.

As a consequence, if we pre-multiply both sides of :math:`y = Ax` by :math:`A^{-1}`, we get :math:`x = A^{-1} y`.

This is the solution that we're looking for.

Determinants
^^^^^^^^^^^^

.. index::
    single: Matrix; Determinants

Another quick comment about square matrices is that to every such matrix we
assign a unique number called the *determinant* of the matrix --- you can find
the expression for it `here <https://en.wikipedia.org/wiki/Determinant>`__.

If the determinant of :math:`A` is not zero, then we say that :math:`A` is
*nonsingular*.

Perhaps the most important fact about determinants is that :math:`A` is nonsingular if and only if :math:`A` is of full column rank.

This gives us a useful one-number summary of whether or not a square matrix can be
inverted.


More Rows than Columns
----------------------

This is the :math:`n \times k` case with :math:`n > k`.

This case is very important in many settings, not least in the setting of linear regression (where :math:`n` is the number of observations, and :math:`k` is the number of explanatory variables).

Given arbitrary :math:`y \in \mathbb R ^n`, we seek an :math:`x \in \mathbb R ^k` such that :math:`y = Ax`.

In this setting, the existence of a solution is highly unlikely.

Without much loss of generality, let's go over the intuition focusing on the case where the columns of
:math:`A` are linearly independent.

It follows that the span of the columns of :math:`A` is a :math:`k`-dimensional subspace of :math:`\mathbb R ^n`.

This span is very "unlikely" to contain arbitrary :math:`y \in \mathbb R ^n`.

To see why, recall the :ref:`figure above <la_3dvec>`, where :math:`k=2` and :math:`n=3`.

Imagine an arbitrarily chosen :math:`y \in \mathbb R ^3`, located somewhere in that three-dimensional space.

What's the likelihood that :math:`y` lies in the span of :math:`\{a_1, a_2\}` (i.e., the two dimensional plane through these points)?

In a sense, it must be very small, since this plane has zero "thickness".

As a result, in the :math:`n > k` case we usually give up on existence.

However, we can still seek the best approximation, for example, an
:math:`x` that makes the distance :math:`\| y - Ax\|` as small as possible.

To solve this problem, one can use either calculus or the theory of orthogonal
projections.

.. only:: html

    The solution is known to be :math:`\hat x = (A'A)^{-1}A'y` --- see for example
    chapter 3 of :download:`these notes <_static/lecture_specific/linear_algebra/course_notes.pdf>`.

.. only:: latex

    The solution is known to be :math:`\hat x = (A'A)^{-1}A'y` --- see for example
    chapter 3 of `these notes <https://lectures.quantecon.org/_downloads/course_notes.pdf>`__.

More Columns than Rows
----------------------


This is the :math:`n \times k` case with :math:`n < k`, so there are fewer
equations than unknowns.

In this case there are either no solutions or infinitely many --- in other words, uniqueness never holds.

For example, consider the case where :math:`k=3` and :math:`n=2`.

Thus, the columns of :math:`A` consists of 3 vectors in :math:`\mathbb R ^2`.

This set can never be linearly independent, since it is possible to find two vectors that span
:math:`\mathbb R ^2`.

(For example, use the canonical basis vectors)

It follows that one column is a linear combination of the other two.

For example, let's say that :math:`a_1 = \alpha a_2 + \beta a_3`.

Then if :math:`y = Ax = x_1 a_1 + x_2 a_2 + x_3 a_3`, we can also write

.. math::

    y
    = x_1 (\alpha a_2 + \beta a_3) + x_2 a_2 + x_3 a_3
    = (x_1 \alpha + x_2) a_2 + (x_1 \beta + x_3) a_3


In other words, uniqueness fails.




Linear Equations with SciPy
---------------------------

.. index::
    single: Linear Algebra; SciPy

Here's an illustration of how to solve linear equations with SciPy's ``linalg`` submodule.

All of these routines are Python front ends to time-tested and highly optimized FORTRAN code

.. code-block:: python3

    A = ((1, 2), (3, 4))
    A = np.array(A)
    y = np.ones((2, 1))  # Column vector
    det(A)  # Check that A is nonsingular, and hence invertible


.. code-block:: python3

    A_inv = inv(A)  # Compute the inverse
    A_inv


.. code-block:: python3

    x = A_inv @ y  # Solution
    A @ x          # Should equal y


.. code-block:: python3

    solve(A, y)  # Produces the same solution



Observe how we can solve for :math:`x = A^{-1} y` by either via ``inv(A) @ y``, or using ``solve(A, y)``.

The latter method uses a different algorithm (LU decomposition) that is numerically more stable, and hence should almost always be preferred.

To obtain the least-squares solution :math:`\hat x = (A'A)^{-1}A'y`, use ``scipy.linalg.lstsq(A, y)``.



.. _la_eigen:

:index:`Eigenvalues` and :index:`Eigenvectors`
==============================================

.. index::
    single: Linear Algebra; Eigenvalues

.. index::
    single: Linear Algebra; Eigenvectors


Let :math:`A` be an :math:`n \times n` square matrix.

If :math:`\lambda` is scalar and :math:`v` is a non-zero vector in :math:`\mathbb R ^n` such that

.. math::

    A v = \lambda v


then we say that :math:`\lambda` is an *eigenvalue* of :math:`A`, and
:math:`v` is an *eigenvector*.

Thus, an eigenvector of :math:`A` is a vector such that when the map :math:`f(x) = Ax` is applied, :math:`v` is merely scaled.

The next figure shows two eigenvectors (blue arrows) and their images under :math:`A` (red arrows).

As expected, the image :math:`Av` of each :math:`v` is just a scaled version of the original



.. code-block:: python3
  :class: collapse

  A = ((1, 2),
       (2, 1))
  A = np.array(A)
  evals, evecs = eig(A)
  evecs = evecs[:, 0], evecs[:, 1]

  fig, ax = plt.subplots(figsize=(10, 8))
  # Set the axes through the origin
  for spine in ['left', 'bottom']:
      ax.spines[spine].set_position('zero')
  for spine in ['right', 'top']:
      ax.spines[spine].set_color('none')
  ax.grid(alpha=0.4)

  xmin, xmax = -3, 3
  ymin, ymax = -3, 3
  ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

  # Plot each eigenvector
  for v in evecs:
      ax.annotate('', xy=v, xytext=(0, 0),
                  arrowprops=dict(facecolor='blue',
                  shrink=0,
                  alpha=0.6,
                  width=0.5))

  # Plot the image of each eigenvector
  for v in evecs:
      v = A @ v
      ax.annotate('', xy=v, xytext=(0, 0),
                  arrowprops=dict(facecolor='red',
                  shrink=0,
                  alpha=0.6,
                  width=0.5))

  # Plot the lines they run through
  x = np.linspace(xmin, xmax, 3)
  for v in evecs:
      a = v[1] / v[0]
      ax.plot(x, a * x, 'b-', lw=0.4)

  plt.show()




The eigenvalue equation is equivalent to :math:`(A - \lambda I) v = 0`, and
this has a nonzero solution :math:`v` only when the columns of :math:`A -
\lambda I` are linearly dependent.

This in turn is equivalent to stating that the determinant is zero.

Hence to find all eigenvalues, we can look for :math:`\lambda` such that the
determinant of :math:`A - \lambda I` is zero.

This problem can be expressed as one of solving for the roots of a polynomial
in :math:`\lambda` of degree :math:`n`.

This in turn implies the existence of :math:`n` solutions in the complex
plane, although some might be repeated.

Some nice facts about the eigenvalues of a square matrix :math:`A` are as follows

#. The determinant of :math:`A` equals  the product of the eigenvalues.

#. The trace of :math:`A` (the sum of the elements on the principal diagonal) equals the sum of the eigenvalues.

#. If :math:`A` is symmetric, then all of its eigenvalues are real.

#. If :math:`A` is invertible and :math:`\lambda_1, \ldots, \lambda_n` are its eigenvalues, then the eigenvalues of :math:`A^{-1}` are :math:`1/\lambda_1, \ldots, 1/\lambda_n`.

A corollary of the first statement is that a matrix is invertible if and only if all its eigenvalues are nonzero.




Using SciPy, we can solve for the eigenvalues and eigenvectors of a matrix as
follows

.. code-block:: python3

    A = ((1, 2),
         (2, 1))

    A = np.array(A)
    evals, evecs = eig(A)
    evals


.. code-block:: python3

    evecs




Note that the *columns* of ``evecs`` are the eigenvectors.

Since any scalar multiple of an eigenvector is an eigenvector with the same
eigenvalue (check it), the eig routine normalizes the length of each eigenvector
to one.

Generalized Eigenvalues
-----------------------

It is sometimes useful to consider the *generalized eigenvalue problem*, which, for given
matrices :math:`A` and :math:`B`, seeks generalized eigenvalues
:math:`\lambda` and eigenvectors :math:`v` such that

.. math::

    A v = \lambda B v


This can be solved in SciPy via ``scipy.linalg.eig(A, B)``.

Of course, if :math:`B` is square and invertible, then we can treat the
generalized eigenvalue problem as an ordinary eigenvalue problem :math:`B^{-1}
A v = \lambda v`, but this is not always the case.


Further Topics
==============

We round out our discussion by briefly mentioning several other important
topics.


Series Expansions
-----------------

.. index::
    single: Linear Algebra; Series Expansions

Recall the usual summation formula for a geometric progression, which states
that if :math:`|a| < 1`, then :math:`\sum_{k=0}^{\infty} a^k = (1 - a)^{-1}`.

A generalization of this idea exists in the matrix setting.

.. _la_mn:

Matrix Norms
^^^^^^^^^^^^

.. index::
    single: Linear Algebra; Matrix Norms

Let :math:`A` be a square matrix, and let

.. math::

    \| A \| := \max_{\| x \| = 1} \| A x \|


The norms on the right-hand side are ordinary vector norms, while the norm on
the left-hand side is a *matrix norm* --- in this case, the so-called
*spectral norm*.


For example, for a square matrix :math:`S`, the condition :math:`\| S \| < 1` means that :math:`S` is *contractive*, in the sense that it pulls all vectors towards the origin [#cfn]_.

.. _la_neumann:

:index:`Neumann's Theorem`
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. index::
    single: Linear Algebra; Neumann's Theorem

Let :math:`A` be a square matrix and let :math:`A^k := A A^{k-1}` with :math:`A^1 := A`.

In other words, :math:`A^k` is the :math:`k`-th power of :math:`A`.

Neumann's theorem states the following: If :math:`\| A^k \| < 1` for some
:math:`k \in \mathbb{N}`, then :math:`I - A` is invertible, and

.. math::
    :label: la_neumann

    (I - A)^{-1} = \sum_{k=0}^{\infty} A^k


.. _la_neumann_remarks:

:index:`Spectral Radius`
^^^^^^^^^^^^^^^^^^^^^^^^

.. index::
    single: Linear Algebra; Spectral Radius

A result known as Gelfand's formula tells us that, for any square matrix :math:`A`,

.. math::

    \rho(A) = \lim_{k \to \infty} \| A^k \|^{1/k}


Here :math:`\rho(A)` is the *spectral radius*, defined as :math:`\max_i |\lambda_i|`, where :math:`\{\lambda_i\}_i` is the set of eigenvalues of :math:`A`.

As a consequence of Gelfand's formula, if all eigenvalues are strictly less than one in modulus,
there exists a :math:`k` with :math:`\| A^k \| < 1`.

In which case :eq:`la_neumann` is valid.


:index:`Positive Definite Matrices`
-----------------------------------

.. index::
    single: Linear Algebra; Positive Definite Matrices

Let :math:`A` be a symmetric :math:`n \times n` matrix.

We say that :math:`A` is

#. *positive definite* if :math:`x' A x > 0` for every :math:`x \in \mathbb R ^n \setminus \{0\}`

#. *positive semi-definite* or *nonnegative definite* if :math:`x' A x \geq 0` for every :math:`x \in \mathbb R ^n`

Analogous definitions exist for negative definite and negative semi-definite matrices.

It is notable that if :math:`A` is positive definite, then all of its eigenvalues
are strictly positive, and hence :math:`A` is invertible (with positive
definite inverse).


.. _la_mcalc:

Differentiating Linear and Quadratic Forms
------------------------------------------

.. index::
    single: Linear Algebra; Differentiating Linear and Quadratic Forms

The following formulas are useful in many economic contexts.  Let

* :math:`z, x` and :math:`a` all be :math:`n \times 1` vectors

* :math:`A` be an :math:`n  \times n` matrix

* :math:`B` be an :math:`m \times n` matrix and :math:`y` be an :math:`m  \times 1` vector

Then

#. :math:`\frac{\partial a' x}{\partial x} = a`

#. :math:`\frac{\partial A x}{\partial x} = A'`

#. :math:`\frac{\partial x'A x}{\partial x} = (A + A') x`

#. :math:`\frac{\partial y'B z}{\partial y} = B z`

#. :math:`\frac{\partial y'B z}{\partial B} = y z'`


Exercise 1 below asks you to apply these formulas.


Further Reading
---------------



The documentation of the ``scipy.linalg`` submodule can be found `here <http://docs.scipy.org/doc/scipy/reference/linalg.html>`__.



Chapters 2 and 3 of the `Econometric Theory <http://www.johnstachurski.net/emet.html>`_ contains
a discussion of linear algebra along the same lines as above, with solved exercises.

If you don't mind a slightly abstract approach, a nice intermediate-level text on linear algebra
is :cite:`Janich1994`.



Exercises
=========


Exercise 1
----------

Let :math:`x` be a given :math:`n \times 1` vector and consider the problem

.. math::

    v(x) =  \max_{y,u} \left\{ - y'P y - u' Q u \right\}


subject to the linear constraint

.. math::

    y = A x + B u


Here

* :math:`P` is an :math:`n \times n` matrix and :math:`Q` is an :math:`m \times m` matrix

* :math:`A` is an :math:`n \times n` matrix and :math:`B` is an :math:`n \times m` matrix

* both :math:`P` and :math:`Q` are symmetric and positive semidefinite

(What must the dimensions of :math:`y` and :math:`u` be to make this a well-posed problem?)

One way to solve the problem is to form the Lagrangian

.. math::

    \mathcal L = - y' P y - u' Q u + \lambda' \left[A x + B u - y\right]


where :math:`\lambda` is an :math:`n \times 1` vector of Lagrange multipliers.

Try applying the formulas given above for differentiating quadratic and linear forms to obtain the first-order conditions for maximizing :math:`\mathcal L` with respect to :math:`y, u` and minimizing it with respect to :math:`\lambda`.

Show that these conditions imply that

#. :math:`\lambda = - 2 P y`.

#. The optimizing choice of :math:`u` satisfies :math:`u = - (Q + B' P B)^{-1} B' P A x`.

#. The function :math:`v` satisfies :math:`v(x) = - x' \tilde P x` where :math:`\tilde P = A' P A - A'P B (Q + B'P B)^{-1} B' P A`.

As we will see, in economic contexts Lagrange multipliers often are shadow prices.

.. note::
    If we don't care about the Lagrange multipliers, we can substitute the constraint into the objective function, and then just maximize :math:`-(Ax + Bu)'P (Ax + Bu) - u' Q u` with respect to :math:`u`.  You can verify that this leads to the same maximizer.


Solutions
=========




Solution to Exercise 1
----------------------

We have an optimization problem:

.. math::

    v(x) = \max_{y,u} \{ -y'Py - u'Qu \}

s.t.

.. math::

    y = Ax + Bu

with primitives

-  :math:`P` be a symmetric and positive semidefinite :math:`n \times n`
   matrix

-  :math:`Q` be a symmetric and positive semidefinite :math:`m \times m`
   matrix

-  :math:`A` an :math:`n \times n` matrix

-  :math:`B` an :math:`n \times m` matrix

The associated Lagrangian is:

.. math::

    L = -y'Py - u'Qu + \lambda' \lbrack Ax + Bu - y \rbrack

1.
^^.

Differentiating Lagrangian equation w.r.t y and setting its derivative
equal to zero yields

.. math::

    \frac{ \partial L}{\partial y} = - (P + P') y - \lambda = - 2 P y - \lambda = 0 \:,

since P is symmetric.

Accordingly, the first-order condition for maximizing L w.r.t. y implies

.. math::

    \lambda = -2 Py \:

2.
^^.

Differentiating Lagrangian equation w.r.t. u and setting its derivative
equal to zero yields

.. math::

    \frac{ \partial L}{\partial u} = - (Q + Q') u - B'\lambda = - 2Qu + B'\lambda = 0 \:

Substituting :math:`\lambda = -2 P y` gives

.. math::

    Qu + B'Py = 0 \:

Substituting the linear constraint :math:`y = Ax + Bu` into above
equation gives

.. math::

    Qu + B'P(Ax + Bu) = 0

.. math::

    (Q + B'PB)u + B'PAx = 0

which is the first-order condition for maximizing L w.r.t. u.

Thus, the optimal choice of u must satisfy

.. math::

    u = -(Q + B'PB)^{-1}B'PAx \:,

which follows from the definition of the first-order conditions for
Lagrangian equation.

3.
^^.

Rewriting our problem by substituting the constraint into the objective
function, we get

.. math::

    v(x) = \max_{u} \{ -(Ax+ Bu)'P(Ax+Bu) - u'Qu \} \:

Since we know the optimal choice of u satisfies :math:`u = -(Q +
B'PB)^{-1}B'PAx`, then

.. math::

    v(x) =  -(Ax+ B u)'P(Ax+B u) - u'Q u  \,\,\,\, with \,\,\,\, u = -(Q + B'PB)^{-1}B'PAx

To evaluate the function

.. math::

   \begin{aligned}
   v(x) &=  -(Ax+ B u)'P(Ax+Bu) - u'Q u \\
   &= -(x'A' + u'B')P(Ax+Bu) - u'Q u \\
   &= - x'A'PAx - u'B'PAx - x'A'PBu - u'B'PBu - u'Qu \\
   &= - x'A'PAx - 2u'B'PAx - u'(Q + B'PB) u
   \end{aligned}

For simplicity, denote by :math:`S := (Q + B'PB)^{-1} B'PA`, then :math:`u = -Sx`.

Regarding the second term :math:`- 2u'B'PAx`,

.. math::

   \begin{aligned}
   -2u'B'PAx &= -2 x'S'B'PAx  \\
   & = 2 x'A'PB( Q + B'PB)^{-1} B'PAx
   \end{aligned}

Notice that the term :math:`(Q + B'PB)^{-1}` is symmetric as both P and Q
are symmetric.

Regarding the third term :math:`- u'(Q + B'PB) u`,

.. math::

   \begin{aligned}
   -u'(Q + B'PB) u &= - x'S' (Q + B'PB)Sx \\
   &= -x'A'PB(Q + B'PB)^{-1}B'PAx
   \end{aligned}

Hence, the summation of second and third terms is
:math:`x'A'PB(Q + B'PB)^{-1}B'PAx`.

This implies that

.. math::

   \begin{aligned}
    v(x) &= - x'A'PAx - 2u'B'PAx - u'(Q + B'PB) u\\
    &= - x'A'PAx + x'A'PB(Q + B'PB)^{-1}B'PAx \\
    &= -x'[A'PA - A'PB(Q + B'PB)^{-1}B'PA] x
   \end{aligned}

Therefore, the solution to the optimization problem
:math:`v(x) = -x' \tilde{P}x` follows the above result by denoting
:math:`\tilde{P} := A'PA - A'PB(Q + B'PB)^{-1}B'PA`




.. rubric:: Footnotes



.. [#fn_mdt] Although there is a specialized matrix data type defined in NumPy, it's more standard to work with ordinary NumPy arrays.  See :ref:`this discussion <numpy_matrix_multiplication>`.


.. [#cfn] Suppose that :math:`\|S \| < 1`. Take any nonzero vector :math:`x`, and let :math:`r := \|x\|`. We have :math:`\| Sx \| = r \| S (x/r) \| \leq r \| S \| < r = \| x\|`. Hence every point is pulled towards the origin.
