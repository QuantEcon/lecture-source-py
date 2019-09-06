.. _orth_proj:

.. include:: /_static/includes/header.raw

.. highlight:: python3


*********************************************
Orthogonal Projections and Their Applications
*********************************************

.. index::
    single: Orthogonal Projection

.. contents:: :depth: 2


Overview
===========

Orthogonal projection is a cornerstone of vector space methods, with many diverse applications.

These include, but are not limited to,

* Least squares projection, also known as linear regression

* Conditional expectations for multivariate normal (Gaussian) distributions

* Gram--Schmidt orthogonalization

* QR decomposition

* Orthogonal polynomials

* etc


In this lecture, we focus on

* key ideas

* least squares regression

We'll require the following imports:

.. code-block:: ipython

    import numpy as np
    from scipy.linalg import qr



Further Reading
---------------

For background and foundational concepts, see our lecture :doc:`on linear algebra <linear_algebra>`.

For more proofs and greater theoretical detail, see `A Primer in Econometric Theory <http://www.johnstachurski.net/emet.html>`_.

For a complete set of proofs in a general setting, see, for example, :cite:`Roman2005`.

For an advanced treatment of projection in the context of least squares prediction, see `this book chapter <http://www.tomsargent.com/books/TOMchpt.2.pdf>`_.


Key Definitions
===============


Assume  :math:`x, z \in \mathbb R^n`.

Define :math:`\langle x,  z\rangle = \sum_i x_i z_i`.

Recall :math:`\|x \|^2 = \langle x, x \rangle`.

The **law of cosines** states that :math:`\langle x, z \rangle = \| x \| \| z \| \cos(\theta)` where :math:`\theta` is the angle between the vectors :math:`x` and :math:`z`.

When :math:`\langle x,  z\rangle = 0`, then :math:`\cos(\theta) = 0` and  :math:`x` and :math:`z` are said to be **orthogonal** and we write :math:`x \perp z`.




.. figure:: /_static/lecture_specific/orth_proj/orth_proj_def1.png

For a linear subspace  :math:`S \subset \mathbb R^n`, we call :math:`x \in \mathbb R^n` **orthogonal to** :math:`S` if :math:`x \perp z` for all :math:`z \in S`, and write :math:`x \perp S`.


.. figure:: /_static/lecture_specific/orth_proj/orth_proj_def2.png



The **orthogonal complement** of linear subspace :math:`S \subset \mathbb R^n` is the set :math:`S^{\perp} := \{x \in \mathbb R^n \,:\, x \perp S\}`.


.. figure:: /_static/lecture_specific/orth_proj/orth_proj_def3.png


:math:`S^\perp` is  a linear subspace of :math:`\mathbb R^n`

* To see this, fix :math:`x, y \in S^{\perp}` and :math:`\alpha, \beta \in \mathbb R`.

* Observe that if :math:`z \in S`, then

.. math::

    \langle \alpha x + \beta y, z \rangle
    = \alpha \langle x, z \rangle + \beta \langle y, z \rangle
     = \alpha \times 0  + \beta \times 0 = 0


* Hence :math:`\alpha x + \beta y \in S^{\perp}`, as was to be shown


A set of vectors :math:`\{x_1, \ldots, x_k\} \subset \mathbb R^n` is called an **orthogonal set** if :math:`x_i \perp x_j` whenever :math:`i \not= j`.


If :math:`\{x_1, \ldots, x_k\}` is an orthogonal set, then the **Pythagorean Law** states that

.. math::

    \| x_1 + \cdots + x_k \|^2
    = \| x_1 \|^2 + \cdots + \| x_k \|^2


For example, when  :math:`k=2`, :math:`x_1 \perp x_2` implies

.. math::

    \| x_1 + x_2 \|^2
     = \langle x_1 + x_2, x_1 + x_2 \rangle
     = \langle x_1, x_1 \rangle + 2 \langle  x_2, x_1 \rangle + \langle x_2, x_2 \rangle
     = \| x_1 \|^2 + \| x_2 \|^2


Linear Independence vs Orthogonality
------------------------------------

If :math:`X \subset \mathbb R^n` is an orthogonal set and :math:`0 \notin X`, then :math:`X` is linearly independent.

Proving this is a nice exercise.

While the converse is not true, a kind of partial converse holds, as we'll :ref:`see below <gram_schmidt>`.


The Orthogonal Projection Theorem
=================================

What vector within a linear subspace of :math:`\mathbb R^n`  best approximates a given vector in :math:`\mathbb R^n`?

The next theorem provides answer to this question.

**Theorem** (OPT) Given :math:`y \in \mathbb R^n` and linear subspace :math:`S \subset \mathbb R^n`,
there exists a unique solution to the minimization problem

.. math::

    \hat y := \argmin_{z \in S} \|y - z\|


The minimizer :math:`\hat y` is the unique vector in :math:`\mathbb R^n` that satisfies

* :math:`\hat y \in S`

* :math:`y - \hat y \perp S`


The vector :math:`\hat y` is called the **orthogonal projection** of :math:`y` onto :math:`S`.

The next figure provides some intuition

.. figure:: /_static/lecture_specific/orth_proj/orth_proj_thm1.png





Proof of Sufficiency
--------------------

We'll omit the full proof.

But we will prove sufficiency of the asserted conditions.

To this end, let :math:`y \in \mathbb R^n` and let :math:`S` be a linear subspace of :math:`\mathbb R^n`.

Let :math:`\hat y` be a vector in :math:`\mathbb R^n` such that :math:`\hat y \in S` and :math:`y - \hat y \perp S`.

Let :math:`z` be any other point in :math:`S` and use the fact that :math:`S` is a linear subspace to deduce

.. math::

    \| y - z \|^2
    = \| (y - \hat y) + (\hat y - z) \|^2
    = \| y - \hat y \|^2  + \| \hat y - z  \|^2


Hence :math:`\| y - z \| \geq \| y - \hat y \|`, which completes the proof.



Orthogonal Projection as a Mapping
----------------------------------

For a linear space :math:`Y` and a fixed linear subspace :math:`S`, we have a functional relationship

.. math::

    y \in Y\; \mapsto \text{ its orthogonal projection } \hat y \in S


By the OPT, this is a well-defined mapping  or *operator* from :math:`\mathbb R^n` to :math:`\mathbb R^n`.

In what follows we denote this operator by a matrix :math:`P`

* :math:`P y` represents the projection :math:`\hat y`.

* This is sometimes expressed as :math:`\hat E_S y = P y`, where :math:`\hat E` denotes a **wide-sense expectations operator** and the subscript :math:`S` indicates that we are projecting :math:`y` onto the linear subspace :math:`S`.

The operator :math:`P` is called the **orthogonal projection mapping onto** :math:`S`.


.. figure:: /_static/lecture_specific/orth_proj/orth_proj_thm2.png



It is immediate from the OPT that for any :math:`y \in \mathbb R^n`

#. :math:`P y \in S` and

#. :math:`y - P y \perp S`

From this, we can deduce additional useful properties, such as

#. :math:`\| y \|^2 = \| P y \|^2 + \| y - P y \|^2` and

#. :math:`\| P y \| \leq \| y \|`

For example, to prove 1, observe that :math:`y  = P y  + y - P y` and apply the Pythagorean law.

Orthogonal Complement
^^^^^^^^^^^^^^^^^^^^^

Let :math:`S \subset \mathbb R^n`.

The **orthogonal complement** of :math:`S` is the linear subspace :math:`S^{\perp}` that satisfies
:math:`x_1 \perp x_2` for every :math:`x_1 \in S` and :math:`x_2 \in S^{\perp}`.

Let :math:`Y` be a linear space with linear subspace :math:`S` and its orthogonal complement :math:`S^{\perp}`.

We write

.. math::

    Y = S \oplus S^{\perp}

to indicate that for every :math:`y \in Y` there is unique :math:`x_1 \in S` and a unique :math:`x_2 \in S^{\perp}`
such that :math:`y = x_1 + x_2`.

Moreover, :math:`x_1 = \hat E_S y` and :math:`x_2 = y - \hat E_S y`.

This amounts to another version of the OPT:

**Theorem**.  If :math:`S` is a linear subspace of :math:`\mathbb R^n`, :math:`\hat E_S y = P y` and :math:`\hat E_{S^{\perp}} y = M y`, then

.. math::

     P y \perp M y
     \quad \text{and} \quad
    y = P y + M y
     \quad \text{for all } \, y \in \mathbb R^n


The next figure illustrates

.. figure:: /_static/lecture_specific/orth_proj/orth_proj_thm3.png



Orthonormal Basis
=================


An orthogonal set of vectors :math:`O \subset \mathbb R^n` is called an **orthonormal set** if :math:`\| u \| = 1` for all :math:`u \in O`.

Let :math:`S` be a linear subspace of :math:`\mathbb R^n` and let :math:`O \subset S`.

If :math:`O` is orthonormal and :math:`\mathop{\mathrm{span}} O = S`, then :math:`O` is called an **orthonormal basis** of :math:`S`.

:math:`O` is necessarily a basis of :math:`S` (being independent by orthogonality and the fact that no element is the zero vector).

One example of an orthonormal set is the canonical basis :math:`\{e_1, \ldots, e_n\}`
that forms an orthonormal basis of :math:`\mathbb R^n`, where :math:`e_i` is the :math:`i` th unit vector.

If :math:`\{u_1, \ldots, u_k\}` is an orthonormal basis of linear subspace :math:`S`, then

.. math::

    x = \sum_{i=1}^k \langle x, u_i \rangle u_i
    \quad \text{for all} \quad
    x \in S


To see this, observe that since :math:`x \in \mathop{\mathrm{span}}\{u_1, \ldots, u_k\}`, we can find
scalars :math:`\alpha_1, \ldots, \alpha_k` that verify

.. math::
    :label: pob

    x = \sum_{j=1}^k \alpha_j u_j


Taking the inner product with respect to :math:`u_i` gives

.. math::

    \langle x, u_i \rangle
    = \sum_{j=1}^k \alpha_j \langle u_j, u_i \rangle
    = \alpha_i


Combining this result with :eq:`pob` verifies the claim.




Projection onto an Orthonormal Basis
------------------------------------

When the subspace onto which are projecting is orthonormal, computing the projection simplifies:

**Theorem** If :math:`\{u_1, \ldots, u_k\}` is an orthonormal basis for :math:`S`, then

.. math::
    :label: exp_for_op

    P y = \sum_{i=1}^k \langle y, u_i \rangle u_i,
    \quad
    \forall \; y \in \mathbb R^n


Proof: Fix :math:`y \in \mathbb R^n` and let :math:`P y` be  defined as in :eq:`exp_for_op`.

Clearly, :math:`P y \in S`.

We claim that :math:`y - P y \perp S` also holds.

It sufficies to show that :math:`y - P y \perp` any basis vector :math:`u_i` (why?).

This is true because

.. math::

    \left\langle y - \sum_{i=1}^k \langle y, u_i \rangle u_i, u_j \right\rangle
    = \langle y, u_j \rangle  - \sum_{i=1}^k \langle y, u_i \rangle
    \langle u_i, u_j  \rangle = 0


Projection Using Matrix Algebra
===============================


Let :math:`S` be a linear subspace of :math:`\mathbb R^n` and  let :math:`y \in \mathbb R^n`.

We want to compute the matrix :math:`P` that verifies

.. math::

   \hat E_S y = P y

Evidently  :math:`Py` is a linear function from :math:`y \in \mathbb R^n` to :math:`P y \in \mathbb R^n`.

This reference is useful `<https://en.wikipedia.org/wiki/Linear_map#Matrices>`_.

**Theorem.** Let the columns of :math:`n \times k` matrix :math:`X` form a basis of :math:`S`.  Then

.. math::

    P = X (X'X)^{-1} X'


Proof: Given arbitrary :math:`y \in \mathbb R^n` and :math:`P = X (X'X)^{-1} X'`, our claim is that

#. :math:`P y \in S`, and

#. :math:`y - P y \perp S`

Claim 1 is true because

.. math::

    P y = X (X' X)^{-1} X' y = X a
    \quad \text{when} \quad
    a := (X' X)^{-1} X' y


An expression of the form :math:`X a` is precisely a linear combination of the
columns of :math:`X`, and hence an element of :math:`S`.

Claim 2 is equivalent to the statement

.. math::

    y - X (X' X)^{-1} X' y \, \perp\,  X b
    \quad \text{for all} \quad
    b \in \mathbb R^K


This is true: If :math:`b \in \mathbb R^K`, then

.. math::

    (X b)' [y - X (X' X)^{-1} X'
    y]
    = b' [X' y - X' y]
    = 0


The proof is now complete.


Starting with the Basis
-----------------------

It is common in applications to start with :math:`n \times k` matrix :math:`X`  with linearly independent columns and let

.. math::

    S := \mathop{\mathrm{span}} X := \mathop{\mathrm{span}} \{\col_1 X, \ldots, \col_k X \}


Then the columns of :math:`X` form a basis of :math:`S`.

From the preceding theorem, :math:`P = X (X' X)^{-1} X' y` projects :math:`y` onto :math:`S`.

In this context, :math:`P` is often called the **projection matrix**

* The matrix :math:`M = I - P` satisfies :math:`M y = \hat E_{S^{\perp}} y` and is sometimes called the **annihilator matrix**.




The Orthonormal Case
--------------------

Suppose that :math:`U` is :math:`n \times k` with orthonormal columns.

Let :math:`u_i := \mathop{\mathrm{col}} U_i` for each :math:`i`, let :math:`S := \mathop{\mathrm{span}} U` and let :math:`y \in \mathbb R^n`.

We know that the projection of :math:`y` onto :math:`S` is

.. math::

    P y = U (U' U)^{-1} U' y


Since :math:`U` has orthonormal columns, we have :math:`U' U = I`.

Hence

.. math::

    P y
    = U U' y
    = \sum_{i=1}^k \langle u_i, y \rangle u_i


We have recovered our earlier result about projecting onto the span of an orthonormal
basis.



Application: Overdetermined Systems of Equations
------------------------------------------------

Let :math:`y \in \mathbb R^n` and let :math:`X` is :math:`n \times k` with linearly independent columns.

Given :math:`X` and :math:`y`, we seek :math:`b \in \mathbb R^k` satisfying the system of linear equations :math:`X b = y`.

If :math:`n > k` (more equations than unknowns), then :math:`b` is said to be **overdetermined**.

Intuitively, we may not be able to find a :math:`b` that satisfies all :math:`n` equations.

The best approach here is to

* Accept that an exact solution may not exist.

* Look instead for an approximate solution.

By approximate solution, we mean a :math:`b \in \mathbb R^k` such that :math:`X b` is as close to :math:`y` as possible.

The next theorem shows that the solution is well defined and unique.

The proof uses the OPT.

**Theorem** The unique minimizer of  :math:`\| y - X b \|` over :math:`b \in \mathbb R^K` is

.. math::

    \hat \beta := (X' X)^{-1} X' y


Proof:  Note that

.. math::

    X \hat \beta = X (X' X)^{-1} X' y =
    P y


Since :math:`P y` is the orthogonal projection onto :math:`\mathop{\mathrm{span}}(X)` we have

.. math::

    \| y - P y \|
    \leq \| y - z \| \text{ for any } z \in \mathop{\mathrm{span}}(X)


Because :math:`Xb \in \mathop{\mathrm{span}}(X)`

.. math::

    \| y - X \hat \beta \|
    \leq \| y - X b \| \text{ for any } b \in \mathbb R^K


This is what we aimed to show.


Least Squares Regression
========================

Let's apply the theory of orthogonal projection to least squares regression.

This approach provides insights about  many geometric  properties of linear regression.

We treat only some examples.


Squared Risk Measures
---------------------

Given pairs :math:`(x, y) \in \mathbb R^K \times \mathbb R`, consider choosing :math:`f \colon \mathbb R^K \to \mathbb R` to minimize
the **risk**

.. math::

    R(f) := \mathbb{E}\, [(y - f(x))^2]


If probabilities and hence :math:`\mathbb{E}\,` are unknown, we cannot solve this problem directly.

However, if a sample is available, we can estimate the risk with the **empirical risk**:

.. math::

    \min_{f \in \mathcal{F}} \frac{1}{N} \sum_{n=1}^N (y_n - f(x_n))^2


Minimizing this expression is called **empirical risk minimization**.

The set :math:`\mathcal{F}` is sometimes called the hypothesis space.

The theory of statistical learning tells us that to prevent overfitting we should take the set :math:`\mathcal{F}` to be relatively simple.

If we let :math:`\mathcal{F}` be the class of linear functions :math:`1/N`, the problem is

.. math::

    \min_{b \in \mathbb R^K} \;
    \sum_{n=1}^N (y_n - b' x_n)^2


This is the sample **linear least squares problem**.



Solution
--------

Define the matrices

.. math::

    y :=
    \left(
    \begin{array}{c}
        y_1 \\
        y_2 \\
        \vdots \\
        y_N
    \end{array}
    \right),
    \quad
    x_n :=
    \left(
    \begin{array}{c}
        x_{n1} \\
        x_{n2} \\
        \vdots \\
        x_{nK}
    \end{array}
    \right)
    = \text{ :math:`n`-th obs on all regressors}


and

.. math::

    X :=
    \left(
    \begin{array}{c}
        x_1'  \\
        x_2'  \\
        \vdots     \\
        x_N'
    \end{array}
    \right)
    :=:
    \left(
    \begin{array}{cccc}
        x_{11} & x_{12} & \cdots & x_{1K} \\
        x_{21} & x_{22} & \cdots & x_{2K} \\
        \vdots & \vdots &  & \vdots \\
        x_{N1} & x_{N2} & \cdots & x_{NK}
    \end{array}
    \right)


We assume throughout that :math:`N > K` and :math:`X` is full column rank.

If you work through the algebra, you will be able to verify that :math:`\| y - X b \|^2 = \sum_{n=1}^N (y_n - b' x_n)^2`.

Since monotone transforms don't affect minimizers, we have

.. math::

    \argmin_{b \in \mathbb R^K} \sum_{n=1}^N (y_n - b' x_n)^2
    = \argmin_{b \in \mathbb R^K} \| y - X b \|


By our results about overdetermined linear systems of equations, the solution is

.. math::

    \hat \beta := (X' X)^{-1} X' y


Let :math:`P` and :math:`M` be the projection and annihilator associated with :math:`X`:

.. math::

    P := X (X' X)^{-1} X'
    \quad \text{and} \quad
    M := I - P


The **vector of fitted values** is

.. math::

    \hat y := X \hat \beta = P y


The **vector of residuals** is

.. math::

    \hat u :=  y - \hat y = y - P y = M y


Here are some more standard definitions:

* The **total sum of squares** is :math:`:=  \| y \|^2`.

* The **sum of squared residuals** is :math:`:= \| \hat u \|^2`.

* The **explained sum of squares** is :math:`:= \| \hat y \|^2`.

 TSS = ESS + SSR

We can prove this easily using the OPT.

From the OPT we have :math:`y =  \hat y + \hat u` and :math:`\hat u \perp \hat y`.

Applying the Pythagorean law completes the proof.




Orthogonalization and Decomposition
===================================

Let's return to the connection between linear independence and orthogonality touched on above.

A result of much interest is a famous algorithm for constructing orthonormal sets from linearly independent sets.

The next section gives details.




.. _gram_schmidt:

Gram-Schmidt Orthogonalization
------------------------------

**Theorem** For each linearly independent set :math:`\{x_1, \ldots, x_k\} \subset \mathbb R^n`, there exists an
orthonormal set :math:`\{u_1, \ldots, u_k\}` with

.. math::

    \mathop{\mathrm{span}} \{x_1, \ldots, x_i\} =
    \mathop{\mathrm{span}} \{u_1, \ldots, u_i\}
    \quad \text{for} \quad
    i = 1, \ldots, k


The **Gram-Schmidt orthogonalization** procedure constructs an orthogonal set :math:`\{ u_1, u_2, \ldots, u_n\}`.

One description of this procedure is as follows:

* For :math:`i = 1, \ldots, k`, form :math:`S_i := \mathop{\mathrm{span}}\{x_1, \ldots, x_i\}` and :math:`S_i^{\perp}`

* Set :math:`v_1 = x_1`

* For :math:`i \geq 2` set :math:`v_i := \hat E_{S_{i-1}^{\perp}} x_i` and :math:`u_i := v_i / \| v_i \|`

The sequence :math:`u_1, \ldots, u_k` has the stated properties.

A Gram-Schmidt orthogonalization construction is a key idea behind the Kalman filter described in :doc:`A First Look at the Kalman filter<kalman>`.

In some exercises below, you are asked to implement this algorithm and test it using projection.



QR Decomposition
----------------

The following result uses the preceding algorithm to produce a useful decomposition.

**Theorem** If :math:`X` is :math:`n \times k` with linearly independent columns, then there exists a factorization :math:`X = Q R` where

* :math:`R` is :math:`k \times k`, upper triangular, and nonsingular

* :math:`Q` is :math:`n \times k` with orthonormal columns

Proof sketch: Let

* :math:`x_j := \col_j (X)`

* :math:`\{u_1, \ldots, u_k\}` be orthonormal with the same span as :math:`\{x_1, \ldots, x_k\}` (to be constructed using Gram--Schmidt)

* :math:`Q` be formed from cols :math:`u_i`

Since :math:`x_j \in \mathop{\mathrm{span}}\{u_1, \ldots, u_j\}`, we have

.. math::

    x_j = \sum_{i=1}^j \langle u_i, x_j  \rangle u_i
    \quad \text{for } j = 1, \ldots, k


Some rearranging gives :math:`X = Q R`.



Linear Regression via QR Decomposition
--------------------------------------

For matrices :math:`X` and :math:`y` that overdetermine :math:`beta` in the linear
equation system :math:`y = X \beta`, we found  the least squares approximator :math:`\hat \beta = (X' X)^{-1} X' y`.

Using the QR decomposition :math:`X = Q R` gives

.. math::

    \begin{aligned}
        \hat \beta
        & = (R'Q' Q R)^{-1} R' Q' y \\
        & = (R' R)^{-1} R' Q' y \\
        & = R^{-1} (R')^{-1} R' Q' y
            = R^{-1} Q' y
    \end{aligned}


Numerical routines would in this case use the alternative form :math:`R \hat \beta = Q' y` and back substitution.





Exercises
=========

Exercise 1
----------

Show that, for any linear subspace :math:`S \subset \mathbb R^n`,  :math:`S \cap S^{\perp} = \{0\}`.

Exercise 2
----------

Let :math:`P = X (X' X)^{-1} X'` and let :math:`M = I - P`.  Show that
:math:`P` and :math:`M` are both idempotent and symmetric.  Can you give any
intuition as to why they should be idempotent?

Exercise 3
----------

Using Gram-Schmidt orthogonalization, produce a linear projection of :math:`y` onto the column space of :math:`X` and verify this using the projection matrix :math:`P := X (X' X)^{-1} X'` and also using QR decomposition, where:

.. math::

    y :=
    \left(
    \begin{array}{c}
        1 \\
        3 \\
        -3
    \end{array}
    \right),
    \quad

and

.. math::

    X :=
    \left(
    \begin{array}{cc}
        1 &  0  \\
        0 & -6 \\
        2 &  2
    \end{array}
    \right)


Solutions
=========



Exercise 1
----------

If :math:`x \in S` and :math:`x \in S^\perp`, then we have in particular
that :math:`\langle x, x \rangle = 0`, ut then :math:`x = 0`.

Exercise 2
----------

Symmetry and idempotence of :math:`M` and :math:`P` can be established
using standard rules for matrix algebra. The intuition behind
idempotence of :math:`M` and :math:`P` is that both are orthogonal
projections. After a point is projected into a given subspace, applying
the projection again makes no difference. (A point inside the subspace
is not shifted by orthogonal projection onto that space because it is
already the closest point in the subspace to itself.).

Exercise 3
----------

Here's a function that computes the orthonormal vectors using the GS
algorithm given in the lecture

.. code-block:: python3

    def gram_schmidt(X):
        """
        Implements Gram-Schmidt orthogonalization.

        Parameters
        ----------
        X : an n x k array with linearly independent columns

        Returns
        -------
        U : an n x k array with orthonormal columns

        """

        # Set up
        n, k = X.shape
        U = np.empty((n, k))
        I = np.eye(n)

        # The first col of U is just the normalized first col of X
        v1 = X[:,0]
        U[:, 0] = v1 / np.sqrt(np.sum(v1 * v1))

        for i in range(1, k):
            # Set up
            b = X[:, i]       # The vector we're going to project
            Z = X[:, 0:i]     # First i-1 columns of X

            # Project onto the orthogonal complement of the col span of Z
            M = I - Z @ np.linalg.inv(Z.T @ Z) @ Z.T
            u = M @ b

            # Normalize
            U[:, i] = u / np.sqrt(np.sum(u * u))

        return U

Here are the arrays we'll work with

.. code-block:: python3

    y = [1, 3, -3]

    X = [[1,  0],
         [0, -6],
         [2,  2]]

    X, y = [np.asarray(z) for z in (X, y)]

First, let's try projection of :math:`y` onto the column space of
:math:`X` using the ordinary matrix expression:

.. code-block:: python3

    Py1 = X @ np.linalg.inv(X.T @ X) @ X.T @ y
    Py1



Now let's do the same using an orthonormal basis created from our
``gram_schmidt`` function

.. code-block:: python3

    U = gram_schmidt(X)
    U



.. code-block:: python3

    Py2 = U @ U.T @ y
    Py2



This is the same answer. So far so good. Finally, let's try the same
thing but with the basis obtained via QR decomposition:

.. code-block:: python3

    Q, R = qr(X, mode='economic')
    Q



.. code-block:: python3

    Py3 = Q @ Q.T @ y
    Py3



Again, we obtain the same answer.
