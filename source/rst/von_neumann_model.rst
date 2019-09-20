.. _von_neumann_model:

.. include:: /_static/includes/header.raw

.. index::
    single: python

***********************************************
Von Neumann Growth Model (and a Generalization)
***********************************************

.. contents:: :depth: 2

**Co-author:** Balint Szoke

This notebook uses the class ``Neumann`` to calculate key objects of a
linear growth model of John von Neumann (1937) :cite:`von1937uber` that was generalized by
Kemeny, Moregenstern and Thompson (1956) :cite:`kemeny1956generalization`.

Objects of interest are the maximal expansion rate (:math:`\alpha`), the
interest factor (:math:`β`), and the optimal intensities (:math:`x`) and
prices (:math:`p`).

In addition to watching how the towering mind of John von Neumann
formulated an equilibrium model of price and quantity vectors in
balanced growth, this notebook shows how fruitfully to employ the
following important tools:

-  a zero-sum two-player game

-  linear programming

-  the Perron-Frobenius theorem

We'll begin with some imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import solve
    from scipy.optimize import fsolve, linprog
    from textwrap import dedent
    %matplotlib inline

    np.set_printoptions(precision=2)

The code below provides the ``Neumann`` class

.. code-block:: python3
  :class: collapse


  class Neumann(object):

      """
      This class describes the Generalized von Neumann growth model as it was
      discussed in Kemeny et al. (1956, ECTA) :cite:`kemeny1956generalization`
      and Gale (1960, Chapter 9.5) :cite:`gale1989theory`:

      Let:
      n ... number of goods
      m ... number of activities
      A ... input matrix is m-by-n
          a_{i,j} - amount of good j consumed by activity i
      B ... output matrix is m-by-n
          b_{i,j} - amount of good j produced by activity i

      x ... intensity vector (m-vector) with non-negative entries
          x'B - the vector of goods produced
          x'A - the vector of goods consumed
      p ... price vector (n-vector) with non-negative entries
          Bp - the revenue vector for every activity
          Ap - the cost of each activity

      Both A and B have non-negative entries. Moreover, we assume that
      (1) Assumption I (every good which is consumed is also produced):
          for all j, b_{.,j} > 0, i.e. at least one entry is strictly positive
      (2) Assumption II (no free lunch):
          for all i, a_{i,.} > 0, i.e. at least one entry is strictly positive

      Parameters
      ----------
      A : array_like or scalar(float)
          Part of the state transition equation.  It should be `n x n`
      B : array_like or scalar(float)
          Part of the state transition equation.  It should be `n x k`
      """

      def __init__(self, A, B):

          self.A, self.B = list(map(self.convert, (A, B)))
          self.m, self.n = self.A.shape

          # Check if (A, B) satisfy the basic assumptions
          assert self.A.shape == self.B.shape, 'The input and output matrices \
                must have the same dimensions!'
          assert (self.A >= 0).all() and (self.B >= 0).all(), 'The input and \
                output matrices must have only non-negative entries!'

          # (1) Check whether Assumption I is satisfied:
          if (np.sum(B, 0) <= 0).any():
              self.AI = False
          else:
              self.AI = True

          # (2) Check whether Assumption II is satisfied:
          if (np.sum(A, 1) <= 0).any():
              self.AII = False
          else:
              self.AII = True

      def __repr__(self):
          return self.__str__()

      def __str__(self):

          me = """
          Generalized von Neumann expanding model:
            - number of goods          : {n}
            - number of activities     : {m}

          Assumptions:
            - AI:  every column of B has a positive entry    : {AI}
            - AII: every row of A has a positive entry       : {AII}

          """
          # Irreducible                                       : {irr}
          return dedent(me.format(n=self.n, m=self.m,
                                  AI=self.AI, AII=self.AII))

      def convert(self, x):
          """
          Convert array_like objects (lists of lists, floats, etc.) into
          well-formed 2D NumPy arrays
          """
          return np.atleast_2d(np.asarray(x))


      def bounds(self):
          """
          Calculate the trivial upper and lower bounds for alpha (expansion rate)
          and beta (interest factor). See the proof of Theorem 9.8 in Gale (1960)
          :cite:`gale1989theory`
          """

          n, m = self.n, self.m
          A, B = self.A, self.B

          f = lambda α: ((B - α * A) @ np.ones((n, 1))).max()
          g = lambda β: (np.ones((1, m)) @ (B - β * A)).min()

          UB = np.asscalar(fsolve(f, 1))  # Upper bound for α, β
          LB = np.asscalar(fsolve(g, 2))  # Lower bound for α, β

          return LB, UB


      def zerosum(self, γ, dual=False):
          """
          Given gamma, calculate the value and optimal strategies of a
          two-player zero-sum game given by the matrix

                  M(gamma) = B - gamma * A

          Row player maximizing, column player minimizing

          Zero-sum game as an LP (primal --> α)

              max (0', 1) @ (x', v)
              subject to
              [-M', ones(n, 1)] @ (x', v)' <= 0
              (x', v) @ (ones(m, 1), 0) = 1
              (x', v) >= (0', -inf)

          Zero-sum game as an LP (dual --> beta)

              min (0', 1) @ (p', u)
              subject to
              [M, -ones(m, 1)] @ (p', u)' <= 0
              (p', u) @ (ones(n, 1), 0) = 1
              (p', u) >= (0', -inf)

          Outputs:
          --------
          value: scalar
              value of the zero-sum game

          strategy: vector
              if dual = False, it is the intensity vector,
              if dual = True, it is the price vector
          """

          A, B, n, m = self.A, self.B, self.n, self.m
          M = B - γ * A

          if dual == False:
              # Solve the primal LP (for details see the description)
              # (1) Define the problem for v as a maximization (linprog minimizes)
              c = np.hstack([np.zeros(m), -1])

              # (2) Add constraints :
              # ... non-negativity constraints
              bounds = tuple(m * [(0, None)] + [(None, None)])
              # ... inequality constraints
              A_iq = np.hstack([-M.T, np.ones((n, 1))])
              b_iq = np.zeros((n, 1))
              # ... normalization
              A_eq = np.hstack([np.ones(m), 0]).reshape(1, m + 1)
              b_eq = 1

              res = linprog(c, A_ub=A_iq, b_ub=b_iq, A_eq=A_eq, b_eq=b_eq,
                            bounds=bounds, options=dict(bland=True, tol=1e-7))

          else:
              # Solve the dual LP (for details see the description)
              # (1) Define the problem for v as a maximization (linprog minimizes)
              c = np.hstack([np.zeros(n), 1])

              # (2) Add constraints :
              # ... non-negativity constraints
              bounds = tuple(n * [(0, None)] + [(None, None)])
              # ... inequality constraints
              A_iq = np.hstack([M, -np.ones((m, 1))])
              b_iq = np.zeros((m, 1))
              # ... normalization
              A_eq = np.hstack([np.ones(n), 0]).reshape(1, n + 1)
              b_eq = 1

              res = linprog(c, A_ub=A_iq, b_ub=b_iq, A_eq=A_eq, b_eq=b_eq,
                            bounds=bounds, options=dict(bland=True, tol=1e-7))

          if res.status != 0:
              print(res.message)

          # Pull out the required quantities
          value = res.x[-1]
          strategy = res.x[:-1]

          return value, strategy


      def expansion(self, tol=1e-8, maxit=1000):
          """
          The algorithm used here is described in Hamburger-Thompson-Weil
          (1967, ECTA). It is based on a simple bisection argument and utilizes
          the idea that for a given γ (= α or β), the matrix "M = B - γ * A"
          defines a two-player zero-sum game, where the optimal strategies are
          the (normalized) intensity and price vector.

          Outputs:
          --------
          alpha: scalar
              optimal expansion rate
          """

          LB, UB = self.bounds()

          for iter in range(maxit):

              γ = (LB + UB) / 2
              ZS = self.zerosum(γ=γ)
              V = ZS[0]     # value of the game with γ

              if V >= 0:
                  LB = γ
              else:
                  UB = γ

              if abs(UB - LB) < tol:
                  γ = (UB + LB) / 2
                  x = self.zerosum(γ=γ)[1]
                  p = self.zerosum(γ=γ, dual=True)[1]
                  break

          return γ, x, p

      def interest(self, tol=1e-8, maxit=1000):
          """
          The algorithm used here is described in Hamburger-Thompson-Weil
          (1967, ECTA). It is based on a simple bisection argument and utilizes
          the idea that for a given gamma (= alpha or beta),
          the matrix "M = B - γ * A" defines a two-player zero-sum game,
          where the optimal strategies are the (normalized) intensity and price
          vector

          Outputs:
          --------
          beta: scalar
              optimal interest rate
          """

          LB, UB = self.bounds()

          for iter in range(maxit):
              γ = (LB + UB) / 2
              ZS = self.zerosum(γ=γ, dual=True)
              V = ZS[0]

              if V > 0:
                  LB = γ
              else:
                  UB = γ

              if abs(UB - LB) < tol:
                  γ = (UB + LB) / 2
                  p = self.zerosum(γ=γ, dual=True)[1]
                  x = self.zerosum(γ=γ)[1]
                  break

          return γ, x, p

Notation
========

We use the following notation.

:math:`\mathbf{0}` denotes
a vector of zeros. We call an :math:`n`-vector - positive or
:math:`x\gg \mathbf{0}` if :math:`x_i>0` for all :math:`i=1,2,\dots,n`
- non-negative or :math:`x\geq \mathbf{0}` if :math:`x_i\geq 0` for
all :math:`i=1,2,\dots,n` - semi-positive or :math:`x > \mathbf{0}` if
:math:`x\geq \mathbf{0}` and :math:`x\neq \mathbf{0}`.

For two conformable vectors :math:`x` and :math:`y`, :math:`x\gg y`,
:math:`x\geq y` and :math:`x> y` mean :math:`x-y\gg \mathbf{0}`,
:math:`x-y \geq \mathbf{0}`, and :math:`x-y > \mathbf{0}`.

By default, all vectors are column vectors, :math:`x^{T}` denotes the
transpose of :math:`x` (i.e. a row vector).

Let :math:`\iota_n` denote a
column vector composed of :math:`n` ones, i.e.
:math:`\iota_n = (1,1,\dots,1)^T`.

Let :math:`e^i` denote the vector (of
arbitrary size) containing zeros except for the :math:`i` th position
where it is one.

We denote matrices by capital letters. For an arbitrary matrix
:math:`A`, :math:`a_{i,j}` represents the entry in its :math:`i` th
row and :math:`j` th column.

:math:`a_{\cdot j}` and :math:`a_{i\cdot}`
denote the :math:`j` th column and :math:`i` th row of :math:`A`,
respectively.

Model Ingredients and Assumptions
=================================

A pair :math:`(A,B)` of :math:`m\times n` non-negative matrices defines
an economy.

-  :math:`m` is the number of *activities* (or sectors)

-  :math:`n` is the number of *goods* (produced and/or used in the
   economy)

-  :math:`A` is called the *input matrix*; :math:`a_{i,j}` denotes the
   amount of good :math:`j` consumed by activity :math:`i`

-  :math:`B` is called the *output matrix*; :math:`b_{i,j}` represents
   the amount of good :math:`j` produced by activity :math:`i`

Two key assumptions restrict economy :math:`(A,B)`:

- **Assumption I:** (every good which is consumed is also produced)

.. math:: b_{.,j} > \mathbf{0}\hspace{5mm}\forall j=1,2,\dots,n

- **Assumption II:** (no free lunch)

.. math:: a_{i,.} > \mathbf{0}\hspace{5mm}\forall i=1,2,\dots,m

A semi-positive :math:`m`-vector:math:`x` denotes the levels at which
activities are operated (*intensity vector*).

Therefore,

-  vector :math:`x^TA` gives the total amount of *goods used in
   production*

-  vector :math:`x^TB` gives *total outputs*

An economy :math:`(A,B)` is said to be *productive*, if there exists a
non-negative intensity vector :math:`x \geq 0` such
that :math:`x^T B > x^TA`.

The semi-positive :math:`n`-vector :math:`p` contains prices assigned to
the :math:`n` goods.

The :math:`p` vector implies *cost* and *revenue* vectors

-  the vector :math:`Ap` tells *costs* of the vector of activities

-  the vector :math:`Bp` tells *revenues* from the vector of activities

A property of an input-output pair :math:`(A,B)` called *irreducibility*
(or indecomposability) determines whether an economy can be decomposed
into multiple ‘’sub-economies’’.

**Definition:** Given an economy :math:`(A,B)`, the set of goods
:math:`S\subset \{1,2,\dots,n\}` is called an *independent subset* if
it is possible to produce every good in :math:`S` without consuming any
good outside :math:`S`. Formally, the set :math:`S` is independent if
:math:`\exists T\subset \{1,2,\dots,m\}` (subset of activities) such
that :math:`a_{i,j}=0`, :math:`\forall i\in T` and :math:`j\in S^c` and
for all :math:`j\in S`, :math:`\exists i\in T`, s.t. :math:`b_{i,j}>0`.
The economy is **irreducible** if there are no proper independent
subsets.

We study two examples, both coming from Chapter 9.6 of Gale (1960) :cite:`gale1989theory`

.. code-block:: python3

    # (1) Irreducible (A, B) example: α_0 = β_0
    A1 = np.array([[0, 1, 0, 0],
                   [1, 0, 0, 1],
                   [0, 0, 1, 0]])

    B1 = np.array([[1, 0, 0, 0],
                   [0, 0, 2, 0],
                   [0, 1, 0, 1]])

    # (2) Reducible (A, B) example: β_0 < α_0
    A2 = np.array([[0, 1, 0, 0, 0, 0],
                   [1, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0, 1],
                   [0, 0, 0, 0, 1, 0]])

    B2 = np.array([[1, 0, 0, 1, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 2, 0],
                   [0, 0, 0, 1, 0, 1]])

The following code sets up our first Neumann economy or ``Neumann``
instance

.. code-block:: python3

    n1 = Neumann(A1, B1)
    n1


.. code-block:: python3

    n2 = Neumann(A2, B2)
    n2

Dynamic Interpretation
======================

Attach a time index :math:`t` to the preceding objects, regard an economy
as a dynamic system, and study sequences

.. math:: \{(A_t,B_t)\}_{t\geq 0}, \hspace{1cm}\{x_t\}_{t\geq 0},\hspace{1cm} \{p_t\}_{t\geq 0}

An interesting special case holds the technology process constant and
investigates the dynamics of quantities and prices only.

Accordingly, in the rest of this notebook, we assume that
:math:`(A_t,B_t)=(A,B)` for all :math:`t\geq 0`.

A crucial element of the dynamic interpretation involves the timing of
production.

We assume that production (consumption of inputs) takes place in period
:math:`t`, while the associated output materializes in period
:math:`t+1`, i.e. consumption of :math:`x_{t}^TA` in period :math:`t`
results in :math:`x^T_{t}B` amounts of output in period :math:`t+1`.

These timing conventions imply the following feasibility condition:

.. math::

  \begin{aligned}
  x^T_{t}B \geq x^T_{t+1} A \hspace{1cm}\forall t\geq 1
  \end{aligned}

which asserts that no more goods can be used today than were produced
yesterday.

Accordingly, :math:`Ap_t` tells the costs of production in period
:math:`t` and :math:`Bp_t` tells revenues in period :math:`t+1`.

Balanced Growth
---------------

We follow John von Neumann in studying “balanced growth”.

Let :math:`./` denote an elementwise division of one vector by another and let
:math:`\alpha >0` be a scalar.

Then *balanced growth* is a situation in which

.. math:: x_{t+1}./x_t = \alpha , \quad \forall t \geq 0

With balanced growth, the law of motion of :math:`x` is evidently :math:`x_{t+1}=\alpha x_t`
and so we can rewrite the feasibility constraint as

.. math:: x^T_{t}B \geq \alpha x^T_t A \hspace{1cm}\forall t

In the same spirit, define :math:`\beta\in\mathbb{R}` as the **interest
factor** per unit of time.

We assume that it is always possible to earn a gross return equal to the
constant interest factor :math:`\beta` by investing “outside the model”.

Under this assumption about outside investment opportunities, a
no-arbitrage condition gives rise to the following (no profit)
restriction on the price sequence:

.. math:: \beta Ap_{t} \geq B p_{t} \hspace{1cm}\forall t

This says that production cannot yield a return greater than that
offered by the investment opportunity (note that we compare values in
period :math:`t+1`).

The balanced growth assumption allows us to drop time subscripts and
conduct an analysis purely in terms of a time-invariant growth rate
:math:`\alpha` and interest factor :math:`\beta`.


Duality
=======

The following two problems are connected by a remarkable dual
relationship between the technological and valuation characteristics of
the economy:

**Definition:** The *technological expansion problem* (TEP) for the economy
:math:`(A,B)` is to find a semi-positive :math:`m`-vector :math:`x>0`
and a number :math:`\alpha\in\mathbb{R}`, s.t.

.. math::
  \begin{aligned}
    &\max_{\alpha} \hspace{2mm} \alpha\\
    &\text{s.t. }\hspace{2mm}x^T B \geq \alpha x^T A
    \end{aligned}

Theorem 9.3 of David Gale’s book :cite:`gale1989theory` assets that if Assumptions I and II are
both satisfied, then a maximum value of :math:`\alpha` exists and it is
positive.

It is called the *technological expansion rate* and is denoted
by :math:`\alpha_0`. The associated intensity vector :math:`x_0` is the
*optimal intensity vector*.

**Definition:** The *economical expansion problem* (EEP) for
:math:`(A,B)` is to find a semi-positive :math:`n`-vector :math:`p>0`
and a number :math:`\beta\in\mathbb{R}`, such that

.. math::
  \begin{aligned}
    &\min_{\beta} \hspace{2mm} \beta\\
    &\text{s.t. }\hspace{2mm}Bp \leq \beta Ap
    \end{aligned}

Assumptions I and II imply existence of a minimum value
:math:`\beta_0>0` called the *economic expansion rate*.

The corresponding price vector :math:`p_0` is the *optimal price vector*.

Evidently, the criterion functions in *technological expansion* problem
and the *economical expansion problem* are both linearly homogeneous, so
the optimality of :math:`x_0` and :math:`p_0` are defined only up to a
positive scale factor.

For simplicity (and to emphasize a close connection to zero-sum games),
in the following, we normalize both vectors
:math:`x_0` and :math:`p_0` to have unit length.

A standard duality argument (see Lemma 9.4. in (Gale, 1960) :cite:`gale1989theory`) implies
that under Assumptions I and II, :math:`\beta_0\leq \alpha_0`.

But in the other direction, that is :math:`\beta_0\geq \alpha_0`,
Assumptions I and II are not sufficient.

Nevertheless, von Neumann (1937) :cite:`von1937uber` proved the following remarkable
“duality-type” result connecting TEP and EEP.

**Theorem 1 (von Neumann):** If the economy :math:`(A,B)` satisfies
Assumptions I and II, then there exists a set
:math:`\left(\gamma^{*}, x_0, p_0\right)`, where
:math:`\gamma^{*}\in[\beta_0, \alpha_0]\subset\mathbb{R}`, :math:`x_0>0`
is an :math:`m`-vector, :math:`p_0>0` is an :math:`n`-vector and the
following holds true

.. math::

  \begin{aligned}
  x_0^T B &\geq \gamma^{* } x_0^T A \\
  Bp_0 &\leq \gamma^{* } Ap_0 \\
  x_0^T\left(B-\gamma^{* } A\right)p_0 &= 0
  \end{aligned}

..

   *Proof (Sketch):* Assumption I and II imply that there exist :math:`(\alpha_0,
   x_0)` and :math:`(\beta_0, p_0)` solving the TEP and EEP, respectively. If
   :math:`\gamma^*>\alpha_0`, then by definition of :math:`\alpha_0`, there cannot
   exist a semi-positive :math:`x` that satisfies :math:`x^T B \geq \gamma^{* }
   x^T A`.  Similarly, if :math:`\gamma^*<\beta_0`, there is no semi-positive
   :math:`p` so that :math:`Bp \leq \gamma^{* } Ap`. Let :math:`\gamma^{*
   }\in[\beta_0, \alpha_0]`, then :math:`x_0^T B \geq \alpha_0 x_0^T A \geq
   \gamma^{* } x_0^T A`.  Moreover, :math:`Bp_0\leq \beta_0 A p_0\leq \gamma^* A
   p_0`. These two inequalities imply :math:`x_0\left(B - \gamma^{* } A\right)p_0
   = 0`.

Here the constant :math:`\gamma^{*}` is both expansion and interest
factor (not necessarily optimal).

We have already encountered and
discussed the first two inequalities that represent feasibility and
no-profit conditions.

Moreover, the equality compactly captures the
requirements that if any good grows at a rate larger than
:math:`\gamma^{*}` (i.e., if it is *oversupplied*), then its price
must be zero; and that if any activity provides negative profit, it must
be unused.

Therefore, these expressions encode all equilibrium conditions
and Theorem I essentially states that under Assumptions I and II there
always exists an equilibrium :math:`\left(\gamma^{*}, x_0, p_0\right)`
with balanced growth.

Note that Theorem I is silent about uniqueness of the equilibrium. In
fact, it does not rule out (trivial) cases with :math:`x_0^TBp_0 = 0` so
that nothing of value is produced.

To exclude such uninteresting cases,
Kemeny, Morgenstern and Thomspson (1956) add an extra requirement

.. math:: x^T_0 B p_0 > 0

and call the resulting equilibria *economic solutions*.

They show that
this extra condition does not affect the existence result, while it
significantly reduces the number of (relevant) solutions.


Interpretation as a Game Theoretic Problem (Two-player Zero-sum Game)
=====================================================================

To compute the equilibrium :math:`(\gamma^{*}, x_0, p_0)`, we follow the
algorithm proposed by Hamburger, Thompson and Weil (1967), building on
the key insight that the equilibrium (with balanced growth) can be
considered as a solution of a particular two-player zero-sum game.
First, we introduce some notations.

Consider the :math:`m\times n` matrix :math:`C` as a payoff matrix,
with the entries representing payoffs from the **minimizing** column
player to the **maximizing** row player and assume that the players can
use mixed strategies: - row player chooses the :math:`m`-vector
:math:`x > \mathbf{0}`, s.t. :math:`\iota_m^T x = 1` - column player
chooses the :math:`n`-vector :math:`p > \mathbf{0}`,
s.t. :math:`\iota_n^T p = 1`.

**Definition:** The :math:`m\times n` matrix game :math:`C` has the
*solution* :math:`(x^*, p^*, V(C))` in mixed strategies, if

.. math::
  \begin{aligned}
  (x^* )^T C e^j \geq V(C)\quad \forall j\in\{1, \dots, n\}\quad \quad
  \text{and}\quad\quad (e^i)^T C p^* \leq V(C)\quad \forall i\in\{1, \dots, m\}
  \end{aligned}

The number :math:`V(C)` is called the *value* of the game.

From the above definition, it is clear that the value :math:`V(C)` has
two alternative interpretations:

* by playing the appropriate mixed
  stategy, the maximizing player can assure himself at least :math:`V(C)`
  (no matter what the column player chooses)

* by playing the appropriate
  mixed stategy, the minimizing player can make sure that the maximizing
  player will not get more than :math:`V(C)` (irrespective of what is the
  maximizing player’s choice)

From the famous theorem of Nash (1951), it follows that there always
exists a mixed strategy Nash equilibrium for any *finite* two-player
zero-sum game.

Moreover, von Neumann’s Minmax Theorem (1928) :cite:`neumann1928theorie` implies that

.. math::
   V(C) = \max_x \min_p \hspace{2mm} x^T C p = \min_p \max_x \hspace{2mm} x^T C p = (x^*)^T C p^*



Connection with Linear Programming (LP)
---------------------------------------

Finding Nash equilibria of a finite two-player zero-sum game can be
formulated as a linear programming problem.

To see this, we introduce
the following notation - For a fixed :math:`x`, let :math:`v` be the
value of the minimization problem:
:math:`v \equiv \min_p x^T C p = \min_j x^T C e^j` - For a fixed
:math:`p`, let :math:`u` be the value of the maximization problem:
:math:`u \equiv \max_x x^T C p = \max_i (e^i)^T C p`.

Then the *max-min problem* (the game from the maximizing player’s point
of view) can be written as the *primal* LP

.. math::
  \begin{aligned}
  V(C) = & \max \hspace{2mm} v \\
  \text{s.t. } \hspace{2mm} v \iota_n^T &\leq x^T C  \\
  x &\geq \mathbf{0} \\
  \iota_n^T x & = 1
  \end{aligned}

while the *min-max problem* (the game from the minimizing player’s point
of view) is the *dual* LP

.. math::
  \begin{aligned}
  V(C) = &\min \hspace{2mm} u \\
  \text{s.t. } \hspace{2mm}u \iota_m &\geq Cp  \\
  p &\geq \mathbf{0} \\
  \iota_m^T p & = 1
  \end{aligned}


Hamburger, Thompson and Weil (1967) :cite:`hamburger1967computation` view the input-output pair of the
economy as payoff matrices of two-player zero-sum games. Using this
interpretation, they restate Assumption I and II as follows

.. math::

  V(-A) < 0\quad\quad \text{and}\quad\quad V(B)>0

..

   *Proof (Sketch)*: \* :math:`\Rightarrow` :math:`V(B)>0` implies
   :math:`x_0^T B \gg \mathbf{0}`, where :math:`x_0` is a maximizing
   vector. Since :math:`B` is non-negative, this requires that each
   column of :math:`B` has at least one positive entry, which is
   Assumption I. \* :math:`\Leftarrow` From Assumption I and the fact
   that :math:`p>\mathbf{0}`, it follows that :math:`Bp > \mathbf{0}`.
   This implies that the maximizing player can always choose :math:`x`
   so that :math:`x^TBp>0`, that is it must be the case
   that :math:`V(B)>0`.

In order to (re)state Theorem I in terms of a particular two-player
zero-sum game, we define the matrix for :math:`\gamma\in\mathbb{R}`

.. math:: M(\gamma) \equiv B - \gamma A

For fixed :math:`\gamma`, treating :math:`M(\gamma)` as a matrix game,
we can calculate the solution of the game

-  If :math:`\gamma > \alpha_0`, then for all :math:`x>0`, there
   :math:`\exists j\in\{1, \dots, n\}`, s.t.
   :math:`[x^T M(\gamma)]_j < 0` implying
   that :math:`V(M(\gamma)) < 0`.
-  If :math:`\gamma < \beta_0`, then for all :math:`p>0`, there
   :math:`\exists i\in\{1, \dots, m\}`, s.t.
   :math:`[M(\gamma)p]_i > 0` implying that :math:`V(M(\gamma)) > 0`.
-  If :math:`\gamma \in \{\beta_0, \alpha_0\}`, then (by Theorem I) the
   optimal intensity and price vectors :math:`x_0` and :math:`p_0`
   satisfy

.. math::
  \begin{aligned}
  x_0^T M(\gamma) \geq \mathbf{0}^T \quad \quad \text{and}\quad\quad M(\gamma) p_0 \leq \mathbf{0}
  \end{aligned}

That is, :math:`(x_0, p_0, 0)` is a solution of the game
:math:`M(\gamma)` so
that :math:`V\left(M(\beta_0)\right) = V\left(M(\alpha_0)\right) = 0`.

\* If :math:`\beta_0 < \alpha_0` and
:math:`\gamma \in (\beta_0, \alpha_0)`, then :math:`V(M(\gamma)) = 0`.

Moreover, if :math:`x'` is optimal for the maximizing player in
:math:`M(\gamma')` for :math:`\gamma'\in(\beta_0, \alpha_0)` and
:math:`p''` is optimal for the minimizing player in :math:`M(\gamma'')`
where :math:`\gamma''\in(\beta_0, \gamma')`, then :math:`(x', p'', 0)`
is a solution for
:math:`M(\gamma)`, :math:`\forall \gamma\in (\gamma'', \gamma')`.

  *Proof (Sketch):* If :math:`x'` is optimal for a maximizing player in
  game :math:`M(\gamma')`, then :math:`(x')^T M(\gamma')\geq \mathbf{0}^T`
  and so for all :math:`\gamma<\gamma'`.

.. math:: (x')^T M(\gamma) = (x')^T M(\gamma') + (x')^T(\gamma' - \gamma)A \geq \mathbf{0}^T

hence :math:`V(M(\gamma))\geq 0`. If :math:`p''` is optimal for a
minimizing player in game :math:`M(\gamma'')`, then :math:`M(\gamma)p \leq \mathbf{0}`
and so for all :math:`\gamma''<\gamma`

.. math:: M(\gamma)p'' = M(\gamma'') + (\gamma'' - \gamma)Ap'' \leq \mathbf{0}

hence :math:`V(M(\gamma))\leq 0`.

It is clear from the above argument that :math:`\beta_0`,
:math:`\alpha_0` are the minimal and maximal :math:`\gamma` for which
:math:`V(M(\gamma))=0`.

Moreover, Hamburger et al. (1967) :cite:`hamburger1967computation` show that the
function :math:`\gamma \mapsto V(M(\gamma))` is continuous and
nonincreasing in :math:`\gamma`.

This suggests an algorithm to compute
:math:`(\alpha_0, x_0)` and :math:`(\beta_0, p_0)` for a given
input-output pair :math:`(A, B)`.

Algorithm
---------

Hamburger, Thompson and Weil (1967) :cite:`hamburger1967computation` propose a simple bisection algorithm
to find the minimal and maximal roots (i.e. :math:`\beta_0` and
:math:`\alpha_0`) of the function :math:`\gamma \mapsto V(M(\gamma))`.

Step 1
~~~~~~

First, notice that we can easily find trivial upper and lower bounds for
:math:`\alpha_0` and :math:`\beta_0`.

\* TEP requires that
:math:`x^T(B-\alpha A)\geq \mathbf{0}^T` and :math:`x > \mathbf{0}`, so
if :math:`\alpha` is so large that
:math:`\max_i\{[(B-\alpha A)\iota_n]_i\} < 0`, then TEP ceases to have a
solution.

Accordingly, let **``UB``** be the :math:`\alpha^{*}` that
solves :math:`\max_i\{[(B-\alpha^{*} A)\iota_n]_i\} = 0`.

\* Similar to
the upper bound, if :math:`\beta` is so low that
:math:`\min_j\{[\iota^T_m(B-\beta A)]_j\}>0`, then the EEP has no
solution and so we can define **``LB``** as the :math:`\beta^{*}` that
solves :math:`\min_j\{[\iota^T_m(B-\beta^{*} A)]_j\}=0`.

The *bounds* method calculates these trivial bounds for us

.. code-block:: python3

    n1.bounds()


Step 2
~~~~~~

Compute :math:`\alpha_0` and :math:`\beta_0`

-  Finding :math:`\alpha_0`

   1. Fix :math:`\gamma = \frac{UB + LB}{2}` and compute the solution
      of the two-player zero-sum game associated
      with :math:`M(\gamma)`. We can use either the primal or the dual
      LP problem.
   2. If :math:`V(M(\gamma)) \geq 0`, then set :math:`LB = \gamma`,
      otherwise let :math:`UB = \gamma`.
   3. Iterate on 1. and 2. until :math:`|UB - LB| < \epsilon`.

-  Finding :math:`\beta_0`

   1. Fix :math:`\gamma = \frac{UB + LB}{2}` and compute the solution
      of the two-player zero-sum game associated.
      with :math:`M(\gamma)`. We can use either the primal or the dual
      LP problem.
   2. If :math:`V(M(\gamma)) > 0`, then set :math:`LB = \gamma`,
      otherwise let :math:`UB = \gamma`.
   3. Iterate on 1. and 2. until :math:`|UB - LB| < \epsilon`.


   *Existence*: Since :math:`V(M(LB))>0` and :math:`V(M(UB))<0` and
   :math:`V(M(\cdot))` is a continuous, nonincreasing function, there is
   at least one :math:`\gamma\in[LB, UB]`, s.t. :math:`V(M(\gamma))=0`.

The *zerosum* method calculates the value and optimal strategies
associated with a given :math:`\gamma`.

.. code-block:: python3

    γ = 2

    print(f'Value of the game with γ = {γ}')
    print(n1.zerosum(γ=γ)[0])
    print('Intensity vector (from the primal)')
    print(n1.zerosum(γ=γ)[1])
    print('Price vector (from the dual)')
    print(n1.zerosum(γ=γ, dual=True)[1])


.. code-block:: python3

    numb_grid = 100
    γ_grid = np.linspace(0.4, 2.1, numb_grid)

    value_ex1_grid = np.asarray([n1.zerosum(γ=γ_grid[i])[0]
                                for i in range(numb_grid)])
    value_ex2_grid = np.asarray([n2.zerosum(γ=γ_grid[i])[0]
                                for i in range(numb_grid)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(r'The function $V(M(\gamma))$', fontsize=16)

    for ax, grid, N, i in zip(axes, (value_ex1_grid, value_ex2_grid),
                              (n1, n2), (1, 2)):
        ax.plot(γ_grid, grid)
        ax.set(title=f'Example {i}', xlabel='$\gamma$')
        ax.axhline(0, c='k', lw=1)
        ax.axvline(N.bounds()[0], c='r', ls='--', label='lower bound')
        ax.axvline(N.bounds()[1], c='g', ls='--', label='upper bound')

    plt.show()

The *expansion* method implements the bisection algorithm for
:math:`\alpha_0` (and uses the primal LP problem for :math:`x_0`)

.. code-block:: python3

    α_0, x, p = n1.expansion()
    print(f'α_0 = {α_0}')
    print(f'x_0 = {x}')
    print(f'The corresponding p from the dual = {p}')


The *interest* method implements the bisection algorithm for
:math:`\beta_0` (and uses the dual LP problem for :math:`p_0`)

.. code-block:: python3

    β_0, x, p = n1.interest()
    print(f'β_0 = {β_0}')
    print(f'p_0 = {p}')
    print(f'The corresponding x from the primal = {x}')

Of course, when :math:`\gamma^*` is unique, it is irrelevant which one
of the two methods we use.

In particular, as will be shown below, in
case of an irreducible :math:`(A,B)` (like in Example 1), the maximal
and minimal roots of :math:`V(M(\gamma))` necessarily coincide implying
a ‘’full duality’’ result, i.e. :math:`\alpha_0 = \beta_0 = \gamma^*`,
and that the expansion (and interest) rate :math:`\gamma^*` is unique.

Uniqueness and Irreducibility
-----------------------------

As an illustration, compute first the maximal and minimal roots of
:math:`V(M(\cdot))` for Example 2, which displays a reducible
input-output pair :math:`(A, B)`

.. code-block:: python3

    α_0, x, p = n2.expansion()
    print(f'α_0 = {α_0}')
    print(f'x_0 = {x}')
    print(f'The corresponding p from the dual = {p}')

.. code-block:: python3

    β_0, x, p = n2.interest()
    print(f'β_0 = {β_0}')
    print(f'p_0 = {p}')
    print(f'The corresponding x from the primal = {x}')

As we can see, with a reducible :math:`(A,B)`, the roots found by the
bisection algorithms might differ, so there might be multiple
:math:`\gamma^*` that make the value of the game
with :math:`M(\gamma^*)` zero. (see the figure above).

Indeed, although the von Neumann theorem assures existence of the
equilibrium, Assumptions I and II are not sufficient for uniqueness.
Nonetheless, Kemeny et al. (1967) show that there are at most finitely
many economic solutions, meaning that there are only finitely many
:math:`\gamma^*` that satisfy :math:`V(M(\gamma^*)) = 0` and
:math:`x_0^TBp_0 > 0` and that for each such :math:`\gamma^*_i`, there
is a self-sufficient part of the economy (a sub-economy) that in
equilibrium can expand independently with the expansion
coefficient :math:`\gamma^*_i`.

The following theorem (see Theorem 9.10. in Gale, 1960 :cite:`gale1989theory`) asserts that
imposing irreducibility is sufficient for uniqueness of
:math:`(\gamma^*, x_0, p_0)`.

**Theorem II:** Consider the conditions of Theorem 1. If the economy
:math:`(A,B)` is irreducible, then :math:`\gamma^*=\alpha_0=\beta_0`.

A Special Case
--------------

There is a special :math:`(A,B)` that allows us to simplify the solution
method significantly by invoking the powerful Perron-Frobenius theorem
for non-negative matrices.

**Definition:** We call an economy *simple* if it satisfies 1.
:math:`n=m` 2. Each activity produces exactly one good 3. Each good is
produced by one and only one activity.

These assumptions imply that :math:`B=I_n`, i.e., that :math:`B` can be
written as an identity matrix (possibly after reshuffling its rows and
columns).

The simple model has the following special property (Theorem 9.11. in :cite:`gale1989theory`): if :math:`x_0` and :math:`\alpha_0>0` solve the TEP
with :math:`(A,I_n)`, then

.. math::
  x_0^T = \alpha_0 x_0^T A\hspace{1cm}\Leftrightarrow\hspace{1cm}x_0^T
  A=\left(\frac{1}{\alpha_0}\right)x_0^T

The latter shows that :math:`1/\alpha_0` is a positive eigenvalue of
:math:`A` and :math:`x_0` is the corresponding non-negative left
eigenvector.

The classical result of **Perron and Frobenius** implies
that a non-negative matrix always has a non-negative
eigenvalue-eigenvector pair.

Moreover, if :math:`A` is irreducible, then
the optimal intensity vector :math:`x_0` is positive and *unique* up to
multiplication by a positive scalar.

Suppose that :math:`A` is reducible with :math:`k` irreducible subsets
:math:`S_1,\dots,S_k`. Let :math:`A_i` be the submatrix corresponding to
:math:`S_i` and let :math:`\alpha_i` and :math:`\beta_i` be the
associated expansion and interest factors, respectively. Then we have

.. math::
  \alpha_0 = \max_i \{\alpha_i\}\hspace{1cm}\text{and}\hspace{1cm}\beta_0 = \min_i \{\beta_i\}
