.. _chang_ramsey:

.. include:: /_static/includes/header.raw


********************************************
Competitive Equilibria of Chang Model
********************************************


.. contents:: :depth: 2


**Co-author: Sebastian Graves**

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install polytope

Overview
=============

This lecture describes how Chang :cite:`chang1998credible`
analyzed **competitive equilibria** and a best competitive equilibrium called a  **Ramsey plan**.

He did this by

* characterizing a competitive equilibrium recursively in a way also employed
  in the :doc:`dynamic Stackelberg problems<dyn_stack>` and :doc:`Calvo model<calvo>` lectures
  to pose Stackelberg problems in linear economies, and then

* appropriately adapting an argument of Abreu, Pearce, and Stachetti
  :cite:`APS1990` to describe key features of the  set of competitive equilibria

Roberto Chang :cite:`chang1998credible`  chose a model of  Calvo :cite:`Calvo1978`
as a simple structure that conveys ideas that apply more broadly.

A textbook version of Chang's model appears in chapter 25 of :cite:`Ljungqvist2012`.


This lecture and :doc:`Credible Government Policies in Chang Model<chang_credible>`
can be viewed as more sophisticated and complete treatments of the
topics discussed in :doc:`Ramsey plans, time inconsistency, sustainable plans<calvo>`.

Both this lecture and :doc:`Credible Government Policies in Chang Model<chang_credible>`
make extensive use of an idea to which we apply the nickname
**dynamic programming squared**.

In dynamic programming squared problems there are typically two interrelated Bellman equations

* A Bellman equation for a set of agents or followers with value or value function :math:`v_a`.

* A Bellman equation for a principal or Ramsey planner or Stackelberg leader with value or
  value function :math:`v_p` in which :math:`v_a` appears as an argument.

We encountered problems with this structure in
:doc:`dynamic Stackelberg problems<dyn_stack>`,
:doc:`optimal taxation with state-contingent debt<opt_tax_recur>`,
and other lectures.


We'll start with some standard imports:

.. code-block:: ipython

  import numpy as np
  import polytope
  import quantecon as qe
  import matplotlib.pyplot as plt
  %matplotlib inline


The Setting
-----------

First, we introduce some notation.

For a sequence of scalars
:math:`\vec z \equiv \{z_t\}_{t=0}^\infty`, let
:math:`\vec z^t = (z_0,  \ldots , z_t)`,
:math:`\vec z_t = (z_t, z_{t+1}, \ldots )`.

An infinitely lived
representative agent and an infinitely lived government exist at dates
:math:`t = 0, 1, \ldots`.

The objects in play are

* an initial quantity :math:`M_{-1}` of nominal money holdings

* a sequence of inverse money growth rates :math:`\vec h` and an associated sequence of nominal money holdings :math:`\vec M`

* a sequence of values of money :math:`\vec q`

* a sequence of real money holdings :math:`\vec m`

* a sequence of total tax collections :math:`\vec x`

* a sequence of per capita rates of consumption :math:`\vec c`

* a sequence of per capita incomes :math:`\vec y`

A benevolent government chooses sequences
:math:`(\vec M, \vec h, \vec x)` subject to a sequence of budget
constraints and other constraints imposed by competitive equilibrium.

Given tax collection and price of money sequences, a representative household chooses
sequences :math:`(\vec c, \vec m)` of consumption and real balances.

In competitive equilibrium, the price of money sequence :math:`\vec q` clears
markets, thereby reconciling  decisions of the government and the
representative household.

Chang adopts a version of a model that :cite:`Calvo1978` designed to exhibit
time-inconsistency of a Ramsey policy in a simple and transparent
setting.

By influencing the representative household’s expectations, government actions at
time :math:`t` affect components of household utilities for periods
:math:`s` before :math:`t`.

When setting a path for monetary
expansion rates, the government takes into account how the
household’s anticipations of the government's future actions affect the household's current decisions.

The ultimate source of time inconsistency is that a
time :math:`0` Ramsey planner  takes these
effects into account in designing a plan of government actions for
:math:`t \geq 0`.


Setting
=========


The Household’s Problem
--------------------------

A representative household faces a nonnegative value of money sequence
:math:`\vec q` and sequences :math:`\vec y, \vec x` of income and total
tax collections, respectively.

The household chooses nonnegative
sequences :math:`\vec c, \vec M` of consumption and nominal balances,
respectively, to maximize

.. math::
  :label: eqn_chang_ramsey1

  \sum_{t=0}^\infty \beta^t \left[ u(c_t) + v(q_t M_t ) \right]


subject to

.. math::
   :label: eqn_chang_ramsey2

   q_t M_t  \leq y_t + q_t M_{t-1} - c_t - x_t


and

.. math::
  :label: eqn_chang_ramsey3

  q_t M_t  \leq \bar m



Here :math:`q_t` is the reciprocal of the price level at :math:`t`,
which we can also call the *value of money*.

Chang :cite:`chang1998credible` assumes that

* :math:`u: \mathbb{R}_+ \rightarrow \mathbb{R}` is twice continuously differentiable, strictly concave, and strictly increasing;

* :math:`v: \mathbb{R}_+ \rightarrow \mathbb{R}` is twice continuously differentiable and strictly concave;

* :math:`u'(c)_{c \rightarrow 0}  = \lim_{m \rightarrow 0} v'(m) = +\infty`;

* there is a finite level :math:`m= m^f` such that :math:`v'(m^f) =0`

The household carries real balances out of a period equal to :math:`m_t = q_t M_t`.

Inequality :eq:`eqn_chang_ramsey2` is the household’s time :math:`t` budget constraint.

It tells how real balances :math:`q_t M_t` carried out of period :math:`t` depend
on income, consumption, taxes, and real balances :math:`q_t M_{t-1}`
carried into the period.

Equation :eq:`eqn_chang_ramsey3` imposes an exogenous upper bound
:math:`\bar m` on the household's choice of real balances, where
:math:`\bar m \geq m^f`.

Government
-----------

The government chooses a sequence of inverse money growth rates with
time :math:`t` component
:math:`h_t \equiv {M_{t-1}\over M_t} \in \Pi \equiv
[ \underline \pi, \overline \pi]`, where
:math:`0 < \underline \pi < 1 < { 1 \over \beta } \leq \overline \pi`.

The government faces a sequence of budget constraints with time
:math:`t` component

.. math:: -x_t = q_t (M_t - M_{t-1})

which by using the definitions of :math:`m_t` and :math:`h_t` can also
be expressed as

.. math:: -x_t = m_t (1-h_t)
   :label: eqn_chang_ramsey2a

The  restrictions :math:`m_t \in [0, \bar m]` and :math:`h_t \in \Pi` evidently
imply that :math:`x_t \in X \equiv [(\underline  \pi -1)\bar m,
(\overline \pi -1) \bar m]`.

We define the set :math:`E \equiv [0,\bar m] \times \Pi \times X`,
so that we require that :math:`(m, h, x) \in E`.

To represent the idea that taxes are distorting, Chang makes the following
assumption about outcomes for per capita output:

.. math:: y_t = f(x_t),
   :label: eqn_chang_ramsey3a

where :math:`f: \mathbb{R}\rightarrow \mathbb{R}` satisfies :math:`f(x)  > 0`,
is twice continuously differentiable, :math:`f''(x) < 0`, and
:math:`f(x) = f(-x)` for all :math:`x \in
\mathbb{R}`, so that subsidies and taxes are equally distorting.

Calvo's and Chang's  purpose is not to model the causes of tax distortions in
any detail but simply to summarize
the *outcome* of those distortions via the function :math:`f(x)`.

A key part of the specification is that tax distortions are increasing in the
absolute value of tax revenues.

**Ramsey plan:**
A Ramsey plan is a competitive equilibrium that
maximizes :eq:`eqn_chang_ramsey1`.

Within-period timing of decisions is as follows:

* first, the government chooses :math:`h_t` and :math:`x_t`;

*  then given :math:`\vec q` and its expectations about future values of :math:`x`
   and :math:`y`\ ’s, the household chooses :math:`M_t` and therefore :math:`m_t`
   because :math:`m_t = q_t M_t`;

* then output :math:`y_t = f(x_t)` is realized;

* finally :math:`c_t = y_t`

This within-period timing confronts the government with
choices framed by how the private sector wants to respond when the
government takes time :math:`t` actions that differ from what the
private sector had expected.


This consideration will be important in lecture :doc:`credible government policies<chang_credible>` when
we study *credible government policies*.

The model is designed to focus on the intertemporal trade-offs between
the welfare benefits of deflation and the welfare costs associated with
the high tax collections required to retire money at a rate that
delivers deflation.

A benevolent time :math:`0` government can promote
utility generating increases in real balances only by imposing sufficiently
large distorting tax collections.

To promote the welfare increasing effects of high real balances, the
government wants to induce  *gradual deflation*.

Household’s Problem
----------------------

Given :math:`M_{-1}` and :math:`\{q_t\}_{t=0}^\infty`, the household’s problem is

.. math::

   \begin{aligned}
   \mathcal{L} & = \max_{\vec c, \vec M}
   \min_{\vec \lambda, \vec \mu} \sum_{t=0}^\infty \beta^t
   \bigl\{ u(c_t) + v(M_t q_t) +
   \lambda_t [ y_t - c_t - x_t + q_t M_{t-1} - q_t M_t ]  \\
   & \quad \quad \quad  + \mu_t [\bar m - q_t  M_t] \bigr\}
   \end{aligned}

First-order conditions with respect to :math:`c_t` and :math:`M_t`, respectively, are

.. math::

   \begin{aligned}
   u'(c_t) & = \lambda_t \\
   q_t [ u'(c_t) - v'(M_t q_t) ] & \leq \beta u'(c_{t+1})
   q_{t+1} , \quad = \ {\rm if} \ M_t q_t < \bar m
   \end{aligned}

The last equation expresses Karush-Kuhn-Tucker complementary slackness
conditions (see `here <https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions>`__).

These insist that the inequality is an equality at an interior solution for :math:`M_t`.

Using :math:`h_t = {M_{t-1}\over M_t}` and :math:`q_t = {m_t \over M_t}` in these first-order conditions and rearranging implies

.. math::
  :label: eqn_chang_ramsey4

  m_t [u'(c_t) - v'(m_t) ] \leq \beta u'(f(x_{t+1})) m_{t+1} h_{t+1},
  \quad = \text{ if } m_t < \bar m

Define the following key variable

.. math:: \theta_{t+1} \equiv u'(f(x_{t+1})) m_{t+1} h_{t+1}
   :label: eqn_chang_ramsey5

This is real money balances at time :math:`t+1` measured in units of marginal
utility, which Chang refers to as ‘the marginal utility of real
balances’.

From the standpoint of the household at time :math:`t`, equation :eq:`eqn_chang_ramsey5`
shows that :math:`\theta_{t+1}` intermediates the influences of
:math:`(\vec x_{t+1}, \vec m_{t+1})` on the household’s choice of real balances
:math:`m_t`.

By "intermediates" we mean that the future paths :math:`(\vec x_{t+1},
\vec m_{t+1})` influence :math:`m_t` entirely through their effects on the
scalar :math:`\theta_{t+1}`.

The observation that the one dimensional promised marginal utility of real
balances :math:`\theta_{t+1}` functions in this way is an important step
in constructing a class of competitive equilibria that have a recursive representation.

A closely related observation pervaded the analysis of Stackelberg plans
in lecture  :doc:`dynamic Stackelberg problems<dyn_stack>`.


Competitive Equilibrium
========================

**Definition:**

* A *government policy* is a pair of sequences :math:`(\vec h,\vec x)` where :math:`h_t \in \Pi  \ \forall t \geq 0`.

* A *price system* is a nonnegative value of money sequence :math:`\vec q`.

* An *allocation* is a  triple of nonnegative sequences :math:`(\vec c, \vec m, \vec y)`.

It is required that time :math:`t` components :math:`(m_t, x_t, h_t) \in E`.

**Definition:**

Given :math:`M_{-1}`, a government policy :math:`(\vec h, \vec x)`, price system :math:`\vec q`, and allocation
:math:`(\vec c, \vec m, \vec y)` are said to be a *competitive equilibrium* if

* :math:`m_t = q_t M_t` and :math:`y_t = f(x_t)`.

* The government budget constraint is satisfied.

* Given :math:`\vec q, \vec x, \vec y`, :math:`(\vec c, \vec m)` solves the household’s problem.

Inventory of Objects in Play
=============================

Chang constructs the following objects

1. A set :math:`\Omega` of initial marginal utilities of money :math:`\theta_0`

   * Let :math:`\Omega` denote the set of initial promised marginal utilities of
     money :math:`\theta_0` associated with competitive equilibria.

   * Chang exploits the fact that a competitive equilibrium consists of a first
     period outcome :math:`(h_0, m_0, x_0)` and a continuation competitive
     equilibrium with marginal utility of money :math:`\theta _1 \in \Omega`.

2. Competitive equilibria that have a recursive representation

   * A competitive equilibrium with a recursive representation consists of an
     initial :math:`\theta_0` and a four-tuple of functions :math:`(h, m, x, \Psi)`
     mapping :math:`\theta` into this period’s :math:`(h, m, x)` and
     next period’s :math:`\theta`, respectively.

   * A competitive equilibrium can be represented recursively by iterating on

     .. math::
        :label: Chang500

        \begin{split}
        h_t & = h(\theta_t) \\
        m_t & = m(\theta_t) \\
        x_t & = x(\theta_t) \\
        \theta_{t+1} & = \Psi(\theta_t)
        \end{split}

     starting from :math:`\theta_0`

     The range and domain of :math:`\Psi(\cdot)` are both :math:`\Omega`

3. A recursive representation of a Ramsey plan

   * A recursive representation of a Ramsey plan is a recursive
     competitive equilibrium :math:`\theta_0, (h, m, x, \Psi)` that, among
     all recursive competitive equilibria, maximizes :math:`\sum_{t=0}^\infty
     \beta^t \left[ u(c_t) + v(q_t M_t ) \right]`.

   * The Ramsey planner chooses :math:`\theta_0, (h, m, x, \Psi)` from among
     the set of recursive competitive equilibria at time :math:`0`.

   * Iterations on the function :math:`\Psi` determine subsequent
     :math:`\theta_t`\ ’s that summarize the aspects of the continuation
     competitive equilibria that influence the household’s decisions.

   * At time :math:`0`, the Ramsey planner commits to this implied sequence
     :math:`\{\theta_t\}_{t=0}^\infty` and therefore to an associated sequence
     of continuation competitive equilibria.

4. A characterization of time-inconsistency of a Ramsey plan

   * Imagine that after a ‘revolution’ at time :math:`t \geq 1`, a new Ramsey
     planner is  given the opportunity to ignore history and solve a brand new
     Ramsey plan.

   * This new planner would want to reset the :math:`\theta_t` associated
     with the original Ramsey plan to :math:`\theta_0`.

   * The incentive to reinitialize :math:`\theta_t` associated with this
     revolution experiment indicates the time-inconsistency of the Ramsey plan.

   * By resetting :math:`\theta` to :math:`\theta_0`, the new planner avoids
     the costs at time :math:`t` that the original Ramsey planner must pay to
     reap the beneficial effects that the original Ramsey plan for
     :math:`s \geq t` had achieved via its influence on the household’s
     decisions for :math:`s = 0, \ldots, t-1`.


Analysis
===========

A competitive equilibrium is a triple of sequences
:math:`(\vec m, \vec x, \vec h)  \in E^\infty` that satisfies
:eq:`eqn_chang_ramsey2`, :eq:`eqn_chang_ramsey3`,  and :eq:`eqn_chang_ramsey4`.

Chang works with  a set of competitive equilibria defined as follows.


**Definition:**  :math:`CE = \bigl\{ (\vec m, \vec x, \vec h) \in E^\infty` such that
:eq:`eqn_chang_ramsey2`, :eq:`eqn_chang_ramsey3`,  and :eq:`eqn_chang_ramsey4`
are satisfied :math:`\bigr\}`.

:math:`CE` is not empty because there exists a competitive equilibrium
with :math:`h_t =1` for all :math:`t \geq 1`, namely, an equilibrium with a
constant money supply and constant price level.

Chang establishes that :math:`CE` is also compact.

Chang makes the following key observation that combines ideas of Abreu, Pearce,
and Stacchetti :cite:`APS1990` with insights of Kydland and Prescott :cite:`kydland1980dynamic`.

**Proposition:**
The continuation of a competitive equilibrium is a competitive equilibrium.

That is, :math:`(\vec m, \vec x, \vec h) \in CE` implies that :math:`(\vec m_t,
\vec x_t, \vec h_t) \in CE \  \forall \ t \geq 1`.

(Lecture :doc:`dynamic Stackelberg problems<dyn_stack>` also used a version of this insight)

We can now state that a **Ramsey problem** is to

.. math:: \max_{(\vec m, \vec x, \vec h) \in E^\infty} \sum_{t=0}^\infty \beta^t \left[ u(c_t) + v(m_t) \right]

subject to restrictions :eq:`eqn_chang_ramsey2`, :eq:`eqn_chang_ramsey3`, and :eq:`eqn_chang_ramsey4`.

Evidently, associated with any competitive equilibrium :math:`(m_0, x_0)` is an
implied value of :math:`\theta_0 = u'(f(x_0))(m_0 + x_0)`.

To bring out a recursive structure inherent in the Ramsey problem, Chang defines the set

.. math::

  \Omega = \left\{ \theta \in \mathbb{R} \
  \text{ such that } \ \theta = u'(f(x_0)) (m_0 + x_0) \ \text{ for some } \
  (\vec m, \vec x, \vec h) \in CE \right\}

Equation :eq:`eqn_chang_ramsey4` inherits from the household’s Euler equation for
money holdings the property that the value of :math:`m_0` consistent with
the representative household’s choices depends on :math:`(\vec h_1, \vec m_1)`.

This dependence is captured in the definition above by making :math:`\Omega`
be the set of first period values of :math:`\theta_0` satisfying
:math:`\theta_0 = u'(f(x_0)) (m_0 + x_0)` for first period component
:math:`(m_0,h_0)` of competitive equilibrium sequences
:math:`(\vec m, \vec x, \vec h)`.

Chang establishes that :math:`\Omega` is a nonempty and compact subset of :math:`\mathbb{R}_+`.

Next Chang advances:

**Definition:** :math:`\Gamma(\theta) = \{ (\vec m, \vec x, \vec h) \in CE |  \theta = u'(f(x_0))(m_0 + x_0) \}`.

Thus, :math:`\Gamma(\theta)` is the set of competitive equilibrium sequences
:math:`(\vec m, \vec x, \vec h)` whose first period components
:math:`(m_0, h_0)` deliver the prescribed value :math:`\theta` for first
period marginal utility.

If we knew the sets :math:`\Omega, \Gamma(\theta)`, we could use the following
two-step procedure to find at least the *value* of the Ramsey
outcome to the representative household

1. Find the indirect value function :math:`w(\theta)` defined as

.. math:: w(\theta) = \max_{(\vec m, \vec x, \vec h) \in \Gamma(\theta)} \sum_{t=0}^\infty \beta^t \left[ u(f(x_t)) + v(m_t) \right]

2. Compute the value of the Ramsey outcome by solving :math:`\max_{\theta \in \Omega} w(\theta)`.

Thus, Chang states the following

**Proposition**:

:math:`w(\theta)` satisfies the Bellman equation

.. math:: w(\theta) = \max_{x,m,h,\theta'} \bigl\{ u(f(x)) + v(m) + \beta w(\theta') \bigr\}
   :label: eqn_chang_ramsey7

where maximization is subject to

.. math::
   (m,x,h)  \in E \  {\rm and} \ \theta' \in \Omega
   :label: eqn_chang_ramsey8

and

.. math::
   \theta = u'(f(x)) (m+x)
   :label: eqn_chang_ramsey9

and

.. math::
   -x = m(1-h)
   :label: eqn_chang_ramsey10

and

.. math::
   m \cdot [ u'(f(x)) - v'(m) ]  \leq \beta \theta' , \quad = \ {\rm if} \ m < \bar m
   :label: eqn_chang_ramsey11

Before we use this proposition to recover a recursive representation of the
Ramsey plan, note that the proposition relies on knowing the set :math:`\Omega`.

To find :math:`\Omega`, Chang uses the insights of Kydland and Prescott
:cite:`kydland1980dynamic` together with a method based on the
Abreu, Pearce, and Stacchetti :cite:`APS1990` iteration to convergence on an
operator :math:`B` that maps continuation values into values.

We want an operator that maps a continuation :math:`\theta` into a current :math:`\theta`.

Chang lets :math:`Q` be a nonempty, bounded subset of :math:`\mathbb{R}`.

Elements of the set :math:`Q` are taken to be candidate values for continuation marginal utilities.

Chang defines an operator

.. math::
  B(Q)  = \theta \in \mathbb{R} \
  \text{ such that there is } \
  (m,x,h, \theta') \in E \times Q

such that :eq:`eqn_chang_ramsey9`, :eq:`eqn_chang_ramsey10`,
and :eq:`eqn_chang_ramsey11` hold.



Thus, :math:`B(Q)` is the set of first period :math:`\theta`\ ’s attainable with
:math:`(m,x,h) \in E` and some :math:`\theta' \in Q`.

**Proposition**:

i. :math:`Q \subset B(Q)` implies :math:`B(Q) \subset \Omega` (‘self-generation’).

ii. :math:`\Omega = B(\Omega)` (‘factorization’).

The proposition characterizes :math:`\Omega` as the largest fixed point
of :math:`B`.

It is easy to establish that :math:`B(Q)` is a monotone
operator.

This property allows Chang to compute :math:`\Omega` as the
limit of iterations on :math:`B` provided that iterations begin from a
sufficiently large initial set.

Some Useful Notation
---------------------

Let :math:`\vec h^t = (h_0, h_1, \ldots, h_t)` denote a history of
inverse money creation rates with time :math:`t` component
:math:`h_t \in \Pi`.

A *government strategy* :math:`\sigma=\{\sigma_t\}_{t=0}^\infty` is a :math:`\sigma_0 \in \Pi`
and for :math:`t \geq 1` a sequence of functions
:math:`\sigma_t: \Pi^{t-1} \rightarrow \Pi`.

Chang restricts the
government’s choice of strategies to the following space:

.. math::
  CE_\pi = \{ {\vec h} \in \Pi^\infty: \text{ there is some } \
  (\vec m, \vec x) \ \text{ such that } \ (\vec m, \vec x, \vec h) \in CE \}

In words, :math:`CE_\pi` is the set of money growth sequences consistent
with the existence of competitive equilibria.

Chang observes that :math:`CE_\pi` is nonempty and compact.

**Definition**: :math:`\sigma` is said to be *admissible* if for all :math:`t \geq 1`
and after any history :math:`\vec h^{t-1}`, the continuation
:math:`\vec h_t` implied by :math:`\sigma` belongs to :math:`CE_\pi`.


Admissibility of :math:`\sigma` means that anticipated policy choices
associated with :math:`\sigma` are consistent with the existence of
competitive equilibria after each possible subsequent history.

After any history :math:`\vec h^{t-1}`, admissibility restricts the government’s
choice in period :math:`t` to the set

.. math:: CE_\pi^0 = \{ h \in \Pi: {\rm there \ is } \ \vec h \in CE_\pi \ {\rm with } \ h=h_0 \}

In words, :math:`CE_\pi^0` is the set of all first period money growth
rates :math:`h=h_0`, each of which is consistent with the existence of a
sequence of money growth rates :math:`\vec h` starting from :math:`h_0`
in the initial period and for which a competitive equilibrium exists.

**Remark:** :math:`CE_\pi^0 = \{h \in \Pi: \text{ there is } \ (m,\theta') \in [0, \bar m] \times \Omega \ \text{ such that } \
m u'[ f((h-1)m) - v'(m)]  \leq \beta \theta' \ \text{ with equality if } \  m < \bar m \}`.

**Definition:**
An *allocation rule* is a sequence of functions
:math:`\vec \alpha = \{\alpha_t\}_{t=0}^\infty` such that
:math:`\alpha_t: \Pi^t \rightarrow [0, \bar m] \times X`.

Thus, the time :math:`t` component of :math:`\alpha_t(h^t)` is a pair of functions
:math:`(m_t(h^t), x_t(h^t))`.

**Definition:** Given an admissible government strategy
:math:`\sigma`, an allocation rule :math:`\alpha` is called
*competitive* if given any history :math:`\vec h^{t-1}` and
:math:`h_t \in CE_\pi^0`, the continuations of :math:`\sigma` and
:math:`\alpha` after :math:`(\vec h^{t-1},h_t)` induce a competitive
equilibrium sequence.

Another Operator
------------------

At this point it is convenient to introduce another operator that can be
used to compute a Ramsey plan.

For computing a Ramsey plan, this
operator is wasteful because it works with a state vector that is bigger
than necessary.

We introduce this operator because it helps to prepare
the way for Chang’s operator called :math:`\tilde D(Z)` that we shall describe in lecture :doc:`credible government policies<chang_credible>`.

It is also useful because a fixed point of the operator to
be defined here provides a good guess for an initial set
from which to initiate iterations on Chang’s set-to-set operator  :math:`\tilde D(Z)`
to be described in lecture :doc:`credible government policies<chang_credible>`.

Let :math:`S` be the set of all pairs :math:`(w, \theta)` of competitive
equilibrium values and associated initial marginal utilities.

Let :math:`W` be a bounded set of *values* in :math:`\mathbb{R}`.

Let :math:`Z` be a nonempty subset of :math:`W \times \Omega`.

Think of using pairs :math:`(w', \theta')` drawn from :math:`Z` as candidate continuation
value, :math:`\theta` pairs.

Define the operator

   .. math:: D(Z) = \Bigl\{ (w,\theta): {\rm there \ is } \ h \in CE_\pi^0

   .. math:: \text{ and a four-tuple } \ (m(h), x(h), w'(h), \theta'(h)) \in [0,\bar m]\times X \times Z

such that

   .. math:: w = u(f(x( h))) + v(m( h)) + \beta w'( h)
      :label: eqn_chang_ramsey120

   .. math:: \theta = u'(f(x( h))) ( m( h) + x( h))
      :label: eqn_chang_ramsey130

   .. math:: x(h) = m(h) (h-1)
      :label: eqn_chang_ramsey_150

   .. math:: m(h) (u'(f(x(h))) - v'(m(h))) \leq \beta \theta'(h)
      :label: eqn_chang_ramsey160

   .. math:: \quad \quad \ \text{ with equality if } m(h) < \bar m \Bigr\}

It is possible to establish.

**Proposition:**

i. If :math:`Z \subset D(Z)`, then :math:`D(Z) \subset S` (‘self-generation’).

ii. :math:`S = D(S)` (‘factorization’).

**Proposition:**

i. Monotonicity of :math:`D`: :math:`Z \subset Z'` implies :math:`D(Z) \subset D(Z')`.

ii. :math:`Z` compact implies that :math:`D(Z)` is compact.

It can be shown that :math:`S` is compact and that therefore there
exists a :math:`(w, \theta)` pair within this set that attains the
highest possible value :math:`w`.

This :math:`(w, \theta)` pair i
associated with a Ramsey plan.

Further, we can compute :math:`S` by
iterating to convergence on :math:`D` provided that one begins with a
sufficiently large initial set :math:`S_0`.


As a very useful by-product, the algorithm that finds the largest fixed
point :math:`S = D(S)` also produces the Ramsey plan, its value
:math:`w`, and the associated competitive equilibrium.

Calculating all Promise-Value Pairs in CE
=============================================================

Above we have defined the :math:`D(Z)` operator as:

.. math::

    D(Z) = \{ (w,\theta): \exists  h \in CE^0_\pi \text{ and } (m(h),x(h),w'(h),\theta'(h)) \in [0,\bar m] \times X \times Z

such that

.. math::  w = u(f(x(h))) + v(m(h)) + \beta w'(h)

.. math::  \theta = u'(f(x(h)))(m(h) + x(h))

.. math::  x(h) = m(h)(h-1)

.. math::  m(h)(u'(f(x(h))) - v'(m(h))) \leq \beta \theta'(h) \text{ (with equality if } m(h) < \bar m) \}

We noted that the set :math:`S` can be found by iterating to convergence
on :math:`D`, provided that we start with a sufficiently large initial
set :math:`S_0`.

Our implementation builds on ideas `in this
notebook <https://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/recursive_repeated_games.ipynb>`__.

To find :math:`S` we use a numerical algorithm called the *outer
hyperplane approximation algorithm*.

It was invented by  Judd, Yeltekin, Conklin :cite:`JuddYeltekinConklin2003`.

This algorithm constructs the smallest convex set that contains the
fixed point of the :math:`D(S)` operator.

Given that we are finding the smallest convex set that contains
:math:`S`, we can represent it on a computer as the intersection of a
finite number of half-spaces.

Let :math:`H` be a set of subgradients, and :math:`C` be a set of
hyperplane levels.

We approximate :math:`S` by:

.. math::  \tilde S = \{(w,\theta)| H \cdot (w,\theta) \leq C \}

A key feature of this algorithm is that we discretize the action space,
i.e., we create a grid of possible values for :math:`m` and :math:`h`
(note that :math:`x` is implied by :math:`m` and :math:`h`). This
discretization simplifies computation of :math:`\tilde S` by allowing us
to find it by solving a sequence of linear programs.

The *outer hyperplane approximation algorithm* proceeds as follows:

1. Initialize subgradients, :math:`H`, and hyperplane levels,
   :math:`C_0`.

2. Given a set of subgradients, :math:`H`, and hyperplane levels,
   :math:`C_t`, for each subgradient :math:`h_i \in H`:

   -  Solve a linear program (described below) for each action in the
      action space.

   -  Find the maximum and update the corresponding hyperplane level,
      :math:`C_{i,t+1}`.

3. If :math:`|C_{t+1}-C_t| > \epsilon`, return to 2.

**Step 1** simply creates a large initial set :math:`S_0`.

Given some set :math:`S_t`, **Step 2** then constructs the set
:math:`S_{t+1} = D(S_t)`. The linear program in Step 2 is designed to
construct a set :math:`S_{t+1}` that is as large as possible while
satisfying the constraints of the :math:`D(S)` operator.

To do this, for each subgradient :math:`h_i`, and for each point in the
action space :math:`(m_j,h_j)`, we solve the following problem:

.. math::  \max_{[w',\theta']} h_i \cdot (w,\theta)

subject to

.. math::  H \cdot (w',\theta') \leq C_t

.. math::  w = u(f(x_j)) + v(m_j) + \beta w'

.. math::  \theta = u'(f(x_j))(m_j + x_j)

.. math::  x_j = m_j(h_j-1)

.. math::  m_j(u'(f(x_j)) - v'(m_j)) \leq \beta \theta'\hspace{2mm} (= \text{if } m_j < \bar m)

This problem maximizes the hyperplane level for a given set of actions.

The second part of Step 2 then finds the maximum possible hyperplane
level across the action space.

The algorithm constructs a sequence of progressively smaller sets
:math:`S_{t+1} \subset S_t \subset S_{t-1} \cdots
\subset S_0`.

**Step 3** ends the algorithm when the difference between these sets is
small enough.

We have created a Python class that solves the model assuming the
following functional forms:

.. math::  u(c) = log(c)

.. math::  v(m) = \frac{1}{500}(m \bar m - 0.5m^2)^{0.5}

.. math::  f(x) = 180 - (0.4x)^2

The remaining parameters :math:`\{\beta, \bar m, \underline h, \bar h\}`
are then variables to be specified for an instance of the Chang class.

Below we use the class to solve the model and plot the resulting
equilibrium set, once with :math:`\beta = 0.3` and once with
:math:`\beta = 0.8`.

(Here we have set the number of subgradients to 10 in order to speed up the
code for now - we can increase accuracy by increasing the number of subgradients)

.. literalinclude:: /_static/lecture_specific/chang_credible/changecon.py

.. code-block:: python3

    ch1 = ChangModel(β=0.3, mbar=30, h_min=0.9, h_max=2, n_h=8, n_m=35, N_g=10)
    ch1.solve_sustainable()

.. code-block:: python3

  def plot_competitive(ChangModel):
      """
      Method that only plots competitive equilibrium set
      """
      poly_C = polytope.Polytope(ChangModel.H, ChangModel.c1_c)
      ext_C = polytope.extreme(poly_C)

      fig, ax = plt.subplots(figsize=(7, 5))

      ax.set_xlabel('w', fontsize=16)
      ax.set_ylabel(r"$\theta$", fontsize=18)

      ax.fill(ext_C[:,0], ext_C[:,1], 'r', zorder=0)
      ChangModel.min_theta = min(ext_C[:, 1])
      ChangModel.max_theta = max(ext_C[:, 1])

      # Add point showing Ramsey Plan
      idx_Ramsey = np.where(ext_C[:, 0] == max(ext_C[:, 0]))[0][0]
      R = ext_C[idx_Ramsey, :]
      ax.scatter(R[0], R[1], 150, 'black', 'o', zorder=1)
      w_min = min(ext_C[:, 0])

      # Label Ramsey Plan slightly to the right of the point
      ax.annotate("R", xy=(R[0], R[1]), xytext=(R[0] + 0.03 * (R[0] - w_min),
                  R[1]), fontsize=18)

      plt.tight_layout()
      plt.show()

  plot_competitive(ch1)

.. code-block:: python3

    ch2 = ChangModel(β=0.8, mbar=30, h_min=0.9, h_max=1/0.8,
                     n_h=8, n_m=35, N_g=10)
    ch2.solve_sustainable()

.. code-block:: python3

    plot_competitive(ch2)

Solving a Continuation Ramsey Planner's Bellman Equation
=============================================================


In this section we solve the Bellman equation confronting a **continuation Ramsey planner**.

The construction of a Ramsey plan is decomposed into a two subproblems in :doc:`Ramsey plans, time inconsistency, sustainable plans<calvo>`
and :doc:`dynamic Stackelberg problems<dyn_stack>`.

* Subproblem 1 is faced by a sequence of continuation Ramsey planners at :math:`t \geq 1`.

* Subproblem 2 is faced by a Ramsey planner at :math:`t = 0`.

The problem is:

.. math::  J(\theta) = \max_{m,x,h,\theta'} u(f(x)) + v(m) + \beta J(\theta')

subject to:

.. math::  \theta \leq u'(f(x))x + v'(m)m + \beta \theta'

.. math::  \theta = u'(f(x))(m + x )

.. math::  x = m(h-1)

.. math::  (m,x,h) \in E

.. math::  \theta' \in \Omega

To solve this Bellman equation, we must know the
set :math:`\Omega`.

We have solved the Bellman equation for the two sets of parameter values
for which we computed the equilibrium value sets above.

Hence for these parameter configurations, we know the bounds of :math:`\Omega`.

The two sets of parameters
differ only in the level of :math:`\beta`.


From the figures earlier in this lecture,  we know that when :math:`\beta = 0.3`,
:math:`\Omega = [0.0088,0.0499]`, and when :math:`\beta = 0.8`,
:math:`\Omega = [0.0395,0.2193]`

.. code-block:: python3

    ch1 = ChangModel(β=0.3, mbar=30, h_min=0.99, h_max=1/0.3,
                     n_h=8, n_m=35, N_g=50)
    ch2 = ChangModel(β=0.8, mbar=30, h_min=0.1, h_max=1/0.8,
                     n_h=20, n_m=50, N_g=50)

.. code-block:: python3

    ch1.solve_bellman(θ_min=0.01, θ_max=0.0499, order=30, tol=1e-6)
    ch2.solve_bellman(θ_min=0.045, θ_max=0.15, order=30, tol=1e-6)


First, a quick check that our approximations of the value functions are
good.

We do this by calculating the residuals between iterates on the value function on a fine grid:

.. code-block:: python3

    max(abs(ch1.resid_grid)), max(abs(ch2.resid_grid))



The value functions plotted below trace out the right edges of the sets
of equilibrium values plotted above

.. code-block:: python3

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, model in zip(axes, (ch1, ch2)):
        ax.plot(model.θ_grid, model.p_iter)
        ax.set(xlabel=r"$\theta$",
               ylabel=r"$J(\theta)$",
               title=rf"$\beta = {model.β}$")

    plt.show()


The next figure plots the optimal policy functions; values of
:math:`\theta',m,x,h` for each value of the state :math:`\theta`:

.. code-block:: python3

    for model in (ch1, ch2):

        fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
        fig.suptitle(rf"$\beta = {model.β}$", fontsize=16)

        plots = [model.θ_prime_grid, model.m_grid,
                 model.h_grid, model.x_grid]
        labels = [r"$\theta'$", "$m$", "$h$", "$x$"]

        for ax, plot, label in zip(axes.flatten(), plots, labels):
            ax.plot(model.θ_grid_fine, plot)
            ax.set_xlabel(r"$\theta$", fontsize=14)
            ax.set_ylabel(label, fontsize=14)

        plt.show()

With the first set of parameter values,  the value of :math:`\theta'` chosen by the Ramsey
planner quickly hits the upper limit of :math:`\Omega`.

But with the second set of parameters  it converges to a value in the interior of the set.

Consequently, the choice of :math:`\bar \theta` is clearly important with the first set of parameter values.

One way of seeing this is plotting :math:`\theta'(\theta)` for each set
of parameters.

With the first set of parameter values,  this function does not intersect the
45-degree line until :math:`\bar \theta`, whereas in the second set of parameter values, it
intersects in the interior.

.. code-block:: python3

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, model in zip(axes, (ch1, ch2)):
        ax.plot(model.θ_grid_fine, model.θ_prime_grid, label=r"$\theta'(\theta)$")
        ax.plot(model.θ_grid_fine, model.θ_grid_fine, label=r"$\theta$")
        ax.set(xlabel=r"$\theta$", title=rf"$\beta = {model.β}$")

    axes[0].legend()
    plt.show()

Subproblem 2 is equivalent to the planner choosing the initial value of
:math:`\theta` (i.e. the value which maximizes the value function).

From this starting point, we can then trace out the paths for
:math:`\{\theta_t,m_t,h_t,x_t\}_{t=0}^\infty` that support this
equilibrium.

These are shown below for both sets of parameters

.. code-block:: python3

    for model in (ch1, ch2):

        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        fig.suptitle(rf"$\beta = {model.β}$")

        plots = [model.θ_series, model.m_series, model.h_series, model.x_series]
        labels = [r"$\theta$", "$m$", "$h$", "$x$"]

        for ax, plot, label in zip(axes.flatten(), plots, labels):
            ax.plot(plot)
            ax.set(xlabel='t', ylabel=label)

        plt.show()

Next Steps
------------

In :doc:`Credible Government Policies in Chang Model<chang_credible>` we shall find
a subset of competitive equilibria that are **sustainable**
in the sense that a sequence of government administrations  that chooses
sequentially, rather than once and for all at time :math:`0` will choose to implement them.

In the process of constructing them, we shall construct another,
smaller set of competitive equilibria.
