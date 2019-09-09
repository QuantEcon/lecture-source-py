.. _calvo:

.. include:: /_static/includes/header.raw

.. highlight:: python3


********************************************************
Ramsey Plans, Time Inconsistency, Sustainable Plans
********************************************************


.. index::
    single: Models; Additive functionals

.. contents:: :depth: 2

**Co-author: Sebastian Graves**

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
=========


This lecture describes a linear-quadratic version of a model that Guillermo Calvo :cite:`Calvo1978`
used to illustrate the **time inconsistency** of optimal government
plans.

Like Chang :cite:`chang1998credible`, we use the model as a laboratory in which to explore the consequences of
different timing protocols for government decision making.

The model focuses attention on intertemporal tradeoffs between

- welfare benefits that anticipated deflation generates by increasing a representative agent's liquidity as measured by his or her real money balances, and

- costs associated with  distorting taxes that must be used to withdraw money from the economy in order to generate anticipated deflation

The model features

-  rational expectations

-  costly government actions at all dates :math:`t \geq 1` that increase household
   utilities at dates before :math:`t`

-  two Bellman equations, one that expresses the private sector's  expectation of future inflation
   as a function of current and future government actions, another that
   describes  the value function of a  Ramsey planner

A theme of this lecture is that  timing protocols affect  outcomes.



We'll use ideas from  papers by Cagan :cite:`Cagan`, Calvo :cite:`Calvo1978`, Stokey :cite:`stokey1989reputation`, :cite:`Stokey1991`,
Chari and Kehoe :cite:`chari1990sustainable`, Chang :cite:`chang1998credible`, and Abreu :cite:`Abreu` as
well as from chapter 19 of :cite:`Ljungqvist2012`.

In addition, we'll use ideas from linear-quadratic dynamic programming
described in  :doc:`Linear Quadratic Control<lqcontrol>` as applied to Ramsey problems
in :doc:`Stackelberg problems<dyn_stack>`.

In particular, we have specified the model in a way that allows us to use
linear-quadratic dynamic programming to compute an optimal government
plan under a timing protocol in which a government chooses an infinite
sequence of money supply growth rates once and for all at time
:math:`0`.

We'll start with some imports:

.. code-block:: ipython

    import numpy as np
    from quantecon import LQ
    import matplotlib.pyplot as plt
    %matplotlib inline

The Model
==========

There is no uncertainty.

Let:

- :math:`p_t` be the log of the price level

- :math:`m_t` be the log of nominal money balances

- :math:`\theta_t = p_{t+1} - p_t` be the net rate of inflation between :math:`t` and :math:`t+1`

- :math:`\mu_t = m_{t+1} - m_t` be the net rate of growth of nominal balances

The demand for real balances is governed by a perfect foresight
version of the Cagan :cite:`Cagan` demand function:

.. math::
    :label: eq_old1

    m_t - p_t = -\alpha(p_{t+1} - p_t) \: , \: \alpha > 0

for :math:`t \geq 0`.

Equation :eq:`eq_old1` asserts that the demand for real balances is inversely
related to the public's expected rate of inflation, which here equals
the actual rate of inflation.

(When there is no uncertainty, an assumption of **rational expectations** simplifies to **perfect foresight**).

(See :cite:`Sargent77hyper` for a rational expectations version of the model when there is uncertainty)

Subtracting the demand function at time :math:`t` from the demand
function at :math:`t+1` gives:

.. math::  \mu_t - \theta_t = -\alpha \theta_{t+1} + \alpha \theta_t

or

.. math::
    :label: eq_old2

     \theta_t = \frac{\alpha}{1+\alpha} \theta_{t+1} + \frac{1}{1+\alpha} \mu_t

Because :math:`\alpha > 0`,  :math:`0 < \frac{\alpha}{1+\alpha} < 1`.


**Definition:** For a scalar :math:`x_t`, let :math:`L^2` be the space of sequences
:math:`\{x_t\}_{t=0}^\infty` satisfying

.. math::  \sum_{t=0}^\infty  x_t^2 < +\infty

We say that a sequence that belongs to :math:`L^2`   is **square summable**.

When we assume that the sequence :math:`\vec \mu = \{\mu_t\}_{t=0}^\infty` is square summable and we require that the sequence
:math:`\vec \theta = \{\theta_t\}_{t=0}^\infty` is square summable,
the linear difference equation :eq:`eq_old2` can be solved forward to get:

.. math::
    :label: eq_old3

     \theta_t = \frac{1}{1+\alpha} \sum_{j=0}^\infty \left(\frac{\alpha}{1+\alpha}\right)^j \mu_{t+j}

**Insight:** In the spirit of Chang :cite:`chang1998credible`, note that equations :eq:`eq_old1` and :eq:`eq_old3` show that :math:`\theta_t` intermediates
how choices of :math:`\mu_{t+j}, \ j=0, 1, \ldots` impinge on time :math:`t`
real balances :math:`m_t - p_t = -\alpha \theta_t`.

We shall use this insight to help us simplify and analyze government policy problems.

That future rates of money creation influence earlier rates of inflation
creates optimal government policy problems in which timing protocols
matter.

We can rewrite the model  as:

.. math::

    \begin{bmatrix}
      1 \\
      \theta_{t+1}
    \end{bmatrix} =
    \begin{bmatrix}
      1 & 0 \\
      0 & \frac{1+\alpha}{\alpha}
    \end{bmatrix}
    \begin{bmatrix}
      1 \\
      \theta_{t}
    \end{bmatrix}  +
    \begin{bmatrix}
      0 \\
      -\frac{1}{\alpha}
    \end{bmatrix}
    \mu_t

or

.. math::
    :label: eq_old4

     x_{t+1} = A x_t + B \mu_t

We write the model in the state-space form :eq:`eq_old4` even though :math:`\theta_0` is to be determined and so is not an initial condition
as it ordinarily would be in the state-space model described in :doc:`Linear Quadratic Control<lqcontrol>`.

We write the model in the form :eq:`eq_old4` because we want to apply an approach described in  :doc:`Stackelberg problems<dyn_stack>`.

Assume that a representative household's utility of real balances at
time :math:`t` is:

.. math::
    :label: eq_old5

    U(m_t - p_t) = a_0 + a_1 (m_t - p_t) - \frac{a_2}{2} (m_t - p_t)^2, \quad a_0 > 0, a_1 > 0, a_2 > 0

The "bliss level" of real balances is then :math:`\frac{a_1}{a_2}`.

The money demand function :eq:`eq_old1` and the utility function :eq:`eq_old5`
imply that utility maximizing or  bliss level of real balances is attained when:

.. math::  \theta_t = \theta^* = -\frac{a_1}{a_2 \alpha}

Below, we introduce the discount factor :math:`\beta \in (0,1)` that a representative household and a benevolent government
both use to discount future utilities.

(If we set parameters so that :math:`\theta^* = \log(\beta)`, then we can
regard a recommendation to set :math:`\theta_t = \theta^*` as a "poor
man's Friedman rule" that attains Milton Friedman's **optimal quantity of money**)

Via equation :eq:`eq_old3`, a government plan
:math:`\vec \mu = \{\mu_t \}_{t=0}^\infty` leads to an equilibrium
sequence of inflation outcomes
:math:`\vec \theta = \{ \theta_t \}_{t=0}^\infty`.

We assume that social costs :math:`\frac{c}{2} \mu_t^2` are incurred at
:math:`t` when the government changes the stock of nominal money
balances at rate :math:`\mu_t`.

Therefore, the one-period welfare function of a benevolent government
is:

.. math::
    :label: eq_old6

    -s(\theta_t, \mu_t) \equiv - r(x_t,\mu_t) = \begin{bmatrix} 1 \\ \theta_t \end{bmatrix}' \begin{bmatrix} a_0 & -\frac{a_1 \alpha}{2} \\ -\frac{a_1 \alpha}{2} & -\frac{a_2 \alpha^2}{2} \end{bmatrix} \begin{bmatrix} 1 \\ \theta_t \end{bmatrix} - \frac{c}{2} \mu_t^2 =  - x_t'Rx_t - Q \mu_t^2

Household welfare is summarized by:

.. math::
    :label: eq_old7

    v_0 = - \sum_{t=0}^\infty \beta^t r(x_t,\mu_t) = - \sum_{t=0}^\infty \beta^t s(\theta_t,\mu_t)

We can represent the dependence of  :math:`v_0` on :math:`(\vec \theta, \vec \mu)` recursively via

.. math::
    :label: eq_old8

    v_t = s(\theta_t, \mu_t) + \beta v_{t+1}

Structure
==========


The following structure is induced by private agents'
behavior as summarized by the demand function for money :eq:`eq_old1` that leads to equation :eq:`eq_old3` that tells how future
settings of :math:`\mu` affect the current value of :math:`\theta`.

Equation :eq:`eq_old3` maps a **policy** sequence of money growth rates
:math:`\vec \mu =\{\mu_t\}_{t=0}^\infty \in L^2`  into an inflation sequence
:math:`\vec \theta = \{\theta_t\}_{t=0}^\infty \in L^2`.

These, in turn, induce a discounted value to a government sequence
:math:`\vec v = \{v_t\}_{t=0}^\infty \in L^2` that satisfies the
recursion

.. math::  v_t = s(\theta_t,\mu_t) + \beta v_{t+1}

where we have called :math:`s(\theta_t, \mu_t) = r(x_t, \mu_t)` as
above.

Thus, we have a triple of sequences
:math:`\vec \mu, \vec \theta, \vec v` associated with a
:math:`\vec \mu \in L^2`.

At this point :math:`\vec \mu \in L^2` is an arbitrary exogenous policy.

To make :math:`\vec \mu` endogenous, we require a theory of government
decisions.



Intertemporal Influences
=========================

Criterion function :eq:`eq_old7` and the constraint system :eq:`eq_old4` exhibit the following
structure:

-  Setting :math:`\mu_t \neq 0` imposes costs
   :math:`\frac{c}{2} \mu_t^2` at time :math:`t` and at no other times;
   but

-  The money growth rate :math:`\mu_t` affects the representative
   household's one-period utilities at all dates
   :math:`s = 0, 1, \ldots, t`.

That settings of :math:`\mu` at one date affect household utilities at
earlier dates sets the stage for the emergence of a time-inconsistent
optimal government plan  under a Ramsey (also called a Stackelberg)  timing protocol.

We'll study outcomes under a Ramsey timing protocol below.

But we'll also study the consequences of other timing protocols.

Four Models of Government Policy
=================================

We consider four models of policymakers that  differ in

-  what a policymaker is allowed to choose, either a sequence
   :math:`\vec \mu` or just a single period  :math:`\mu_t`.

-  when a policymaker chooses, either at time :math:`0` or at times
   :math:`t \geq 0`.

-  what a policymaker assumes about how its choice of :math:`\mu_t`
   affects private agents' expectations about earlier and later
   inflation rates.

In two of our models, a single policymaker  chooses a sequence
:math:`\{\mu_t\}_{t=0}^\infty` once and for all, taking into account how
:math:`\mu_t` affects household one-period utilities at dates :math:`s = 0, 1, \ldots, t-1`

-  these two models  thus employ a  **Ramsey** or **Stackelberg** timing protocol.

In two other models, there is a sequence of policymakers, each of whom
sets :math:`\mu_t` at one :math:`t` only

- Each such policymaker ignores  effects that its choice of :math:`\mu_t` has on household one-period utilities at dates :math:`s = 0, 1, \ldots, t-1`.

The four models differ with respect to timing protocols, constraints on
government choices, and government policymakers' beliefs about how their
decisions affect private agents' beliefs about future government
decisions.

The models are

-  A single Ramsey planner chooses a sequence
   :math:`\{\mu_t\}_{t=0}^\infty` once and for all at time :math:`0`.

-  A single Ramsey planner chooses a sequence
   :math:`\{\mu_t\}_{t=0}^\infty` once and for all at time :math:`0`
   subject to the constraint that :math:`\mu_t = \mu` for all
   :math:`t \geq 0`.

-  A sequence of separate policymakers chooses :math:`\mu_t` for :math:`t =0, 1, 2, \ldots`

    -  a time :math:`t` policymaker chooses :math:`\mu_t` only and forecasts that future government decisions are unaffected by its choice.

-  A sequence of separate policymakers chooses :math:`\mu_t` for :math:`t =0, 1, 2, \ldots`

     - a time :math:`t` policymaker chooses  only :math:`\mu_t` but believes that its choice of :math:`\mu_t`  shapes private agents' beliefs about  future rates of money creation and inflation, and through them, future government actions.

A Ramsey Planner
===================

First, we consider a Ramsey planner that  chooses
:math:`\{\mu_t, \theta_t\}_{t=0}^\infty` to maximize :eq:`eq_old7`
subject to the law of motion :eq:`eq_old4`.

We can split this problem into two stages, as in :doc:`Stackelberg problems<dyn_stack>` and  :cite:`Ljungqvist2012` Chapter 19.

In the first stage, we take the initial inflation rate :math:`\theta_0` as given,
and then solve the resulting LQ dynamic programming problem.

In the second stage, we maximize over the initial inflation rate :math:`\theta_0`.

Define a feasible set of
:math:`(\overrightarrow x_1, \overrightarrow \mu_0)` sequences:

.. math::

  \Omega(x_0) = \left \lbrace ( \overrightarrow x_1, \overrightarrow \mu_0) : x_{t+1}
  = A x_t + B \mu_t \: , \: \forall t \geq 0 \right \rbrace

Subproblem 1
-------------

The value function

.. math::
  J(x_0) = \max_{(\overrightarrow x_1, \overrightarrow \mu_0) \in \Omega(x_0)}
  \sum_{t=0}^\infty \beta^t r(x_t,\mu_t)

satisfies the Bellman equation

.. math::
  J(x) = \max_{\mu,x'}\{-r(x,\mu) + \beta J(x')\}

subject to:

.. math::
  x' = Ax + B\mu

As in :doc:`Stackelberg problems<dyn_stack>`, we map this problem into a linear-quadratic control problem and then carefully use the
optimal value function associated with it.

Guessing that :math:`J(x) = - x'Px` and substituting into the Bellman
equation gives rise to the algebraic matrix Riccati equation:

.. math::  P = R + \beta A'PA - \beta^2 A'PB(Q + \beta B'PB)^{-1} B'PA

and the optimal decision rule

.. math::  \mu_t = - F x_t

where

.. math::  F = \beta (Q + \beta B'PB)^{-1} B'PA

The QuantEcon `LQ <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lqcontrol.py>`_ class solves for :math:`F` and :math:`P` given inputs
:math:`Q, R, A, B`, and :math:`\beta`.

Subproblem 2
-------------

The value of the Ramsey problem is

.. math::  V = \max_{x_0} J(x_0)

The value function

.. math:: J(x_0) = -\begin{bmatrix} 1 & \theta_0 \end{bmatrix} \begin{bmatrix} P_{11} & P_{12} \\ P_{21} & P_{22} \end{bmatrix} \begin{bmatrix} 1 \\ \theta_0 \end{bmatrix} = -P_{11} - 2 P_{21} \theta_0 - P_{22} \theta_0^2

Maximizing this with respect to :math:`\theta_0` yields the FOC:

.. math::  - 2 P_{21} - 2 P_{22} \theta_0 =0

which implies

.. math::  \theta_0^* = - \frac{P_{21}}{P_{22}}

Representation of Ramsey Plan
------------------------------

The preceding calculations indicate that we can represent a Ramsey plan
:math:`\vec \mu` recursively with the following system created in the spirit of Chang :cite:`chang1998credible`:

.. math::
    :label: eq_old9

    \begin{aligned}
    \theta_0 & = \theta_0^* \\
    \mu_t &  = b_0 + b_1 \theta_t \\
    \theta_{t+1} & = d_0 + d_1 \theta_t
    \end{aligned}

To interpret this system, think of the  sequence
:math:`\{\theta_t\}_{t=0}^\infty` as a sequence of
synthetic **promised inflation rates** that are just computational devices for
generating a sequence :math:`\vec\mu` of money growth rates that are to
be substituted into equation :eq:`eq_old3` to form actual rates of inflation.

It can be verified that if we substitute a plan
:math:`\vec \mu = \{\mu_t\}_{t=0}^\infty` that satisfies these equations
into equation :eq:`eq_old3`, we obtain the same sequence :math:`\vec \theta`
generated by the system :eq:`eq_old9`.

(Here an application of the **Big :math:`K`, little :math:`k`** trick could once again be enlightening)

Thus, our construction of a Ramsey plan guarantees that **promised
inflation** equals **actual inflation**.

### Multiple roles of :math:`\theta_t`

The inflation rate :math:`\theta_t` that appears in the system :eq:`eq_old9` and
equation :eq:`eq_old3` plays three roles simultaneously:

-  In equation :eq:`eq_old3`, :math:`\theta_t` is the actual rate of inflation
   between :math:`t` and :math:`t+1`.

-  In equation  :eq:`eq_old2` and :eq:`eq_old3`, :math:`\theta_t` is also the public's
   expected rate of inflation between :math:`t` and :math:`t+1`.

-  In system :eq:`eq_old9`, :math:`\theta_t` is a promised rate of inflation
   chosen by the Ramsey planner at time :math:`0`.



Time Inconsistency
-------------------

As discussed in :doc:`Stackelberg problems<dyn_stack>` and :doc:`Optimal taxation with state-contingent debt<opt_tax_recur>`, a continuation Ramsey plan is not a Ramsey plan.

This is a concise way of characterizing the time inconsistency of a Ramsey plan.

The time inconsistency of a Ramsey plan has motivated other models of government decision making
that alter either

- the timing protocol and/or

- assumptions about how government decision makers think their decisions affect private agents' beliefs about future government decisions

A Constrained-to-a-Constant-Growth-Rate Ramsey Government
==========================================================

We now consider the following peculiar model of optimal government behavior.

We have created this model in order to highlight an aspect of an optimal government policy associated with its time inconsistency,
namely, the feature that optimal settings of the  policy instrument vary over time.

Instead of allowing the Ramsey government to choose different settings of its instrument at different moments, we now assume that
at time :math:`0`, a Ramsey  government at time :math:`0` once and for all  chooses a **constant** sequence
:math:`\mu_t = \check \mu` for all :math:`t \geq 0` to maximize

.. math::  U(-\alpha \check \mu) - \frac{c}{2} \check \mu^2

Here we have imposed the perfect foresight outcome implied by equation :eq:`eq_old2` that
:math:`\theta_t = \check \mu` when the government chooses a constant
:math:`\mu` for all :math:`t \geq 0`.

With the quadratic form :eq:`eq_old5` for the utility function :math:`U`, the
maximizing :math:`\bar \mu` is

.. math::  \check \mu = - \frac{\alpha a_1}{\alpha^2 a_2 + c }

**Summary:** We have  introduced the constrained-to-a-constant :math:`\mu`
government in order to highlight  time-variation of
:math:`\mu_t` as a telltale sign of  time inconsistency of a Ramsey plan.


Markov Perfect Governments
===========================

We now  change the timing protocol by considering a sequence of
government policymakers, the time :math:`t` representative of which
chooses :math:`\mu_t` and expects all future governments to set
:math:`\mu_{t+j} = \bar \mu`.

This assumption mirrors an assumption made in a different setting  :doc:`Markov Perfect Equilibrium<markov_perf>`.



Further, a government  policymaker at :math:`t` believes that :math:`\bar \mu` is
unaffected by its choice of :math:`\mu_t`.

The time :math:`t` rate of inflation is then:

.. math::  \theta_t = \frac{\alpha}{1+\alpha} \bar \mu + \frac{1}{1+\alpha} \mu_t

The time :math:`t` government policymaker then chooses :math:`\mu_t` to
maximize:

.. math::  W = U(-\alpha \theta_t) - \frac{c}{2} \mu_t^2 + \beta V(\bar \mu)

where :math:`V(\bar \mu)` is the time :math:`0` value :math:`v_0` of
recursion :eq:`eq_old8` under a money supply growth rate that is forever constant
at :math:`\bar \mu`.

Substituting for :math:`U` and :math:`\theta_t` gives:

.. math::  W = a_0 + a_1(-\frac{\alpha^2}{1+\alpha} \bar \mu - \frac{\alpha}{1+\alpha} \mu_t) - \frac{a_2}{2}((-\frac{\alpha^2}{1+\alpha} \bar \mu - \frac{\alpha}{1+\alpha} \mu_t)^2 - \frac{c}{2} \mu_t^2 + \beta V(\bar \mu)

The first-order necessary condition for :math:`\mu_t` is then:

.. math::  - \frac{\alpha}{1+\alpha} a_1 - a_2(-\frac{\alpha^2}{1+\alpha} \bar \mu - \frac{\alpha}{1+\alpha} \mu_t)(- \frac{\alpha}{1+\alpha}) - c \mu_t = 0

Rearranging we get:

.. math::  \mu_t = \frac{- a_1}{\frac{1+\alpha}{\alpha}c + \frac{\alpha}{1+\alpha}a_2} - \frac{\alpha^2 a_2}{\left[ \frac{1+\alpha}{\alpha}c + \frac{\alpha}{1+\alpha} a_2 \right] (1+\alpha)}\bar \mu

A **Markov Perfect Equilibrium** (MPE) outcome sets
:math:`\mu_t = \bar \mu`:

.. math::  \mu_t = \bar \mu = \frac{-a_1}{\frac{1+\alpha}{\alpha}c + \frac{\alpha}{1+\alpha} a_2 + \frac{\alpha^2}{1+\alpha} a_2}

In light of results  presented in the previous section, this can be
simplified to:

.. math::  \bar \mu = - \frac{\alpha a_1}{\alpha^2 a_2 + (1+\alpha)c}



Equilibrium Outcomes for Three Models of Government Policy Making
==================================================================

Below we compute sequences :math:`\{ \theta_t,\mu_t \}` under a Ramsey
plan and compare these with the constant levels of :math:`\theta` and
:math:`\mu` in a) a Markov Perfect Equilibrium, and b) a Ramsey plan
in which the planner is restricted to choose :math:`\mu_t = \check\mu`
for all :math:`t \geq 0`.

We denote the Ramsey sequence as :math:`\theta^R, \mu^R` and the MPE
values as :math:`\theta^{MPE}, \mu^{MPE}`.

The bliss level of inflation is denoted by :math:`\theta^*`.


First, we will create a class `ChangLQ` that solves the models and stores their values

.. code-block:: python3

    class ChangLQ:
        """
        Class to solve LQ Chang model
        """
        def __init__(self, α, α0, α1, α2, c, T=1000, θ_n=200):

            # Record parameters
            self.α, self.α0, self.α1 = α, α0, α1
            self.α2, self.c, self.T, self.θ_n = α2, c, T, θ_n

            # Create β using "Poor Man's Friedman Rule"
            self.β = np.exp(-α1 / (α * α2))

            # Solve the Ramsey Problem #

            # LQ Matrices
            R = -np.array([[α0,            -α1 * α / 2],
                           [-α1 * α/2, -α2 * α**2 / 2]])
            Q = -np.array([[-c / 2]])
            A = np.array([[1, 0], [0, (1 + α) / α]])
            B = np.array([[0], [-1 / α]])

            # Solve LQ Problem (Subproblem 1)
            lq = LQ(Q, R, A, B, beta=self.β)
            self.P, self.F, self.d = lq.stationary_values()

            # Solve Subproblem 2
            self.θ_R = -self.P[0, 1] / self.P[1, 1]

            # Find bliss level of θ
            self.θ_B = np.log(self.β)

            # Solve the Markov Perfect Equilibrium
            self.μ_MPE = -α1 / ((1 + α) / α * c + α / (1 + α)
                          * α2 + α**2 / (1 + α) * α2)
            self.θ_MPE = self.μ_MPE
            self.μ_check = -α * α1 / (α2 * α**2 + c)

            # Calculate value under MPE and Check economy
            self.J_MPE  = (α0 + α1 * (-α * self.μ_MPE) - α2 / 2
                          * (-α * self.μ_MPE)**2 - c/2 * self.μ_MPE**2) / (1 - self.β)
            self.J_check = (α0 + α1 * (-α * self.μ_check) - α2/2
                            * (-α * self.μ_check)**2 - c / 2 * self.μ_check**2) \
                            / (1 - self.β)

            # Simulate Ramsey plan for large number of periods
            θ_series = np.vstack((np.ones((1, T)), np.zeros((1, T))))
            μ_series = np.zeros(T)
            J_series = np.zeros(T)
            θ_series[1, 0] = self.θ_R
            μ_series[0] = -self.F.dot(θ_series[:, 0])
            J_series[0] = -θ_series[:, 0] @ self.P @ θ_series[:, 0].T
            for i in range(1, T):
                θ_series[:, i] = (A - B @ self.F) @ θ_series[:, i-1]
                μ_series[i] = -self.F @ θ_series[:, i]
                J_series[i] = -θ_series[:, i] @ self.P @ θ_series[:, i].T

            self.J_series = J_series
            self.μ_series = μ_series
            self.θ_series = θ_series

            # Find the range of θ in Ramsey plan
            θ_LB = min(θ_series[1, :])
            θ_LB = min(θ_LB, self.θ_B)
            θ_UB = max(θ_series[1, :])
            θ_UB = max(θ_UB, self.θ_MPE)
            θ_range = θ_UB - θ_LB
            self.θ_LB = θ_LB - 0.05 * θ_range
            self.θ_UB = θ_UB + 0.05 * θ_range
            self.θ_range = θ_range

            # Find value function and policy functions over range of θ
            θ_space = np.linspace(self.θ_LB, self.θ_UB, 200)
            J_space = np.zeros(200)
            check_space = np.zeros(200)
            μ_space = np.zeros(200)
            θ_prime = np.zeros(200)
            for i in range(200):
                J_space[i] = - np.array((1, θ_space[i])) \
                            @ self.P @ np.array((1, θ_space[i])).T
                μ_space[i] = - self.F @ np.array((1, θ_space[i]))
                x_prime = (A - B @ self.F) @ np.array((1, θ_space[i]))
                θ_prime[i] = x_prime[1]
                check_space[i] = (α0 + α1 * (-α * θ_space[i]) -
                α2/2 * (-α * θ_space[i])**2 - c/2 * θ_space[i]**2) / (1 - self.β)

            J_LB = min(J_space)
            J_UB = max(J_space)
            J_range = J_UB - J_LB
            self.J_LB = J_LB - 0.05 * J_range
            self.J_UB = J_UB + 0.05 * J_range
            self.J_range = J_range
            self.J_space = J_space
            self.θ_space = θ_space
            self.μ_space = μ_space
            self.θ_prime = θ_prime
            self.check_space = check_space


We will create an instance of `ChangLQ` with the following parameters

.. code-block:: python3

    clq = ChangLQ(α=1, α0=1, α1=0.5, α2=3, c=2)
    clq.β

The following code generates a figure that plots the value function from the Ramsey Planner's
problem, which is maximized at :math:`\theta^R_0`.

The figure also shows the limiting value :math:`\theta_\infty^R` to which  the inflation rate :math:`\theta_t` converges under the Ramsey plan and compares it to the MPE value and the bliss value.

.. code-block:: python3

    def plot_value_function(clq):
        """
        Method to plot the value function over the relevant range of θ

        Here clq is an instance of ChangLQ

        """
        fig, ax = plt.subplots()

        ax.set_xlim([clq.θ_LB, clq.θ_UB])
        ax.set_ylim([clq.J_LB, clq.J_UB])

        # Plot value function
        ax.plot(clq.θ_space, clq.J_space, lw=2)
        plt.xlabel(r"$\theta$", fontsize=18)
        plt.ylabel(r"$J(\theta)$", fontsize=18)

        t1 = clq.θ_space[np.argmax(clq.J_space)]
        tR = clq.θ_series[1, -1]
        θ_points = [t1, tR, clq.θ_B, clq.θ_MPE]
        labels = [r"$\theta_0^R$", r"$\theta_\infty^R$",
                  r"$\theta^*$", r"$\theta^{MPE}$"]

        # Add points for θs
        for θ, label in zip(θ_points, labels):
            ax.scatter(θ, clq.J_LB + 0.02 * clq.J_range, 60, 'black', 'v')
            ax.annotate(label,
                        xy=(θ, clq.J_LB + 0.01 * clq.J_range),
                        xytext=(θ - 0.01 * clq.θ_range,
                        clq.J_LB + 0.08 * clq.J_range),
                        fontsize=18)
        plt.tight_layout()
        plt.show()

    plot_value_function(clq)


The next code generates a figure that plots the value function from the Ramsey Planner's
problem as well as that for a Ramsey planner that  must choose a constant
:math:`\mu` (that in turn  equals an  implied constant :math:`\theta`).

.. code-block:: python3

    def compare_ramsey_check(clq):
        """
        Method to compare values of Ramsey and Check

        Here clq is an instance of ChangLQ
        """
        fig, ax = plt.subplots()
        check_min = min(clq.check_space)
        check_max = max(clq.check_space)
        check_range = check_max - check_min
        check_LB = check_min - 0.05 * check_range
        check_UB = check_max + 0.05 * check_range
        ax.set_xlim([clq.θ_LB, clq.θ_UB])
        ax.set_ylim([check_LB, check_UB])
        ax.plot(clq.θ_space, clq.J_space, lw=2, label=r"$J(\theta)$")

        plt.xlabel(r"$\theta$", fontsize=18)
        ax.plot(clq.θ_space, clq.check_space,
                lw=2, label=r"$V^\check(\theta)$")
        plt.legend(fontsize=14, loc='upper left')

        θ_points = [clq.θ_space[np.argmax(clq.J_space)],
                        clq.μ_check]
        labels = [r"$\theta_0^R$", r"$\theta^\check$"]

        for θ, label in zip(θ_points, labels):
            ax.scatter(θ, check_LB + 0.02 * check_range, 60, 'k', 'v')
            ax.annotate(label,
                        xy=(θ, check_LB + 0.01 * check_range),
                        xytext=(θ - 0.02 * check_range,
                                check_LB + 0.08 * check_range),
                        fontsize=18)
        plt.tight_layout()
        plt.show()

    compare_ramsey_check(clq)


The next code generates  figures that plot the policy functions for a continuation Ramsey
planner.

The left figure shows the choice of :math:`\theta'` chosen by a
continuation Ramsey planner who inherits :math:`\theta`.

The right figure plots a continuation Ramsey planner's choice of
:math:`\mu` as a function of an inherited :math:`\theta`.

.. code-block:: python3

    def plot_policy_functions(clq):
        """
        Method to plot the policy functions over the relevant range of θ

        Here clq is an instance of ChangLQ
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        labels = [r"$\theta_0^R$", r"$\theta_\infty^R$"]

        ax = axes[0]
        ax.set_ylim([clq.θ_LB, clq.θ_UB])
        ax.plot(clq.θ_space, clq.θ_prime,
                label=r"$\theta'(\theta)$", lw=2)
        x = np.linspace(clq.θ_LB, clq.θ_UB, 5)
        ax.plot(x, x, 'k--', lw=2, alpha=0.7)
        ax.set_ylabel(r"$\theta'$", fontsize=18)

        θ_points = [clq.θ_space[np.argmax(clq.J_space)],
                        clq.θ_series[1, -1]]

        for θ, label in zip(θ_points, labels):
            ax.scatter(θ, clq.θ_LB + 0.02 * clq.θ_range, 60, 'k', 'v')
            ax.annotate(label,
                        xy=(θ, clq.θ_LB + 0.01 * clq.θ_range),
                        xytext=(θ - 0.02 * clq.θ_range,
                                clq.θ_LB + 0.08 * clq.θ_range),
                        fontsize=18)

        ax = axes[1]
        μ_min = min(clq.μ_space)
        μ_max = max(clq.μ_space)
        μ_range = μ_max - μ_min
        ax.set_ylim([μ_min - 0.05 * μ_range, μ_max + 0.05 * μ_range])
        ax.plot(clq.θ_space, clq.μ_space, lw=2)
        ax.set_ylabel(r"$\mu(\theta)$", fontsize=18)

        for ax in axes:
            ax.set_xlabel(r"$\theta$", fontsize=18)
            ax.set_xlim([clq.θ_LB, clq.θ_UB])

        for θ, label in zip(θ_points, labels):
            ax.scatter(θ, μ_min - 0.03 * μ_range, 60, 'black', 'v')
            ax.annotate(label, xy=(θ, μ_min - 0.03 * μ_range),
                        xytext=(θ - 0.02 * clq.θ_range,
                                μ_min + 0.03 * μ_range),
                        fontsize=18)
        plt.tight_layout()
        plt.show()

    plot_policy_functions(clq)


The following code generates a figure that plots sequences of :math:`\mu` and :math:`\theta`
in the Ramsey plan and compares these to the constant levels in a MPE
and in a Ramsey plan with a government restricted to set :math:`\mu_t`
to a constant for all :math:`t`.

.. code-block:: python3

    def plot_ramsey_MPE(clq, T=15):
        """
        Method to plot Ramsey plan against Markov Perfect Equilibrium

        Here clq is an instance of ChangLQ
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        plots = [clq.θ_series[1, 0:T], clq.μ_series[0:T]]
        MPEs = [clq.θ_MPE, clq.μ_MPE]
        labels = [r"\theta", r"\mu"]

        axes[0].hlines(clq.θ_B, 0, T-1, 'r', label=r"$\theta^*$")

        for ax, plot, MPE, label in zip(axes, plots, MPEs, labels):
            ax.plot(plot, label=r"$" + label + "^R$")
            ax.hlines(MPE, 0, T-1, 'orange', label=r"$" + label + "^{MPE}$")
            ax.hlines(clq.μ_check, 0, T, 'g', label=r"$" + label + "^\check$")
            ax.set_xlabel(r"$t$", fontsize=16)
            ax.set_ylabel(r"$" + label + "_t$", fontsize=18)
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    plot_ramsey_MPE(clq)


Time Inconsistency of Ramsey Plan
-----------------------------------

The variation over time in :math:`\vec \mu` chosen by the Ramsey planner
is a symptom of time inconsistency.

-  The Ramsey planner reaps immediate benefits from promising lower
   inflation later to be achieved by costly distorting taxes.

-  These benefits are intermediated by reductions in expected inflation
   that precede the  reductions in money creation rates that rationalize them, as indicated by
   equation :eq:`eq_old3`.

-  A government authority offered the opportunity to ignore effects on
   past utilities and to reoptimize at date :math:`t \geq 1` would, if allowed, want
   to deviate from a Ramsey plan.

**Note:** A modified Ramsey plan constructed under the restriction that
:math:`\mu_t` must be constant over time is time consistent (see
:math:`\check \mu` and :math:`\check \theta` in the above graphs).

Meaning of Time Inconsistency
------------------------------

In settings in which governments actually choose sequentially, many economists
regard a time inconsistent plan implausible because of the incentives to
deviate that occur along the plan.

A way to summarize this *defect* in a Ramsey plan is to say that it
is not credible because there  endure incentives for policymakers
to deviate from it.

For that reason, the Markov perfect equilibrium concept attracts many
economists.

* A Markov perfect equilibrium plan is constructed to insure that government policymakers who choose sequentially do not want to deviate from it.

The *no incentive to deviate from the plan* property is what makes the Markov perfect equilibrium concept attractive.

Ramsey Plan Strikes Back
--------------------------

Research by Abreu :cite:`Abreu`,  Chari and Kehoe :cite:`chari1990sustainable`
:cite:`stokey1989reputation`, and Stokey :cite:`Stokey1991` discovered conditions under which a Ramsey plan can be rescued from the complaint that it is not credible.

They  accomplished this by expanding the
description of a plan to include expectations about **adverse consequences** of deviating from
it that can serve to deter deviations.

We turn to such theories of **sustainable plans** next.

A Fourth Model of Government Decision Making
=============================================

This is a model in which

-  The government chooses :math:`\{\mu_t\}_{t=0}^\infty` not once and
   for all at :math:`t=0` but chooses to set :math:`\mu_t` at time :math:`t`, not before.

-  private agents' forecasts of
   :math:`\{\mu_{t+j+1}, \theta_{t+j+1}\}_{j=0}^\infty` respond to
   whether the government at :math:`t` **confirms** or **disappoints**
   their forecasts of :math:`\mu_t` brought into period :math:`t` from
   period :math:`t-1`.

-  the government at each time :math:`t` understands how private agents'
   forecasts will respond to its choice of :math:`\mu_t`.

-  at each :math:`t`, the government chooses :math:`\mu_t` to maximize
   a continuation discounted utility of a representative household.



A Theory of Government Decision Making
---------------------------------------


:math:`\vec \mu` is chosen by a sequence of government
decision makers, one for each :math:`t \geq 0`.

We assume the following within-period and between-period timing protocol
for each :math:`t \geq 0`:

-  at time :math:`t-1`, private agents expect  that the government will set
   :math:`\mu_t = \tilde \mu_t`, and more generally that it will set
   :math:`\mu_{t+j} = \tilde \mu_{t+j}` for all :math:`j \geq 0`.

-  Those forecasts determine a
   :math:`\theta_t = \tilde \theta_t` and an associated log
   of real balances :math:`m_t - p_t = -\alpha\tilde \theta_t` at
   :math:`t`.

-  Given those expectations and the associated :math:`\theta_t`, at
   :math:`t` a government is free to set :math:`\mu_t \in {\bf R}`.

-  If the government at :math:`t` **confirms** private agents'
   expectations by setting :math:`\mu_t = \tilde \mu_t` at time
   :math:`t`, private agents expect the continuation government policy
   :math:`\{\tilde \mu_{t+j+1}\}_{j=0}^\infty` and therefore bring
   expectation :math:`\tilde \theta_{t+1}` into period :math:`t+1`.

-  If the government at :math:`t` **disappoints** private agents by setting
   :math:`\mu_t \neq \tilde \mu_t`, private agents expect
   :math:`\{\mu^A_j\}_{j=0}^\infty` as the
   continuation policy for :math:`t+1`, i.e.,
   :math:`\{\mu_{t+j+1}\} = \{\mu_j^A \}_{j=0}^\infty` and therefore
   expect :math:`\theta_0^A` for :math:`t+1`. Here :math:`\vec \mu^A = \{\mu_j^A \}_{j=0}^\infty` is
   an alternative government plan to be described below.

Temptation to Deviate from Plan
--------------------------------

The government's one-period return function :math:`s(\theta,\mu)`
described in equation :eq:`eq_old6` above has the property that for all
:math:`\theta`

.. math::  s(\theta, 0 ) \geq s(\theta, \mu) \quad

This inequality implies that whenever the policy calls for the
government to set :math:`\mu \neq 0`, the government could raise its
one-period return by setting :math:`\mu =0`.

Disappointing private sector expectations in that way would increase the
government's **current** payoff but would have adverse consequences for
**subsequent** government payoffs because the private sector would alter
its expectations about future settings of :math:`\mu`.

The **temporary** gain constitutes the government's temptation to
deviate from a plan.

If the government at :math:`t` is to resist the temptation to raise its
current payoff, it is only because it forecasts adverse  consequences that
its setting of :math:`\mu_t` would bring for subsequent government payoffs via  alterations  in the private sector's expectations.

Sustainable or Credible Plan
=============================

We call a plan :math:`\vec \mu` **sustainable** or **credible** if at
each :math:`t \geq 0` the government chooses to confirm private
agents' prior expectation of its setting for :math:`\mu_t`.

The government will choose to confirm prior expectations if the
long-term **loss** from disappointing private sector expectations --
coming from the government's understanding of the way the private sector
adjusts its  expectations in response to having its prior
expectations at :math:`t` disappointed -- outweigh the short-term
**gain** from disappointing those expectations.

The theory of sustainable or credible plans assumes throughout that private sector
expectations about what future governments will do are based on the
assumption that governments at times :math:`t \geq 0` will act to
maximize the continuation discounted utilities that describe those
governments' purposes.

This aspect of the theory means that credible plans come in **pairs**:

-  a credible (continuation) plan to be followed if the government at
   :math:`t` **confirms** private sector expectations

-  a credible plan to be followed if the government at :math:`t`
   **disappoints** private sector expectations

That credible plans come in pairs seems to bring an explosion of plans to keep track of

* each credible plan itself consists of two credible plans

* therefore, the number of plans underlying one plan is unbounded

But Dilip Abreu showed how to render manageable the number of plans that must be kept track of.

The key is an  object called a **self-enforcing** plan.


Abreu's Self-Enforcing Plan
----------------------------

A plan :math:`\vec \mu^A` is said to be **self-enforcing** if

-  the consequence of disappointing private agents' expectations at time
   :math:`j` is to **restart** the plan at time :math:`j+1`

-  that consequence is sufficiently adverse that it deters all
   deviations from the plan

More precisely, a government plan :math:`\vec \mu^A` is
**self-enforcing** if

.. math::
    :label: eq_old10

    \begin{aligned}
    v_j^A & = s(\theta^A_j, \mu^A_j) + \beta v_{j+1}^A \\
    & \geq s(\theta^A_j, 0 ) + \beta v_0^A \equiv v_j^{A,D}, \quad j \geq 0
    \end{aligned}

(Here it is useful to recall that setting :math:`\mu=0` is the maximizing choice for the government's one-period return function)

The first line tells the consequences of confirming private agents'
expectations, while the second line tells the consequences of
disappointing private agents' expectations.

A consequence of the definition is that a self-enforcing plan is
credible.

Self-enforcing plans can be used to construct other credible plans, including ones with better values.

A sufficient condition for a plan :math:`\vec \mu` to be **credible** or
**sustainable** is that

.. math::

    \begin{aligned}
    \tilde v_j & = s(\tilde \theta_j, \mu_j) + \beta \tilde v_{j+1} \\
    & \geq s(\tilde \theta_j, 0) + \beta v_0^A \quad \forall j \geq 0
    \end{aligned}

Abreu taught us that  key step in constructing a credible plan is first constructing a
self-enforcing plan that has a low time :math:`0` value.

The idea is to use the self-enforcing plan as a continuation plan whenever
the government's choice at time :math:`t` fails to confirm private
agents' expectation.

We shall use a construction featured in :cite:`Abreu` to construct a
self-enforcing plan with low time :math:`0` value.

Abreu Carrot-Stick Plan
-------------------------------------------

:cite:`Abreu` invented a way to create a self-enforcing plan with a low
initial value.

Imitating his idea, we can construct a self-enforcing plan
:math:`\vec \mu` with a low time :math:`0` value to the government by
insisting that future government decision makers set :math:`\mu_t` to a value yielding low
one-period utilities to the household for a long time, after which
government  decisions thereafter  yield high one-period utilities.

-  Low one-period utilities early are a **stick**

-  High one-period utilities later are a **carrot**

Consider a plan :math:`\vec \mu^A` that sets
:math:`\mu_t^A = \bar \mu` (a high positive
number) for :math:`T_A` periods, and then reverts to the Ramsey plan.

Denote this sequence by :math:`\{\mu_t^A\}_{t=0}^\infty`.

The sequence of inflation rates implied by this plan,
:math:`\{\theta_t^A\}_{t=0}^\infty`, can be calculated using:

.. math:: \theta_t^A = \frac{1}{1+\alpha} \sum_{j=0}^{\infty} \left(\frac{\alpha}{1+\alpha}\right)^j \mu^A_{t+j}

The value of :math:`\{\theta_t^A,\mu_t^A \}_{t=0}^\infty` is

.. math::  v^A_0 = \sum_{t=0}^{T_A-1} \beta^t s(\theta_t^A,\mu_t^A) +\beta^{T_A} J(\theta^R_0)

Example of Self-Enforcing Plan
---------------------------------

The following example implements an Abreu stick-and-carrot plan.

The government sets :math:`\mu_t^A = 0.1` for :math:`t=0, 1, \ldots, 9`
and then starts the **Ramsey plan**.

We have computed outcomes for this plan.

For this plan, we plot the :math:`\theta^A`, :math:`\mu^A` sequences as
well as the implied :math:`v^A` sequence.

Notice that because the government sets money supply growth high for 10
periods, inflation starts high.

Inflation gradually slowly declines immediately because people immediately expect the government to lower the money growth rate after period
:math:`10`.

From the 10th period onwards, the inflation rate :math:`\theta^A_t`
associated with this **Abreu plan** starts the Ramsey plan from its
beginning, i.e., :math:`\theta^A_{t+10} =\theta^R_t \ \ \forall t \geq 0`.

.. code-block:: python3

    def abreu_plan(clq, T=1000, T_A=10, μ_bar=0.1, T_Plot=20):

        # Append Ramsey μ series to stick μ series
        clq.μ_A = np.append(np.ones(T_A) * μ_bar, clq.μ_series[:-T_A])

        # Calculate implied stick θ series
        clq.θ_A = np.zeros(T)
        discount = np.zeros(T)
        for t in range(T):
            discount[t] = (clq.α / (1 + clq.α))**t
        for t in range(T):
            length = clq.μ_A[t:].shape[0]
            clq.θ_A[t] = 1 / (clq.α + 1) * sum(clq.μ_A[t:] * discount[0:length])

        # Calculate utility of stick plan
        U_A = np.zeros(T)
        for t in range(T):
            U_A[t] = clq.β**t * (clq.α0 + clq.α1 * (-clq.θ_A[t])
                     - clq.α2 / 2 * (-clq.θ_A[t])**2 - clq.c * clq.μ_A[t]**2)

        clq.V_A = np.zeros(T)
        for t in range(T):
            clq.V_A[t] = sum(U_A[t:] / clq.β**t)

        # Make sure Abreu plan is self-enforcing
        clq.V_dev = np.zeros(T_Plot)
        for t in range(T_Plot):
            clq.V_dev[t] = (clq.α0 + clq.α1 * (-clq.θ_A[t])
                            - clq.α2 / 2 * (-clq.θ_A[t])**2) \
                            + clq.β * clq.V_A[0]

        fig, axes = plt.subplots(3, 1, figsize=(8, 12))

        axes[2].plot(clq.V_dev[0:T_Plot], label="$V^{A, D}_t$", c="orange")

        plots = [clq.θ_A, clq.μ_A, clq.V_A]
        labels = [r"$\theta_t^A$", r"$\mu_t^A$", r"$V^A_t$"]

        for plot, ax, label in zip(plots, axes, labels):
            ax.plot(plot[0:T_Plot], label=label)
            ax.set(xlabel="$t$", ylabel=label)
            ax.legend()

        plt.tight_layout()
        plt.show()

    abreu_plan(clq)


To confirm that the plan :math:`\vec \mu^A` is **self-enforcing**,  we
plot an object that we call :math:`V_t^{A,D}`, defined in the second line of equation :eq:`eq_old10` above.

:math:`V_t^{A,D}` is the value at :math:`t` of deviating from the
self-enforcing plan :math:`\vec \mu^A` by setting :math:`\mu_t = 0` and
then restarting the plan at :math:`v^A_0` at :math:`t+1`.

Notice that  :math:`v_t^A > v_t^{A,D}`.

This confirms that :math:`\vec \mu^A` is a self-enforcing plan.

We can also verify the inequalities required for :math:`\vec \mu^A` to
be self-confirming numerically as follows

.. code-block:: python3

    np.all(clq.V_A[0:20] > clq.V_dev[0:20])


Given that plan :math:`\vec \mu^A` is self-enforcing, we can check that
the Ramsey plan :math:`\vec \mu^R` is sustainable by verifying that:

.. math::  v^R_t \geq s(\theta^R_t,0) + \beta v^A_0 , \quad \forall t \geq 0

.. code-block:: python3

    def check_ramsey(clq, T=1000):
        # Make sure Ramsey plan is sustainable
        R_dev = np.zeros(T)
        for t in range(T):
            R_dev[t] = (clq.α0 + clq.α1 * (-clq.θ_series[1, t])
                        - clq.α2 / 2 * (-clq.θ_series[1, t])**2) \
                        + clq.β * clq.V_A[0]

        return np.all(clq.J_series > R_dev)

    check_ramsey(clq)


Recursive Representation of a Sustainable Plan
-----------------------------------------------

We can represent a sustainable plan recursively by taking the
continuation value :math:`v_t` as a state variable.

We form the following 3-tuple of functions:

.. math::
    :label: eq_old11

    \begin{aligned}
    \hat \mu_t & = \nu_\mu(v_t) \\
    \theta_t & = \nu_\theta(v_t) \\
    v_{t+1} & = \nu_v(v_t, \mu_t )
    \end{aligned}

In addition to these equations, we need an initial value :math:`v_0` to
characterize a sustainable plan.

The first equation of :eq:`eq_old11` tells the recommended value of
:math:`\hat \mu_t` as a function of the promised value :math:`v_t`.

The second equation of :eq:`eq_old11`  tells the inflation rate as a function of
:math:`v_t`.

The third equation of :eq:`eq_old11`  updates the continuation value in a way that
depends on whether the government at :math:`t` confirms private agents'
expectations by setting :math:`\mu_t` equal to the recommended value
:math:`\hat \mu_t`, or whether it disappoints those expectations.

Comparison of Equilibrium Values
=================================

We have computed plans for

-  an ordinary (unrestricted) Ramsey planner who chooses a sequence
   :math:`\{\mu_t\}_{t=0}^\infty` at time :math:`0`

-  a Ramsey planner restricted to choose a constant :math:`\mu` for all
   :math:`t \geq 0`

-  a Markov perfect sequence of governments

Below we compare equilibrium time zero values for these three.

We confirm that the value delivered by the unrestricted Ramsey planner
exceeds the value delivered by the restricted Ramsey planner which in
turn exceeds the value delivered by the Markov perfect sequence of
governments.

.. code-block:: python3

    clq.J_series[0]



.. code-block:: python3

    clq.J_check



.. code-block:: python3

    clq.J_MPE

We have also computed **sustainable plans** for a government or sequence
of governments that choose sequentially.

These include

-  a **self-enforcing** plan that gives a low initial value :math:`v_0`.

-  a better plan -- possibly one that attains values associated with
   Ramsey plan -- that is not self-enforcing.


Note on Dynamic Programming Squared
====================================

The theory deployed  in this lecture is an application of what we  nickname **dynamic programming squared**.

The nickname refers to the fact that a value satisfying one Bellman equation is itself an argument in a second Bellman equation.

Thus, our models have involved two Bellman equations:

-  equation :eq:`eq_old1` expresses how :math:`\theta_t` depends on :math:`\mu_t`
   and :math:`\theta_{t+1}`

-  equation :eq:`eq_old4` expresses how value :math:`v_t` depends on
   :math:`(\mu_t, \theta_t)` and :math:`v_{t+1}`

A value :math:`\theta` from one Bellman equation appears as an argument of a second Bellman equation for another value :math:`v`.
