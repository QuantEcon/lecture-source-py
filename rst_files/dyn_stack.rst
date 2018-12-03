.. _dyn_stack:

.. include:: /_static/includes/lecture_howto_py.raw

******************************
Dynamic Stackelberg Problems
******************************

.. contents:: :depth: 2

Overview
==================

Previous lectures including :doc:`LQ dynamic programming <lqcontrol>`, :doc:`rational expectations equilibrium <rational_expectations>`, and :doc:`Markov perfect equilibrium <markov_perf>`  lectures have studied  decision problems that are recursive in what we can call "natural" state variables, such as

* stocks of capital (fiscal, financial and human)
* wealth
* information that helps forecast future prices and quantities that impinge on future payoffs

Optimal decision rules are functions of the natural state variables in problems that are recursive in the natural state variables

In this lecture, we  describe   problems that are not recursive in the natural state variables

Kydland and Prescott :cite:`KydlandPrescott1977`, :cite:`Prescott1977` and Calvo :cite:`Calvo1978` gave examples of such decision problems

These problems  have the following features

* Time :math:`t \geq 0` actions of decision makers called  followers depend on  time :math:`s \geq t` decisions of another decision maker called a Stackelberg leader

* At  time :math:`t=0`, the Stackelberg leader chooses his actions for all times :math:`s \geq 0`

* In choosing actions for all times at time :math:`0`, the  Stackelberg leader can be said to *commit to a plan*

* The Stackelberg leader has distinct optimal decision rules at time :math:`t=0`, on the one hand, and at times :math:`t \geq 1`, on the other hand

* The Stackelberg leader's decision rules for :math:`t=0` and :math:`t \geq 1` have distinct state variables

* Variables that encode *history dependence* appear in optimal decision rules of the  Stackelberg leader at times :math:`t \geq 1`

* These properties of the Stackelberg leader's  decision rules are symptoms of the *time inconsistency of optimal government plans*

An example of a time inconsistent optimal rule is that of a

* a large agent (e.g., a government) that confronts a competitive market composed of many small private agents, and in which

* private agents' decisions at each date are influenced by their *forecasts* of the large agent's future actions

The *rational expectations* equilibrium concept plays an essential role

A rational expectations restriction implies that when it chooses its future actions, the Stackelberg leader also chooses the followers' expectations about those actions

The Stackelberg leader understands and exploits that situation

In a rational expectations equilibrium,  the Stackelberg leader's time :math:`t` actions confirm private agents' forecasts of those actions

The requirement to confirm prior followers' forecasts puts constraints on the Stackelberg leader's  time :math:`t` decisions that prevent its problem from being recursive in natural state variables

These additional constraints make the Stackelberg leader's decision rule at :math:`t` depend on the entire history of the natural state variables from time :math:`0` to time :math:`t`

This lecture displays these principles within the tractable framework of linear quadratic problems

It is based on chapter 19 of :cite:`Ljungqvist2012`



The Stackelberg Problem
=======================

We use the optimal linear regulator (a.k.a. the linear-quadratic dynamic programming problem described in :doc:`LQ Dynamic Programming problems <lqcontrol>`)  to solve a linear quadratic version of what is known as a dynamic Stackelberg problem

For now we refer to the Stackelberg leader as the government and the Stackelberg follower as the representative agent or private sector

Soon we'll give an application with another interpretation of these two decision makers

Let :math:`z_t` be an :math:`n_z \times 1` vector of natural state variables

Let :math:`x_t` be an :math:`n_x \times 1` vector of endogenous forward-looking variables that are physically free to jump at :math:`t`

Let :math:`u_t` be a vector of government instruments

The :math:`z_t` vector is inherited physically from the past

But :math:`x_t` is inherited  as a consequence of decisions made by the Stackelberg planner at time :math:`t=0`

Included in :math:`x_t` might be prices and quantities that adjust instantaneously to clear markets at time :math:`t`

Let :math:`y_t = \begin{bmatrix} z_t \\ x_t \end{bmatrix}`

Define the government's one-period loss function [#f1]_

.. math::
    :label: target

    r(y, u)  =  y' R y  + u' Q u


Subject to an initial condition for :math:`z_0`, but not for :math:`x_0`, a government wants to maximize

.. math::
    :label: new1_dyn_stack

    -\sum_{t=0}^\infty \beta^t r(y_t, u_t)


The government makes policy in light of the model

.. math::
    :label: new2

    \begin{bmatrix} I & 0 \\ G_{21} & G_{22} \end{bmatrix}
    \begin{bmatrix}    z_{t+1} \\  x_{t+1} \end{bmatrix}
    = \begin{bmatrix}  \hat A_{11}  &  \hat A_{12} \\ \hat A_{21} & \hat A_{22}  \end{bmatrix} \begin{bmatrix}  z_t \\ x_t \end{bmatrix} + \hat B u_t


We assume that the matrix on the left is invertible, so that we can multiply both sides of :eq:`new2` by its inverse to obtain

.. NOTE: I omitted a footnote here.

.. math::
    :label: new3

    \begin{bmatrix}    z_{t+1} \\  x_{t+1} \end{bmatrix}
    = \begin{bmatrix}  A_{11}  &   A_{12} \\ A_{21} &  A_{22}  \end{bmatrix}
    \begin{bmatrix}  z_t \\ x_t \end{bmatrix} +  B u_t


or

.. math::
    :label: new30

    y_{t+1} = A y_t + B u_t



The private sector's behavior is summarized by the second block of equations of :eq:`new3` or :eq:`new30`

These equations typically include the first-order conditions of private agents' optimization problem (i.e., their Euler equations)

These Euler equations summarize the forward-looking aspect of private agents' behavior and express how their time :math:`t` decisions depend on government actions at times :math:`s \geq t`

When combined with a stability condition to be imposed below, the
Euler equations
summarize the private sector's best response to the
sequence of actions by the government

The government maximizes :eq:`new1_dyn_stack` by choosing sequences :math:`\{u_t, x_t, z_{t+1}\}_{t=0}^\infty` subject to :eq:`new30` and an initial condition for :math:`z_0`

Note that we have an initial condition for :math:`z_0` but  not for :math:`x_0`

:math:`x_0` is among the variables to be chosen at time :math:`0` by the Stackelberg leader


The government uses its understanding of the responses restricted by :eq:`new30` to manipulate private sector actions

To indicate the features of the Stackelberg leader's problem that make :math:`x_t` a vector of forward-looking
variables, write the second block of system :eq:`new2`
as

.. math::
    :label: eqn:xlawforward

    x_t =  \phi_1 z_t + \phi_2 z_{t+1} +  \phi_3 u_t + \phi_0 x_{t+1}


where :math:`\phi_0 = \hat A_{22}^{-1} G_{22}`

The models we study in this chapter typically satisfy


*Forward-Looking Stability Condition*
The eigenvalues of :math:`\phi_0`
are bounded in modulus by :math:`\beta^{-.5}`



This stability condition  makes equation  :eq:`eqn:xlawforward` explosive if solved 'backwards' but stable if solved 'forwards'

See the appendix of chapter 2 of :cite:`Ljungqvist2012`

So we solve equation :eq:`eqn:xlawforward` forward to get

.. math::
    :label: bell101

    x_t = \sum_{j=0}^\infty \phi_0^j \left[ \phi_1 z_{t+j} + \phi_2 z_{t+j+1} + \phi_3 u_{t+j} \right]


In choosing :math:`u_t` for :math:`t \geq 1` at time :math:`0`, the government  takes into account how future :math:`z` and :math:`u` affect earlier
:math:`x` through equation :eq:`bell101`

The lecture on :doc:`history dependent policies <hist_dep_policies>` analyzes an example about *Ramsey taxation* in which, as is typical in such problems, the last :math:`n_x` equations of :eq:`new3` or :eq:`new30` constitute
*implementability constraints* that are formed by the Euler equations of a competitive fringe or private sector


A :ref:`certainty equivalence principle <lq_cert_eq>` allows us to work with a nonstochastic model (see :doc:`LQ dynamic programming <lqcontrol>`)

That is, we would attain the same decision rule if we were to replace :math:`x_{t+1}` 
with the forecast :math:`E_t x_{t+1}` and to add a shock process :math:`C \epsilon_{t+1}` to the right side of :eq:`new30`, where :math:`\epsilon_{t+1}` is an IID random vector with mean  zero and identity covariance matrix

Let :math:`s^t` denote the history of any variable :math:`s` from :math:`0` to :math:`t`

:cite:`MillerSalmon1985`, :cite:`HansenEppleRoberds1985`, :cite:`PearlmanCurrieLevine1986`, 
:cite:`Sargent1987`, :cite:`Pearlman1992`, and others have all studied versions of the following problem:

.. CC: Also listed was  Miller and Salmon 1982.  I found a 1983 version of their same paper that was published in '85, but not another one, so I'm not sure what paper this referred to.

**Problem S:** The *Stackelberg problem* is to maximize :eq:`new1_dyn_stack` by choosing an 
:math:`x_0` and a sequence of decision rules, the time :math:`t` component of which 
maps a time :math:`t` history of the natural state :math:`z^t` into a 
time :math:`t` decision :math:`u_t` of the Stackelberg leader

The Stackelberg leader chooses this sequence of decision rules once and for all at time :math:`t=0`

Another way to say this is that he *commits* to this sequence of decision rules at time :math:`0`

The maximization is subject to a given initial condition for :math:`z_0`

But :math:`x_0` is among the objects to be chosen by the Stackelberg leader

The optimal decision rule is history dependent, meaning that :math:`u_t` depends 
not only on :math:`z_t` but at :math:`t \geq 1` also on lags of :math:`z`

History dependence has two sources: (a) the government's ability to commit [#f2]_ 
to a sequence of rules at time :math:`0` as in the lecture on 
:doc:`history dependent policies <hist_dep_policies>`, and (b) the forward-looking 
behavior of the private sector embedded in the second block of equations :eq:`new3` as exhibited by :eq:`bell101`



Solving the Stackelberg Problem
===============================

Some Basic Notation
--------------------


For any vector :math:`a_t`, define :math:`\vec a_t = [a_t, a_{t+1}, \ldots]`

Define a feasible set of :math:`(\vec y_1, \vec u_0)` sequences

.. math::

    \Omega(y_0) = \left\{ (\vec y_1, \vec u_0) :  y_{t+1} = A y_t + B u_t, \forall t \geq 0 \right\}


Note that in the definition of :math:`\Omega(y_0)`, :math:`y_0` is taken as given

Eventually, the  :math:`x_0` component of :math:`y_0` will be chosen, though it is taken as given in :math:`\Omega(y_0)`

Two Subproblems
----------------------

Once again we use backward induction

We express the Stackelberg problem in terms of **two subproblems**:

Subproblem 1 is solved by a **continuation Stackelberg leader** at each date :math:`t \geq 1`

Subproblem 2 is solved the **Stackelberg leader** at :math:`t=0`

Subproblem 1
^^^^^^^^^^^^^

.. math::
    :label: stacksub1

    v(y_0) = \max_{(\vec y_1, \vec u_0) \in \Omega(y_0)} - \sum_{t=0}^\infty \beta^t r(y_t, u_t)


Subproblem 2
^^^^^^^^^^^^^

.. math::
    :label: stacksub2

    w(z_0) = \max_{x_0} v(y_0)


Subproblem 1 takes the vector of forward-looking variables  :math:`x_0` as given

Subproblem 2 optimizes over  :math:`x_0`

The value function :math:`w(z_0)` tells the value of the Stackelberg plan as a 
function of the vector of natural state variables at time :math:`0`,
:math:`z_0`


Two Bellman equations
----------------------


We now describe Bellman equations for :math:`v(y)` and :math:`w(z_0)`

Subproblem 1
--------------

The value function :math:`v(y)` in  subproblem 1 satisfies the Bellman equation

.. math::
    :label: bell1_dyn_stack

    v(y) = \max_{u, y^*}  \left\{ - r(y,u) + \beta v(y^*) \right\}


where the maximization is subject to

.. math::
    :label: bell2_dyn_stack

    y^* = A y + B u


and  :math:`y^*` denotes next period's value

Substituting :math:`v(y) = - y'P y` into  Bellman equation :eq:`bell1_dyn_stack`  gives

.. math::

    -y' P y = {\rm max}_{  u, y^*} \left\{ -  y' R y -   u'Q     u - \beta y^{* \prime} P y^* \right\}


which as in  lecture :doc:`linear regulator <lqcontrol>` gives rise to the algebraic matrix Riccati equation

.. math::
    :label: bell3_dyn_stack

    P = R + \beta A' P A - \beta^2 A' P   B (  Q  + \beta   B' P   B)^{-1}   B' P A


and the optimal decision rule coefficient vector

.. math::
    :label: bell4_dyn_stack

    F = \beta(   Q + \beta   B' P   B)^{-1}  B' P A 


where the optimal decision rule is

.. math::
    :label: bell5_dyn_stack

    u_t = - F y_t


Subproblem 2
--------------

The value function :math:`v(y_0)` satisfies

.. math::
    :label: valuefny

    v(y_0) = - z_0 ' P_{11} z_0 - 2 x_0' P_{21} z_0 - x_0' P_{22} x_0


where

.. math::

    P =
    \left[
    \begin{array}{cc}
        P_{11} & P_{12} \\
        P_{21} & P_{22}
    \end{array}
    \right]


We find an optimal :math:`x_0` by equating to zero the gradient of :math:`v(y_0)` with respect to  :math:`x_0`:

.. math::

    -2 P_{21} z_0 - 2 P_{22} x_0 =0,


which implies that

.. math::
    :label:  king6x0

    x_0 = - P_{22}^{-1} P_{21} z_0

Summary
-------

We solve the Stackelberg problem by

* formulating a particular optimal linear regulator

* solving the associated matrix Riccati equation :eq:`bell3_dyn_stack` for :math:`P`

* computing :math:`F`

* then partitioning :math:`P` to obtain representation :eq:`king6x0`



Manifestation of time inconsistency
------------------------------------

We have seen that for :math:`t \geq 0` the optimal decision rule for the Stackelberg leader has the form

.. math::

    u_t = - F y_t


or

.. math::

    u_t  = f_{11}  z_{t} + f_{12} x_{t}


where for :math:`t \geq 1`, :math:`x_t` is effectively a state variable, 
albeit not a *natural* one,  inherited from the past

The means that for :math:`t \geq 1`, :math:`x_t` is  *not* a  function of 
:math:`z_t` only (though it is at :math:`t=0`) and that :math:`x_t` exerts an independent influence on :math:`u_t`


The situation is different at :math:`t=0`

For :math:`t=0`, the optimal choice of :math:`x_0 = - P_{22}^{-1} P_{21} z_0` described in equation :eq:`king6x0`   implies that

.. math::
    :label: vonzer3c

    u_0 = (f_{11} - f_{12}P_{22}^{-1} P_{2,1})  z_0


So for :math:`t=0`, :math:`u_0` is a linear function of the natural state variable :math:`z_0` only

But for :math:`t \geq 0`, :math:`x_t \neq - P_{22}^{-1} P_{21} z_t`

Nor does :math:`x_t` equal any other linear combination of :math:`z_t`  for :math:`t \geq 1`

This means that :math:`x_t` has an independent role in shaping :math:`u_t` for :math:`t \geq 1`

All of this means that  the Stackelberg leader's decision rule at :math:`t \geq 1` differs from its decision rule at :math:`t =0`

As indicated at the beginning of this lecture, this difference is a 
symptom of the *time inconsistency* of the optimal Stackelberg plan


Shadow prices
==============

The history dependence of the government's plan can be expressed in the 
dynamics of Lagrange multipliers :math:`\mu_x` on the last :math:`n_x` equations of :eq:`new2` or :eq:`new3`

These multipliers measure the cost today of honoring past government 
promises about current and future settings of :math:`u`

We shall soon show that as a result of optimally choosing :math:`x_0`, it is 
appropriate to initialize the multipliers to zero at time :math:`t=0`

This is true because at :math:`t=0`, there are no past promises about :math:`u` to honor

But the multipliers :math:`\mu_x` take nonzero values thereafter, reflecting 
future costs to the government of confirming the private sector's
earlier expectations about its time :math:`t` actions


From  the :doc:`linear regulator <lqcontrol>` lecture, the formula :math:`\mu_t = P y_t` for the vector of
shadow prices on the transition equations is

.. math::

    \mu_t =
    \left[\begin{array}{c}
    \mu_{zt} \\
    \mu_{xt}
    \end{array}
    \right]


The shadow price :math:`\mu_{xt}` on the forward-looking variables :math:`x_t` evidently equals

.. math::
    :label: eqnmux

    \mu_{xt} = P_{21} z_t + P_{22} x_t


So  :eq:`king6x0` is equivalent with

.. math::
    :label: mu0condition

    \mu_{x0} = 0 


A Large Firm With a Competitive Fringe
======================================

As an example, this section studies the equilibrium of an industry with a large firm that acts as a Stackelberg leader with respect to a competitive fringe

Sometimes the large firm is called ‘the monopolist' even though there are actually many firms in the industry

The industry produces a single nonstorable homogeneous good, the quantity of which is chosen in the previous period

One large firm produces :math:`Q_t` and a representative firm in a competitive fringe produces :math:`q_t`

The representative firm in the competitive fringe acts as a price taker and chooses sequentially

The large firm commits to a policy at time :math:`0`, taking into account its 
ability to manipulate the price sequence, both directly through the effects of its 
quantity choices on prices, and indirectly through the responses of the competitive fringe to its forecasts of prices [#f8]_

The costs of production are :math:`{\cal C}_t = e Q_t + .5 g Q_t^2+ .5 c (Q_{t+1} - Q_{t})^2` for 
the large firm and :math:`\sigma_t= d q_t + .5 h q_t^2 + .5 c (q_{t+1} - q_t)^2` for the 
competitive firm, where :math:`d>0, e >0, c>0, g >0, h>0` are cost parameters

There is a linear inverse demand curve

.. math::
    :label: oli1

    p_t = A_0 - A_1 (Q_t + \overline q_t) + v_t,


where :math:`A_0, A_1` are both positive and :math:`v_t` is a disturbance to demand governed by

.. math::
    :label: oli2

    v_{t+1}= \rho v_t + C_\epsilon \check \epsilon_{t+1}


where :math:`| \rho | < 1` and :math:`\check \epsilon_{t+1}` is an IID sequence 
of random variables with mean zero and variance :math:`1`

In :eq:`oli1`, :math:`\overline q_t` is equilibrium output of the representative competitive firm

In equilibrium, :math:`\overline q_t = q_t`, but we must distinguish between 
:math:`q_t` and :math:`\overline q_t` in posing the optimum problem of a competitive firm

The competitive fringe
----------------------

The representative competitive firm regards :math:`\{p_t\}_{t=0}^\infty` as an 
exogenous stochastic process and chooses an output plan to maximize

.. math::
    :label: oli3

    E_0 \sum_{t=0}^\infty \beta^t \left\{ p_t q_t - \sigma_t \right\}, \quad \beta \in(0,1)


subject to :math:`q_0` given, where :math:`E_t` is the mathematical expectation based on time :math:`t` information

Let :math:`i_t = q_{t+1} - q_t`

We regard :math:`i_t` as the representative firm's control at :math:`t`

The first-order conditions for maximizing :eq:`oli3` are

.. math::
    :label: oli4

    i_t =  E_t  \beta i_{t+1} -c^{-1} \beta h  q_{t+1} + c^{-1} \beta  E_t( p_{t+1} -d)


for :math:`t \geq 0`

We appeal to a :ref:`certainty equivalence principle <lq_cert_eq>` to justify working 
with a non-stochastic version of :eq:`oli4` formed by dropping the expectation 
operator and the random term :math:`\check \epsilon_{t+1}` from :eq:`oli2`

We use a method of :cite:`Sargent1979` and :cite:`Townsend1983` [#f9]_

We shift :eq:`oli1` forward one period, replace conditional expectations with 
realized values, use :eq:`oli1` to substitute for :math:`p_{t+1}` in :eq:`oli4`, 
and set :math:`q_t = \overline q_t` and :math:`i_t = \overline i_t` for all :math:`t\geq 0` to get


.. math::
    :label: oli5

    \overline i_t = \beta \overline i_{t+1}  - c^{-1} \beta h \overline q_{t+1} + c^{-1} \beta (A_0-d) - c^{-1} \beta    A_1 \overline q_{t+1} -  c^{-1} \beta    A_1 Q_{t+1} + c^{-1} \beta    v_{t+1}


Given sufficiently stable sequences :math:`\{Q_t, v_t\}`, we could solve :eq:`oli5` and :math:`\overline i_t = \overline q_{t+1} - \overline q_t` to express the competitive fringe's output sequence as a function of the (tail of the) monopolist's output sequence

(This would be a version of representation :eq:`bell101`)

It is this feature that makes the monopolist's problem fail to be recursive 
in the natural state variables :math:`\overline q, Q`

The monopolist arrives at period :math:`t >0` facing the constraint that it 
must confirm the expectations about its time :math:`t` decision upon which 
the competitive fringe based its decisions at dates before :math:`t`

The monopolist's problem
------------------------

The monopolist views the sequence of the competitive firm's  Euler equations as constraints on its own opportunities

They are *implementability constraints* on the monopolist's choices

Including the implementability constraints, we can represent the constraints in terms of the transition law facing the monopolist:

.. math::
    :label: oli6

    \begin{aligned}
    \begin{bmatrix}
    1 & 0 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 \\
    A_0 -d & 1 & - A_1 & - A_1 -h & c \end{bmatrix}
    \begin{bmatrix}  1 \\ v_{t+1} \\ Q_{t+1} \\ \overline q_{t+1} \\ \overline i_{t+1} \end{bmatrix}
    & = \begin{bmatrix}
    1 & 0 & 0 & 0 & 0 \\
    0 & \rho & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 1 & 1 \\
    0 & 0 & 0 & 0 & {c\over \beta} \end{bmatrix}
    \begin{bmatrix}  1 \\ v_t \\ Q_t \\ \overline q_t \\ \overline i_t \end{bmatrix} + 
    \begin{bmatrix}  0 \\ 0 \\ 1 \\ 0 \\ 0  \end{bmatrix} u_t
    \end{aligned}


where :math:`u_t = Q_{t+1} - Q_t` is the control of the monopolist at time :math:`t`

The last row portrays the implementability constraints :eq:`oli5`

Represent :eq:`oli6` as

.. math::
    :label: oli6a

    y_{t+1} = A y_t + B u_t


Although we have included the competitive fringe's choice variable 
:math:`\overline i_t` as a component of the "state" :math:`y_t` in the 
monopolist's transition law :eq:`oli6a`, :math:`\overline i_t` is actually a "jump" variable

Nevertheless, the analysis above implies that the solution of the large firm's 
problem is encoded in the Riccati equation associated with :eq:`oli6a` as the transition law

Let's decode it

To match our general setup, we partition :math:`y_t` as :math:`y_t' = \begin{bmatrix} z_t' &  x_t' \end{bmatrix}` where :math:`z_t' = \begin{bmatrix}  1 & v_t & Q_t & \overline q_t  \end{bmatrix}` and :math:`x_t = \overline i_t`

The monopolist's problem is

.. math::

    \max_{\{u_t, p_{t+1}, Q_{t+1}, \overline q_{t+1}, \overline i_t\}}
    \sum_{t=0}^\infty \beta^t \left\{ p_t Q_t  - {\cal C}_t \right\}


subject to the given initial condition for :math:`z_0`, equations :eq:`oli1` and 
:eq:`oli5` and :math:`\overline i_t = \overline q_{t+1} - \overline q_t`, as well 
as the laws of motion of the natural state variables :math:`z`

Notice that the monopolist in effect chooses the price sequence, as well as the 
quantity sequence of the competitive fringe, albeit subject to the restrictions imposed by the behavior of consumers, as summarized by the demand curve :eq:`oli1` and the implementability constraint :eq:`oli5` that describes the best responses  of firms in  the competitive fringe

By substituting :eq:`oli1` into the above objective function, the monopolist's problem can be expressed as

.. math::
    :label: oli7

    \max_{\{u_t\}} \sum_{t=0}^\infty \beta^t \left\{ (A_0 - A_1 (\overline q_t + Q_t) + v_t) Q_t - eQ_t - .5gQ_t^2 - .5 c u_t^2 \right\}


subject to :eq:`oli6a`

This can be written

.. math::
    :label: oli9

    \max_{\{u_t\}} - \sum_{t=0}^\infty \beta^t \left\{ y_t' R y_t +   u_t' Q u_t \right\}


subject to :eq:`oli6a` where

.. math::

    R =  - \begin{bmatrix}
    0 & 0 & {A_0-e \over 2} & 0 & 0 \\
    0 & 0 & {1 \over 2} & 0 & 0 \\
    {A_0-e \over 2} & {1 \over 2} & - A_1 -.5g
    & -{A_1 \over 2} & 0 \\
    0 & 0 & -{A_1 \over 2} & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 \end{bmatrix}


and :math:`Q= {c \over 2}`

Under the Stackelberg plan, :math:`u_t = - F y_t`, which implies that
the evolution of :math:`y`  under the Stackelberg plan as

.. math::
    :label: oli20

    \overline y_{t+1} = (A - BF) \overline y_t


where :math:`\overline y_t = \begin{bmatrix}  1 & v_t & Q_t & \overline q_t & \overline i_t  \end{bmatrix}'`


Recursive formulation of a follower's problem
----------------------------------------------

We now make use of a "Big :math:`K`, little :math:`k`" trick (see  :doc:`rational expectations equilibrium <rational_expectations>`)  to formulate a recursive version of a follower's problem cast in terms of an ordinary Bellman equation


The individual firm faces :math:`\{p_t\}` as a price taker and believes

.. math::
    :label: oli22

    \begin{aligned}
    p_t & = a_0 - a_1 Q_t -a_1 \overline q_t + v_t \\
        &  \equiv E_p \begin{bmatrix} \overline y_t  \end{bmatrix}
    \end{aligned}


(Please remember that :math:`\overline q_t` is a component of :math:`\overline y_t`)

From the point of the view of a representative firm in the competitive fringe, :math:`\{\overline y_t\}` is an exogenous process

A representative fringe firm wants to forecast :math:`\overline y` because it 
wants to forecast what it regards as the exogenous price process :math:`\{p_t\}`

Therefore it wants to forecast the determinants of future prices

    * future values of :math:`Q` and

    * future values of :math:`\overline q`

An individual follower firm  confronts state :math:`\begin{bmatrix} \overline y_t & q_t \end{bmatrix}'` where :math:`q_t` is its current output as opposed to :math:`\overline q` within :math:`\overline y`

It believes that it chooses future values of :math:`q_t` but not future values of :math:`\overline q_t`

(This is an application of a ''Big :math:`K`, little :math:`k`'' idea)

The follower faces law of motion

.. math::
    :label: oli21

    \begin{bmatrix} \overline y_{t+1} \\
    q_{t+1} \end{bmatrix} = \begin{bmatrix} A - BF & 0 \\
    0  & 1 \end{bmatrix}  \begin{bmatrix} \overline y_{t} \\
    q_{t} \end{bmatrix} + \begin{bmatrix} 0 \cr 1 \end{bmatrix} i_t


We calculated :math:`F` and therefore :math:`A - B F` earlier

We can restate the optimization problem of the representative competitive firm

The firm takes :math:`\overline y_t` as an exogenous process and  chooses an output plan :math:`\{q_t\}` to maximize

.. math::

    E_0 \sum_{t=0}^\infty \beta^t \left\{ p_t q_t - \sigma_t \right\}, \quad \beta \in(0,1)


subject to :math:`q_0` given the law of motion :eq:`oli20` and the price function :eq:`oli22` and where the costs are still :math:`\sigma_t= d q_t + .5 h q_t^2 + .5 c (q_{t+1} - q_t)^2`

The representative firm's problem is a linear-quadratic dynamic programming 
problem with matrices :math:`A_s, B_s, Q_s, R_s` that can be constructed
easily from the above information

The representative firm's decision rule can be represented as

.. math::
    :label: oli23

    i_t = - F_s \begin{bmatrix} 1 \\
                                v_t \\
                                Q_t \\
                                \overline q_t \\
                                \overline i_t \\
                                q_t \end{bmatrix}


Now let's stare at the decision rule :eq:`oli23` for :math:`i_t`, apply 
"Big :math:`K`, little :math:`k`" logic again, and ask what we want in order to  
verify a recursive representation of a representative follower's choice problem

* We want decision rule :eq:`oli23` to have the property that :math:`i_t = \overline i_t` 
when we evaluate it at :math:`q_t = \overline q_t`

We inherit these desires from a "Big :math:`K`, little :math:`k`" logic

Here we apply a  "Big :math:`K`, little :math:`k`" logic in two parts to make 
the "representative firm be representative" *after* solving the
representative firm's optimization problem

    * We want :math:`q_t = \overline q_t`

    * We want :math:`i_t = \overline i_t`






Numerical example
-----------------

We computed the optimal Stackelberg plan for parameter settings 
:math:`A_0, A_1, \rho, C_\epsilon, c, d, e, g, h,  \beta` = :math:`100, 1, .8, .2, 1,  20, 20, .2, .2, .95` [#f10]_

.. TODO: The parameter listing might look better in a table or maybe an math align style equation.  It is hard to tell what value goes with what parameter when they are grouped like this.

For these parameter values the monopolist's decision rule is

.. math::

    u_t = (Q_{t+1} - Q_t) =\begin{bmatrix}  83.98 & 0.78 &  -0.95 &  -1.31  &  -2.07   \end{bmatrix}
    \begin{bmatrix} 1 \\
                                 v_t \\
                                 Q_t \\
                                 \overline q_t \\
                                 \overline i_t
                                 \end{bmatrix}


for :math:`t \geq 0`

and

.. math::

    x_0 \equiv \overline i_0 = \begin{bmatrix} 31.08 &   0.29  & -0.15  & -0.56     \end{bmatrix} \begin{bmatrix} 1 \\
                                v_0 \\
                                Q_0 \\
                                \overline q_0
                                \end{bmatrix}


For this example, starting from :math:`z_0 =\begin{bmatrix} 1 & v_0 & Q_0 & \overline q_0\end{bmatrix}  = \begin{bmatrix} 1& 0 & 25 & 46 \end{bmatrix}`,  the monopolist chooses to set :math:`i_0=1.43`

That choice implies that

* :math:`i_1=0.25`, and

* :math:`z_1 = \begin{bmatrix} 1 &  v_1 & Q_1 & \overline {q}_1 \end{bmatrix}  =  \begin{bmatrix} 1 & 0 & 21.83 & 47.43 \end{bmatrix}`

A monopolist who started from the initial conditions :math:`\tilde z_0= z_1` would set :math:`i_0=1.10` instead of :math:`.25` as called for under the original optimal plan


The preceding little calculation reflects the time inconsistency of the monopolist's optimal plan


The recursive representation of the decision rule for a representative fringe firm  is

.. math::

    i_t = \begin{bmatrix} 0 & 0 & 0 & .34 & 1 & - .34   \end{bmatrix}  \begin{bmatrix} 1 \\
                                v_t \\
                                Q_t \\
                                \overline q_t \\
                                \overline i_t \\
                                q_t \end{bmatrix} ,


which we have computed by solving the appropriate  linear-quadratic dynamic programming problem described above

Notice that, as expected, :math:`i_t = \overline i_t` when we evaluate this decision rule at :math:`q_t = \overline q_t`




Another Example
================

Please see :doc:`Ramsey plans, time Inconsistency, sustainable Plans<calvo>` for a Stackelberg plan computed using methods described here



Concluding Remarks
==================

This lecture is our first encounter with a class of problems in which optimal decision rules are history dependent [#f12]_

We shall encounter other examples in lectures :doc:`optimal taxation with state-contingent debt <opt_tax_recur>`
and :doc:`optimal taxation without state-contingent debt <amss>`




Many more examples of  such problems are described in chapters 20-24 of :cite:`Ljungqvist2012`


Exercises
=========

Exercise 1
----------

There is no uncertainty

For :math:`t \geq 0`, a monetary authority sets the growth of (the log of) money according to

.. math::
    :label: ex1a

    m_{t+1} = m_t + u_t


subject to the initial condition :math:`m_0>0` given

The demand for money is

.. math::
    :label: ex1b

    m_t - p_t = - \alpha (p_{t+1} - p_t)


where :math:`\alpha > 0` and :math:`p_t` is the log of the price level

Equation :eq:`ex1a` can be interpreted as the Euler equation of the holders of money

**a.** Briefly interpret how :eq:`ex1a` makes the demand for real balances vary 
inversely with the expected rate of inflation. Temporarily (only for this part of the exercise)
 drop :eq:`ex1a` and assume instead that :math:`\{m_t\}` is a given sequence satisfying 
 :math:`\sum_{t=0}^\infty m_t^2 < + \infty` -- solve the difference 
 equation :eq:`ex1a` "forward" to express :math:`p_t` as a function of current 
 and future values of :math:`m_s`. Note how future values of :math:`m` influence the current price level

At time :math:`0`, a monetary authority chooses (commits to) a possibly 
history-dependent strategy for setting :math:`\{u_t\}_{t=0}^\infty`

The monetary authority orders sequences :math:`\{m_t, p_t\}_{t=0}^\infty` according to

.. math::
    :label: ex1c

    -\sum_{t=0}^\infty .95^t \left[  (p_t - \overline p)^2 +
    u_t^2 + .00001 m_t^2  \right]


Assume that :math:`m_0=10, \alpha=5, \bar p=1`

**b.** Please briefly interpret this problem as one where the monetary 
authority wants to stabilize the price level, subject to costs of adjusting the 
money supply and some implementability constraints (we include the term 
:math:`.00001m_t^2` for purely technical reasons that you need not discuss)

**c.** Please write and run a Python program to find the optimal sequence :math:`\{u_t\}_{t=0}^\infty`

**d.** Display the optimal decision rule for :math:`u_t` as a function of :math:`u_{t-1},  m_t, m_{t-1}`

**e.** Compute the optimal :math:`\{m_t, p_t\}_t` sequence for :math:`t=0, \ldots,  10`

*Hints:*

* The optimal :math:`\{m_t\}` sequence must satisfy :math:`\sum_{t=0}^\infty (.95)^t m_t^2 < +\infty`

* Code can be found in the file `lqcontrol.py <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lqcontrol.py>`_ from the `QuantEcon.py <http://quantecon.org/python_index.html>`_ package that implements the optimal linear regulator


Exercise 2
----------

A representative consumer has quadratic utility functional

.. math::
    :label: ex2a

    \sum_{t=0}^\infty \beta^t \left\{ -.5 (b -c_t)^2 \right\}


where :math:`\beta \in (0,1)`, :math:`b = 30`, and :math:`c_t` is time :math:`t` consumption

The consumer faces a sequence of budget constraints

.. math::
    :label: ex2b

    c_t + a_{t+1} = (1+r)a_t + y_t - \tau_t


where

* :math:`a_t` is the household's holdings of an asset at the beginning of :math:`t`

* :math:`r >0` is a constant net interest rate satisfying :math:`\beta (1+r) <1`

* :math:`y_t` is the consumer's endowment at :math:`t`

The consumer's plan for :math:`(c_t, a_{t+1})` has to obey the boundary 
condition :math:`\sum_{t=0}^\infty \beta^t a_t^2 < + \infty`

Assume that :math:`y_0, a_0` are given initial conditions and that :math:`y_t` obeys

.. math::
    :label: ex2c

    y_t = \rho y_{t-1}, \quad t \geq 1,


where :math:`|\rho| <1`. Assume that :math:`a_0=0`, :math:`y_0=3`, and :math:`\rho=.9`

At time :math:`0`, a planner commits to a plan for taxes :math:`\{\tau_t\}_{t=0}^\infty`

The planner designs the plan to maximize

.. math::
    :label: ex2d

    \sum_{t=0}^\infty \beta^t \left\{ -.5 (c_t-b)^2 -   \tau_t^2\right\}


over :math:`\{c_t, \tau_t\}_{t=0}^\infty` subject to the implementability constraints in :eq:`ex2b` for :math:`t \geq 0` and

.. math::
    :label: ex2e

    \lambda_t =  \beta (1+r) \lambda_{t+1}


for :math:`t\geq 0`, where :math:`\lambda_t \equiv (b-c_t)`

**a.** Argue that :eq:`ex2e` is the Euler equation for a consumer who maximizes 
:eq:`ex2a` subject to :eq:`ex2b`, taking :math:`\{\tau_t\}` as a given sequence

**b.** Formulate the planner's problem as a Stackelberg problem

**c.** For :math:`\beta=.95, b=30, \beta(1+r)=.95`, formulate an artificial optimal 
linear regulator problem and use it to solve the Stackelberg problem

**d.** Give a recursive representation of the Stackelberg plan for :math:`\tau_t`


.. rubric:: Footnotes

.. [#f1] The problem assumes that there are no cross products between states and controls in the return function.  A simple transformation  converts a problem whose return function has cross products into an equivalent problem that has no cross products. For example, see :cite:`HansenSargent2008` (chapter 4, pp. 72-73).

.. [#f2] The government would make different choices were it to choose sequentially, that is,  were it to select its time :math:`t` action at time :math:`t`. See the lecture on :doc:`history dependent policies <hist_dep_policies>`

.. [#f8] :cite:`HansenSargent2008` (chapter 16), uses this model as a laboratory to illustrate an equilibrium concept featuring robustness in which at least one of the agents has doubts about the stochastic specification of the demand shock process.

.. [#f9] They used this method to compute a rational expectations competitive equilibrium.  Their key step was to eliminate price and output by substituting from the inverse demand curve and the production function into the firm's first-order conditions to get a difference equation in capital.

.. [#f10] These calculations were performed by functions located in `dyn_stack/oligopoly.py <https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/dyn_stack/oligopoly.py>`_.

.. [#f12] For another application of the techniques in this lecture and how they related to the method recommended by :cite:`KydlandPrescott1980`, please see :doc:`this lecture <hist_dep_policies>` .

.. TODO: Is the reference to KydlandPrescott to the correct paper?  When I think Kydland Prescott 1980, this is the paper I think of.  Does Evans Sargent refer to your paper with David Evans on History Dependent Public Policy?  I couldn't find the journal or other information on this paper.



.. TODO: in f4, f13 we need to fill in section link given placeholder XXXXXX (CC: What links?)
