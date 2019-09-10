.. _opt_tax_recur:

.. include:: /_static/includes/header.raw

*******************************************
Optimal Taxation with State-Contingent Debt
*******************************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon


Overview
=========


This lecture describes a celebrated model of optimal fiscal policy by Robert E.
Lucas, Jr., and Nancy Stokey  :cite:`LucasStokey1983`.

The model revisits classic issues about how to pay for a war.

Here a *war* means a more  or less temporary surge in an exogenous government expenditure process.

The model features

* a government that must finance an exogenous stream of government expenditures with  either

   * a flat rate tax on labor, or

   * purchases and sales from a full array of Arrow state-contingent securities

* a representative household that values consumption and leisure

* a linear production function mapping labor into a single good

* a Ramsey planner who at time :math:`t=0` chooses a plan for taxes and
  trades of `Arrow securities <https://en.wikipedia.org/wiki/Arrow_security>`__ for all :math:`t \geq 0`

After first presenting the model in a space of sequences, we shall represent it
recursively in terms of two Bellman equations formulated along lines that we
encountered in :doc:`Dynamic Stackelberg models <dyn_stack>`.

As in :doc:`Dynamic Stackelberg models <dyn_stack>`, to apply dynamic programming
we shall define the state vector artfully.

In particular, we shall include forward-looking variables that summarize  optimal
responses of private agents to a Ramsey plan.

See :doc:`Optimal taxation <lqramsey>` for analysis within a linear-quadratic setting.

Let's start with some standard imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    

A Competitive Equilibrium with Distorting Taxes
===============================================


For :math:`t \geq 0`, a history :math:`s^t = [s_t, s_{t-1}, \ldots, s_0]` of an
exogenous state :math:`s_t` has joint probability density :math:`\pi_t(s^t)`.

We begin by assuming that government purchases :math:`g_t(s^t)` at time :math:`t \geq 0` depend on :math:`s^t`.

Let :math:`c_t(s^t)`,  :math:`\ell_t(s^t)`, and :math:`n_t(s^t)` denote consumption,
leisure, and labor supply, respectively, at history :math:`s^t` and date :math:`t`.

A representative  household is endowed with one unit of time that can be divided
between leisure :math:`\ell_t` and labor :math:`n_t`:

.. math::
    :label: feas1_opt_tax

    n_t(s^t) + \ell_t(s^t) = 1


Output equals :math:`n_t(s^t)` and can be divided between :math:`c_t(s^t)` and :math:`g_t(s^t)`

.. math::
    :label: TSs_techr_opt_tax

    c_t(s^t) + g_t(s^t) = n_t(s^t)


A representative household's preferences over :math:`\{c_t(s^t), \ell_t(s^t)\}_{t=0}^\infty` are ordered by

.. math::
    :label: TS_prefr_opt_tax

    \sum_{t=0}^\infty \sum_{s^t} \beta^t \pi_t(s^t) u[c_t(s^t), \ell_t(s^t)]


where the utility function :math:`u` is  increasing, strictly concave, and three
times continuously differentiable in both arguments.

The technology pins down a pre-tax wage rate to unity for all :math:`t, s^t`.

The government imposes a flat-rate tax :math:`\tau_t(s^t)` on labor income at
time :math:`t`, history :math:`s^t`.

There are complete markets in one-period Arrow securities.

One unit of an Arrow security  issued at time :math:`t` at history :math:`s^t`
and promising to pay one unit of time :math:`t+1` consumption in state :math:`s_{t+1}`
costs :math:`p_{t+1}(s_{t+1}|s^t)`.

The government issues one-period Arrow securities each period.

The government has a sequence of budget constraints whose time :math:`t \geq 0` component is

.. math::
    :label: TS_govr

    g_t(s^t) =   \tau_t(s^t)  n_t(s^t) + \sum_{s_{t+1}} p_{t+1}(s_{t+1} | s^t) b_{t+1}(s_{t+1} | s^t) -
    b_t(s_t | s^{t-1})


where

* :math:`p_{t+1}(s_{t+1}|s^t)` is a competitive equilibrium price of one unit of
  consumption at date :math:`t+1` in state :math:`s_{t+1}` at date :math:`t` and history :math:`s^t`.

* :math:`b_t(s_t|s^{t-1})` is government debt falling due at time :math:`t`, history :math:`s^t`.


Government debt :math:`b_0(s_0)` is an exogenous initial condition.

The representative household has a sequence of budget constraints whose time :math:`t\geq 0` component is

.. math::
    :label: TS_bcr

    c_t(s^t) + \sum_{s_{t+1}} p_t(s_{t+1} | s^t) b_{t+1}(s_{t+1} | s^t)
    =  \left[1-\tau_t(s^t)\right]  n_t(s^t) + b_t(s_t | s^{t-1}) \quad \forall t \geq 0


A  **government policy** is an exogenous sequence :math:`\{g(s_t)\}_{t=0}^\infty`,
a tax rate sequence :math:`\{\tau_t(s^t)\}_{t=0}^\infty`, and a government debt sequence :math:`\{b_{t+1}(s^{t+1})\}_{t=0}^\infty`.

A **feasible allocation** is a consumption-labor supply plan :math:`\{c_t(s^t), n_t(s^t)\}_{t=0}^\infty`
that satisfies :eq:`TSs_techr_opt_tax` at all :math:`t, s^t`.

A **price system** is a sequence of Arrow security prices :math:`\{p_{t+1}(s_{t+1} | s^t) \}_{t=0}^\infty`.

The household faces the price system as a price-taker and takes the government policy as given.

The household chooses :math:`\{c_t(s^t), \ell_t(s^t)\}_{t=0}^\infty` to maximize :eq:`TS_prefr_opt_tax` subject to :eq:`TS_bcr` and :eq:`feas1_opt_tax` for all :math:`t, s^t`.

A **competitive equilibrium with distorting taxes** is a feasible allocation,
a price system, and a government policy such that

* Given the price system and the government policy, the allocation solves the
  household's optimization problem.

* Given the allocation, government policy, and  price system, the government's
  budget constraint is satisfied for all :math:`t, s^t`.

Note: There are many competitive equilibria with distorting taxes.

They are indexed by different government policies.

The **Ramsey problem** or **optimal taxation problem** is to choose a competitive
equilibrium with distorting taxes that maximizes :eq:`TS_prefr_opt_tax`.

Arrow-Debreu Version of Price System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We find it convenient sometimes to work with the Arrow-Debreu price system that is
implied by a sequence of Arrow securities prices.

Let :math:`q_t^0(s^t)` be the price at time :math:`0`, measured in time :math:`0`
consumption goods, of one unit of consumption at time :math:`t`,
history :math:`s^t`.

The following recursion relates Arrow-Debreu prices :math:`\{q_t^0(s^t)\}_{t=0}^\infty`
to Arrow securities prices :math:`\{p_{t+1}(s_{t+1}|s^t)\}_{t=0}^\infty`

.. math::
    :label: TS_foc

    q^0_{t+1}(s^{t+1}) = p_{t+1}(s_{t+1}|s^t) q^0_t(s^t) \quad s.t. \quad q_0^0(s^0) = 1

Arrow-Debreu prices are useful when we want to compress a sequence of budget
constraints into a single intertemporal budget constraint, as we shall find it
convenient to do below.

Primal Approach
^^^^^^^^^^^^^^^

We apply a popular approach to solving a Ramsey problem, called the *primal approach*.

The idea is to use first-order conditions for household optimization to
eliminate taxes and prices in favor of quantities, then pose an optimization problem
cast entirely in terms of quantities.

After Ramsey quantities have been found, taxes and prices can then be unwound
from the allocation.

The primal approach uses four steps:

1. Obtain  first-order conditions of the household's problem and solve them for :math:`\{q^0_t(s^t), \tau_t(s^t)\}_{t=0}^\infty` as functions of
   the allocation :math:`\{c_t(s^t), n_t(s^t)\}_{t=0}^\infty`.

2. Substitute these expressions for taxes and prices in terms of the allocation
   into the household's present-value budget constraint.

   * This intertemporal constraint involves only the allocation and is regarded
     as an *implementability constraint*.

3. Find the allocation that maximizes the utility of the representative household
   :eq:`TS_prefr_opt_tax` subject to  the feasibility constraints :eq:`feas1_opt_tax`
   and :eq:`TSs_techr_opt_tax`  and the implementability condition derived in step 2.

   *  This optimal allocation is called the **Ramsey allocation**.

4. Use the Ramsey  allocation together with the formulas from step 1 to find
   taxes and prices.


The Implementability Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By sequential substitution of one one-period budget constraint :eq:`TS_bcr` into
another, we can obtain the household's present-value budget constraint:

.. math::
    :label: TS_bcPV2

    \sum_{t=0}^\infty \sum_{s^t} q^0_t(s^t) c_t(s^t) =
    \sum_{t=0}^\infty \sum_{s^t} q^0_t(s^t) [1-\tau_t(s^t)] n_t(s^t) +
    b_0


:math:`\{q^0_t(s^t)\}_{t=1}^\infty` can be interpreted as a time :math:`0`
Arrow-Debreu price system.

To approach the Ramsey problem, we study the household's optimization problem.

First-order conditions for the household’s problem for :math:`\ell_t(s^t)`
and :math:`b_t(s_{t+1}| s^t)`, respectively, imply

.. math::
    :label: LSA_taxr

    (1 - \tau_t(s^t)) = {\frac{u_l(s^t)}{u_c(s^t)}}


and

.. math::
    :label: LS101

    p_{t+1}(s_{t+1}| s^t) = \beta \pi(s_{t+1} | s^t) \left({\frac{u_c(s^{t+1})}{u_c({s^t})}} \right)


where :math:`\pi(s_{t+1} | s^t)` is the probability distribution of :math:`s_{t+1}`
conditional on history :math:`s^t`.

Equation :eq:`LS101` implies that the Arrow-Debreu price system satisfies

.. math::
    :label: LS102

    q_t^0(s^t) = \beta^{t} \pi_{t}(s^{t})
                                {u_c(s^{t})  \over u_c(s^0)}


Using the first-order conditions :eq:`LSA_taxr` and :eq:`LS101` to eliminate
taxes and prices from :eq:`TS_bcPV2`, we derive the *implementability condition*

.. math::
    :label: TSs_cham1

    \sum_{t=0}^\infty  \sum_{s^t} \beta^t \pi_t(s^t)
             [u_c(s^t) c_t(s^t) - u_\ell(s^t) n_t(s^t)] -
             u_c(s^0) b_0 = 0


The **Ramsey problem** is to choose a feasible  allocation  that maximizes

.. math::
    :label: TS_prefr2

    \sum_{t=0}^\infty \sum_{s^t} \beta^t \pi_t(s^t) u[c_t(s^t), 1 - n_t(s^t)]


subject to  :eq:`TSs_cham1`.


Solution Details
^^^^^^^^^^^^^^^^

First, define a "pseudo utility function"

.. math::
    :label: TS_cham17

    V\left[c_t(s^t), n_t(s^t), \Phi\right] =
    u[c_t(s^t),1-n_t(s^t)] +
    \Phi \left[ u_c(s^t) c_t(s^t) -
    u_\ell(s^t) n_t(s^t) \right]


where :math:`\Phi` is a Lagrange multiplier on the implementability condition :eq:`TS_bcPV2`.

Next form the Lagrangian

.. math::
    :label: TS_chamlag

    J  = \sum_{t=0}^\infty
    \sum_{s^t} \beta^t \pi_t(s^t)
    \Bigl\{
        V[c_t(s^t), n_t(s^t), \Phi]   + \theta_t(s^t)
       \Bigl[ n_t(s^t) - c_t(s^t) - g_t(s_t) \Bigr]
    \Bigr\} - \Phi u_c(0) b_0


where  :math:`\{\theta_t(s^t); \forall s^t\}_{t\geq0}` is a sequence of Lagrange
multipliers on the feasible conditions :eq:`TSs_techr_opt_tax`.

Given an initial government debt :math:`b_0`,  we want to maximize :math:`J`
with respect to :math:`\{c_t(s^t), n_t(s^t); \forall s^t \}_{t\geq0}`   and to minimize with respect
to :math:`\{\theta(s^t); \forall s^t \}_{t\geq0}`.

The first-order conditions for the Ramsey problem for periods :math:`t \geq 1` and :math:`t=0`, respectively, are

.. math::
    :label: eqFONCRamsey1

     \begin{aligned}
        c_t(s^t)\rm{:} &
        \; (1+\Phi) u_c(s^t) + \Phi \left[u_{cc}(s^t) c_t(s^t) -
        u_{\ell c}(s^t) n_t(s^t) \right]  - \theta_t(s^t) = 0, \quad t \geq 1
        \\
        n_t(s^t)\rm{:} &
        \; -(1+\Phi) u_{\ell}(s^t) - \Phi \left[u_{c\ell}(s^t) c_t(s^t) -
        u_{\ell \ell}(s^t) n_t(s^t) \right] + \theta_t(s^t) = 0, \quad t \geq 1
    \end{aligned}


and

.. math::
    :label: eqFONCRamsey0

    \begin{aligned}
        c_0(s^0, b_0)\rm{:} &
        \; (1+\Phi) u_c(s^0, b_0) + \Phi \left[u_{cc}(s^0, b_0) c_0(s^0, b_0) -
        u_{\ell c}(s^0, b_0) n_0(s^0, b_0) \right]  - \theta_0(s^0, b_0)   \\
           & \quad \quad \quad \quad \quad \quad  - \Phi u_{cc}(s^0, b_0) b_0 = 0
        \\
        n_0(s^0, b_0)\rm{:} &
        \; -(1+\Phi) u_{\ell}(s^0, b_0) - \Phi \left[u_{c\ell}(s^0, b_0) c_0(s^0, b_0) -
        u_{\ell \ell}(s^0, b_0) n_0(s^0, b_0) \right] + \theta_0(s^0, b_0) \\
          & \quad \quad \quad \quad \quad \quad + \Phi u_{c \ell}(s^0, b_0) b_0 = 0
    \end{aligned}

Please note how these first-order conditions differ between :math:`t=0` and :math:`t \geq 1`.

It is instructive to use first-order conditions :eq:`eqFONCRamsey1` for
:math:`t \geq 1` to eliminate the multipliers :math:`\theta_t(s^t)`.

For convenience, we suppress the time subscript and the index :math:`s^t` and obtain

.. math::
    :label: TS_barg

    \begin{aligned}
      (1+\Phi) &u_c(c,1-c-g) + \Phi \bigl[c u_{cc}(c,1-c-g) -
      (c+g) u_{\ell c}(c,1-c-g) \bigr]
        \\
        &= (1+\Phi) u_{\ell}(c,1-c-g) + \Phi \bigl[c u_{c\ell}(c,1-c-g) -
        (c+g) u_{\ell \ell}(c,1-c-g)  \bigr]
    \end{aligned}


where we have imposed conditions :eq:`feas1_opt_tax` and :eq:`TSs_techr_opt_tax`.

Equation :eq:`TS_barg` is one equation that can be solved to express the
unknown :math:`c` as a function of the  exogenous variable :math:`g`.


We also know that  time :math:`t=0` quantities :math:`c_0` and :math:`n_0` satisfy

.. math::
    :label: TS_barg_aust

    \begin{aligned}
          (1+\Phi) &u_c(c,1-c-g) + \Phi \bigl[c u_{cc}(c,1-c-g) -
          (c+g) u_{\ell c}(c,1-c-g) \bigr]
            \\
            &= (1+\Phi) u_{\ell}(c,1-c-g) + \Phi \bigl[c u_{c\ell}(c,1-c-g) -
            (c+g) u_{\ell \ell}(c,1-c-g)  \bigr] + \Phi (u_{cc} - u_{c,\ell}) b_0
    \end{aligned}

Notice that a counterpart to :math:`b_0` does *not* appear
in :eq:`TS_barg`, so :math:`c` does not depend on it for :math:`t \geq 1`.

But things are different for time :math:`t=0`.

An analogous argument for the :math:`t=0` equations :eq:`eqFONCRamsey0` leads
to one equation that can be solved for :math:`c_0` as a function of the
pair :math:`(g(s_0), b_0)`.

These outcomes mean that the following statement would be  true even when
government purchases are history-dependent functions :math:`g_t(s^t)` of the
history of :math:`s^t`.

**Proposition:**
If government purchases are equal after two histories
:math:`s^t` and :math:`\tilde s^\tau` for :math:`t,\tau\geq0`, i.e., if

.. math::

    g_t(s^t) = g^\tau(\tilde s^\tau) = g


then it follows from :eq:`TS_barg` that the Ramsey  choices of consumption and leisure,
:math:`(c_t(s^t),\ell_t(s^t))` and :math:`(c_j(\tilde s^\tau),\ell_j(\tilde
s^\tau))`, are identical.

The proposition asserts that the optimal allocation is a function of the
currently realized quantity of government purchases :math:`g` only and does
*not* depend on the specific history that preceded that realization of :math:`g`.




The Ramsey Allocation for a Given Multiplier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Temporarily take  :math:`\Phi` as given.

We shall compute :math:`c_0(s^0, b_0)` and :math:`n_0(s^0, b_0)` from the first-order
conditions :eq:`eqFONCRamsey0`.

Evidently, for :math:`t \geq 1`, :math:`c` and
:math:`n` depend on the time :math:`t` realization of :math:`g` only.

But for :math:`t=0`, :math:`c` and :math:`n`  depend on both :math:`g_0` and the
government’s initial debt :math:`b_0`.

Thus, while :math:`b_0`
influences :math:`c_0` and :math:`n_0`, there appears no analogous
variable :math:`b_t` that influences :math:`c_t` and :math:`n_t` for
:math:`t \geq 1`.

The absence of :math:`b_t` as a determinant of the  Ramsey allocation for
:math:`t \geq 1` and its presence for :math:`t=0` is a symptom of the
*time-inconsistency* of a Ramsey plan.

:math:`\Phi` has to take a value that assures that
the household and the government’s budget constraints are both
satisfied at a candidate Ramsey allocation and price system associated
with that :math:`\Phi`.


Further Specialization
^^^^^^^^^^^^^^^^^^^^^^

At this point, it is useful to specialize the model in the following ways.

We assume that :math:`s` is governed by a finite state Markov chain with states
:math:`s\in [1, \ldots, S]` and transition matrix :math:`\Pi`, where

.. math::

    \Pi(s'|s) = {\rm Prob}(s_{t+1} = s'| s_t =s)


Also, assume that government purchases :math:`g` are an exact time-invariant function
:math:`g(s)` of :math:`s`.

We maintain these assumptions throughout the remainder of this lecture.


Determining the Multiplier
^^^^^^^^^^^^^^^^^^^^^^^^^^

We complete the Ramsey plan by computing the Lagrange multiplier :math:`\Phi`
on the implementability constraint :eq:`TSs_cham1`.

Government budget balance restricts :math:`\Phi` via the following line of reasoning.

The household's first-order conditions imply

.. math::
    :label: LSA_ta

    (1 - \tau_t(s^t)) = {\frac{u_l(s^t)}{u_c(s^t)} }


and the  implied one-period Arrow securities prices

.. math::
    :label: LSA_Arro

    p_{t+1}(s_{t+1}| s^t) = \beta \Pi(s_{t+1} | s_t) {\frac{u_c(s^{t+1})}{u_c({s^t})}}


Substituting from  :eq:`LSA_ta`, :eq:`LSA_Arro`, and the feasibility
condition :eq:`TSs_techr_opt_tax`  into the recursive version :eq:`TS_bcr` of
the household budget constraint gives

.. math::
    :label: LSA_budget

    \begin{aligned}
        u_c(s^t) [ n_t(s^t) - g_t(s^t)] +
        \beta \sum_{s_{t+1}} \Pi (s_{t+1}| s_t) u_c(s^{t+1}) b_{t+1}(s_{t+1} | s^t) \\ =
        u_l (s^t) n_t(s^t) + u_c(s^t) b_t(s_t | s^{t-1})
    \end{aligned}


Define :math:`x_t(s^t) = u_c(s^t) b_t(s_t | s^{t-1})`.

Notice that :math:`x_t(s^t)` appears on the right side of :eq:`LSA_budget` while
:math:`\beta` times the conditional expectation of
:math:`x_{t+1}(s^{t+1})` appears on the left side.

Hence the equation shares much of the structure of a simple asset pricing equation with
:math:`x_t` being analogous to the price of the asset at time :math:`t`.

We learned earlier that for a Ramsey allocation
:math:`c_t(s^t), n_t(s^t)` and :math:`b_t(s_t|s^{t-1})`, and therefore
also :math:`x_t(s^t)`, are each functions of :math:`s_t` only,  being
independent of the history :math:`s^{t-1}` for :math:`t \geq 1`.

That means that we can express equation :eq:`LSA_budget` as

.. math::
    :label: LSA_budget2

    u_c(s)
    [ n(s) - g(s)]  + \beta
    \sum_{s'} \Pi(s' | s) x'(s')  = u_l(s) n(s) + x(s)


where :math:`s'` denotes a next period value of :math:`s` and
:math:`x'(s')` denotes a next period value of :math:`x`.

Equation :eq:`LSA_budget2` is easy to solve for :math:`x(s)` for
:math:`s = 1, \ldots , S`.

If we let :math:`\vec n, \vec g, \vec x`
denote :math:`S \times 1` vectors whose :math:`i`\ th elements are the
respective :math:`n, g`, and :math:`x` values when :math:`s=i`, and let
:math:`\Pi` be the transition matrix for the Markov state :math:`s`,
then we can express :eq:`LSA_budget2` as the matrix equation

.. math::
    :label: LSA_budget20

    \vec u_c(\vec n - \vec g) + \beta \Pi \vec x = \vec u_l \vec n + \vec x


This is a system of :math:`S` linear equations in the :math:`S \times 1`
vector :math:`x`, whose solution is

.. math::
    :label: LSA_xsol

    \vec x= (I - \beta \Pi )^{-1} [ \vec u_c (\vec n-\vec g) - \vec u_l \vec n]


In these equations, by :math:`\vec u_c \vec n`, for example, we mean
element-by-element multiplication of the two vectors.

After solving for :math:`\vec x`, we can find :math:`b(s_t|s^{t-1})` in Markov
state :math:`s_t=s` from :math:`b(s) = {\frac{x(s)}{u_c(s)}}` or the matrix equation

.. math::
    :label: LSA_bsol

    \vec b = {\frac{ \vec x }{\vec u_c}}


where division here means an element-by-element division of the respective
components of the :math:`S \times 1` vectors :math:`\vec x` and
:math:`\vec u_c`.

Here is a computational algorithm:

1.  Start with a guess for the value for :math:`\Phi`, then use the
    first-order conditions and the feasibility conditions to compute
    :math:`c(s_t), n(s_t)` for :math:`s \in [1,\ldots, S]` and
    :math:`c_0(s_0,b_0)` and :math:`n_0(s_0, b_0)`, given :math:`\Phi`.

    * these are :math:`2  (S+1)` equations in :math:`2  (S+1)` unknowns.

2. Solve the :math:`S` equations :eq:`LSA_xsol` for the :math:`S` elements
   of :math:`\vec x`.

   * these depend on :math:`\Phi`.

3. Find a :math:`\Phi` that satisfies

    .. math::
        :label: Bellman2cons

        u_{c,0} b_0 = u_{c,0} (n_0 - g_0) - u_{l,0} n_0  + \beta \sum_{s=1}^S \Pi(s | s_0) x(s)

   by gradually raising :math:`\Phi` if the left side of :eq:`Bellman2cons`
   exceeds the right side and lowering :math:`\Phi` if the left side is less than the right side.

4. After computing a Ramsey allocation,  recover the flat tax rate on
   labor from :eq:`LSA_taxr` and the implied one-period Arrow securities
   prices from :eq:`LS101`.

In summary, when :math:`g_t` is a time-invariant function of a Markov state
:math:`s_t`, a Ramsey plan can be constructed by solving :math:`3S +3`
equations in :math:`S` components each of :math:`\vec c`, :math:`\vec n`, and
:math:`\vec x` together with :math:`n_0, c_0`, and :math:`\Phi`.




Time Inconsistency
^^^^^^^^^^^^^^^^^^

Let :math:`\{\tau_t(s^t)\}_{t=0}^\infty, \{b_{t+1}(s_{t+1}| s^t)\}_{t=0}^\infty`
be a time :math:`0`, state :math:`s_0` Ramsey plan.

Then :math:`\{\tau_j(s^j)\}_{j=t}^\infty, \{b_{j+1}(s_{j+1}| s^j)\}_{j=t}^\infty`
is a time :math:`t`, history :math:`s^t` continuation of a time
:math:`0`, state :math:`s_0` Ramsey plan.

A time :math:`t`, history :math:`s^t` Ramsey plan is a Ramsey plan that
starts from initial conditions :math:`s^t, b_t(s_t|s^{t-1})`.

A time :math:`t`, history :math:`s^t`
continuation of a time :math:`0`, state :math:`0` Ramsey plan is
*not* a time :math:`t`, history :math:`s^t` Ramsey plan.

The means that a Ramsey plan is *not time consistent*.

Another way to say the same thing is that a Ramsey plan is *time inconsistent*.

The reason is that a continuation Ramsey plan takes :math:`u_{ct} b_t(s_t|s^{t-1})` as given, not
:math:`b_t(s_t|s^{t-1})`.

We shall discuss this more below.




Specification with CRRA Utility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



In our calculations below and in a :doc:`subsequent lecture <amss>` based on an extension of the Lucas-Stokey model
by  Aiyagari, Marcet, Sargent, and Seppälä (2002) :cite:`aiyagari2002optimal`, we shall modify the one-period utility function assumed above.

(We adopted the preceding utility specification because it was the one used in  the original :cite:`LucasStokey1983` paper)

We will  modify their specification by instead assuming that the  representative agent has  utility function

.. math::

    u(c,n) = {\frac{c^{1-\sigma}}{1-\sigma}} - {\frac{n^{1+\gamma}}{1+\gamma}}


where  :math:`\sigma > 0`, :math:`\gamma >0`.

We continue to assume that

.. math::

    c_t + g_t = n_t

We eliminate leisure from the model.

We also eliminate Lucas and Stokey's restriction that :math:`\ell_t + n_t \leq 1`.

We replace these two things with the assumption that
labor :math:`n_t \in [0, +\infty]`.

With these adjustments, the analysis of Lucas and Stokey prevails once we make the following replacements

.. math::

        \begin{aligned}
        u_\ell(c, \ell) &\sim - u_n(c, n) \\
        u_c(c,\ell) &\sim u_c(c,n) \\
        u_{\ell,\ell}(c,\ell) &\sim u_{nn}(c,n) \\
        u_{c,c}(c,\ell)& \sim u_{c,c}(c,n) \\
        u_{c,\ell} (c,\ell) &\sim 0
        \end{aligned}


With these understandings, equations :eq:`TS_barg` and :eq:`TS_barg_aust` simplify in the case of the CRRA utility function.

They become

.. math::
    :label: TS_barg10

    (1+\Phi) [u_c(c) + u_n(c+g)] + \Phi[c u_{cc}(c) + (c+g) u_{nn}(c+g)] = 0

and

.. math::
    :label: TS_barg11

    (1+\Phi) [u_c(c_0) + u_n(c_0+g_0)] + \Phi[c_0 u_{cc}(c_0) + (c_0+g_0) u_{nn}(c_0+g_0)] - \Phi u_{cc}(c_0) b_0 = 0

In equation :eq:`TS_barg10`, it is understood that :math:`c` and :math:`g` are each functions of the Markov state :math:`s`.

In addition, the time :math:`t=0` budget constraint is satisfied at :math:`c_0` and initial government debt
:math:`b_0`:

.. math::
    :label: opt_tax_eqn_10

    b_0 + g_0 = \tau_0 (c_0 + g_0) + \frac{\bar b}{R_0}

where :math:`R_0` is the gross interest rate for the Markov state :math:`s_0` that is assumed to prevail at time :math:`t =0`
and :math:`\tau_0` is the time :math:`t=0` tax rate.


In equation :eq:`opt_tax_eqn_10`, it is understood that

.. math::
    :nowrap:

    \begin{aligned}
    \tau_0 = 1 - \frac{u_{l,0}}{u_{c,0}} \\
    R_0 =  \beta  \sum_{s=1}^S \Pi(s | s_0) \frac{u_c(s)}{u_{c,0}}
    \end{aligned}


Sequence Implementation
^^^^^^^^^^^^^^^^^^^^^^^



The above steps are implemented in a class called `SequentialAllocation`

.. literalinclude:: /_static/lecture_specific/opt_tax_recur/sequential_allocation.py




Recursive Formulation of the Ramsey Problem
===========================================

:math:`x_t(s^t) = u_c(s^t) b_t(s_t | s^{t-1})` in equation :eq:`LSA_budget`
appears to be a purely “forward-looking” variable.

But :math:`x_t(s^t)` is a also a  natural candidate for a state variable in
a recursive formulation of the Ramsey problem.

Intertemporal Delegation
^^^^^^^^^^^^^^^^^^^^^^^^

To express a Ramsey plan recursively, we imagine that a time :math:`0`
Ramsey planner is followed by a sequence of continuation Ramsey planners
at times :math:`t = 1, 2, \ldots`.

A “continuation Ramsey planner” has a
different objective function and faces different constraints than a
Ramsey planner.

A key step in representing a Ramsey plan recursively is
to regard the marginal utility scaled government debts
:math:`x_t(s^t) = u_c(s^t) b_t(s_t|s^{t-1})` as predetermined quantities
that continuation Ramsey planners at times :math:`t \geq 1` are
obligated to attain.

Continuation Ramsey planners do this by choosing continuation policies that induce the representative
household to make choices that imply that :math:`u_c(s^t) b_t(s_t|s^{t-1})= x_t(s^t)`.

A time :math:`t\geq 1` continuation Ramsey planner
delivers :math:`x_t` by choosing a suitable :math:`n_t, c_t` pair and a list of
:math:`s_{t+1}`-contingent continuation quantities :math:`x_{t+1}` to
bequeath to a time :math:`t+1` continuation Ramsey planner.

A time :math:`t \geq 1` continuation Ramsey planner faces :math:`x_t, s_t` as
state variables.

But the time :math:`0` Ramsey planner faces :math:`b_0`, not :math:`x_0`,
as a state variable.

Furthermore, the Ramsey planner cares about :math:`(c_0(s_0), \ell_0(s_0))`, while
continuation Ramsey planners do not.

The time :math:`0` Ramsey planner
hands :math:`x_1` as a function of :math:`s_1` to a time :math:`1`
continuation Ramsey planner.

These lines of delegated authorities and
responsibilities across time express the continuation Ramsey planners’
obligations to implement their parts of the original Ramsey plan,
designed once-and-for-all at time :math:`0`.

Two Bellman Equations
^^^^^^^^^^^^^^^^^^^^^

After :math:`s_t` has been realized at time :math:`t \geq 1`, the state
variables confronting the time :math:`t` **continuation Ramsey planner** are
:math:`(x_t, s_t)`.

* Let :math:`V(x, s)` be the value of a **continuation Ramsey plan** at :math:`x_t = x, s_t =s` for :math:`t \geq 1`.

* Let :math:`W(b, s)` be the value of a **Ramsey plan** at time :math:`0` at :math:`b_0=b` and :math:`s_0 = s`.

We work backward by presenting a Bellman equation for
:math:`V(x,s)` first, then a Bellman equation for :math:`W(b,s)`.

The Continuation Ramsey Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Bellman equation for a time :math:`t \geq 1` continuation Ramsey
planner is

.. math::
    :label: LSA_Bellman1

    V(x, s) = \max_{n, \{x'(s')\}} u(n-g(s), 1-n) + \beta \sum_{s'\in S} \Pi(s'| s) V(x', s')


where maximization over :math:`n` and the :math:`S` elements of
:math:`x'(s')` is subject to the single implementability constraint for
:math:`t \geq 1`.

.. math::
    :label: LSA_Bellman1cons

    x = u_c(n-g(s)) -  u_l n + \beta \sum_{s' \in {\cal S}} \Pi(s' | s) x'(s')


Here :math:`u_c` and :math:`u_l` are today’s values of the marginal utilities.

For each given value of :math:`x, s`, the continuation Ramsey planner chooses :math:`n` and :math:`x'(s')`
for each :math:`s' \in {\cal S}`.

Associated with a value function :math:`V(x,s)` that solves Bellman equation :eq:`LSA_Bellman1`
are :math:`S+1` time-invariant policy functions

.. math::
    :label: RRpolicyt

    \begin{aligned}
        n_t & = f(x_t, s_t), \quad t \geq 1
        \\
        x_{t+1}(s_{t+1}) & = h(s_{t+1}; x_t, s_t), \, s_{t+1} \in  {\cal S}, \, t \geq 1
    \end{aligned}


The Ramsey Problem
^^^^^^^^^^^^^^^^^^

The Bellman equation for the time :math:`0` Ramsey planner is

.. math::
    :label: LSA_Bellman2

    W(b_0, s_0) = \max_{n_0, \{x'(s_1)\}} u(n_0 - g_0, 1 - n_0) + \beta \sum_{s_1 \in {\cal S}} \Pi(s_1| s_0) V( x'(s_1), s_1)


where maximization over :math:`n_0` and the :math:`S` elements of
:math:`x'(s_1)` is subject to the time :math:`0` implementability
constraint

.. math::
    :label: Bellman2cons2

    u_{c,0} b_0 = u_{c,0} (n_0 - g_0) - u_{l,0} n_0  + \beta \sum_{s_1\in {\cal S}} \Pi(s_1 | s_0) x'(s_1)


coming from restriction :eq:`Bellman2cons`.

Associated with a value function :math:`W(b_0, n_0)` that solves  Bellman equation :eq:`LSA_Bellman2` are
:math:`S +1` time :math:`0` policy functions

.. math::
    :label: RRpolicy0

    \begin{aligned}
        n_0
            & = f_0(b_0, s_0) \cr
        x_1(s_1)
            & = h_0(s_1; b_0, s_0)
    \end{aligned}


Notice the appearance of state variables :math:`(b_0, s_0)` in the time
:math:`0` policy functions for the Ramsey planner as compared to
:math:`(x_t, s_t)` in the policy functions :eq:`RRpolicyt` for the time :math:`t \geq 1`
continuation Ramsey planners.

The value function :math:`V(x_t, s_t)` of the time :math:`t`
continuation Ramsey planner equals
:math:`E_t \sum_{\tau = t}^\infty \beta^{\tau - t} u(c_t, l_t)`, where
the consumption and leisure processes are evaluated along the original
time :math:`0` Ramsey plan.

First-Order Conditions
^^^^^^^^^^^^^^^^^^^^^^

Attach a Lagrange multiplier :math:`\Phi_1(x,s)` to constraint :eq:`LSA_Bellman1cons` and a
Lagrange multiplier :math:`\Phi_0` to constraint :eq:`Bellman2cons`.

Time :math:`t \geq 1`: the first-order conditions for the time :math:`t \geq 1` constrained
maximization problem on the right side of the continuation Ramsey
planner’s Bellman equation :eq:`LSA_Bellman1` are

.. math::
    :label: LSARxt

    \beta \Pi(s' | s) V_x (x', s') - \beta \Pi(s' | s) \Phi_1 = 0


for :math:`x'(s')` and

.. math::
    :label: LSARnt

    (1 + \Phi_1) (u_c - u_l ) + \Phi_1 \left[ n (u_{ll} - u_{lc}) + (n-g(s)) (u_{cc} - u_{lc})  \right] = 0


for :math:`n`.

Given :math:`\Phi_1`, equation :eq:`LSARnt` is one equation to be
solved for :math:`n` as a function of :math:`s` (or of :math:`g(s)`).

Equation :eq:`LSARxt` implies :math:`V_x(x', s')= \Phi_1`, while an envelope
condition is :math:`V_x(x,s) = \Phi_1`, so it follows that

.. math::
    :label: LSAenv

    V_x(x', s') = V_x(x,s) = \Phi_1(x,s)


Time :math:`t=0`: For the time :math:`0` problem on the right side of the Ramsey planner’s
Bellman equation :eq:`LSA_Bellman2`,  first-order conditions are

.. math::
    :label: LSAx0

    V_x(x(s_1), s_1) = \Phi_0


for :math:`x(s_1), s_1 \in  {\cal S}`, and

.. math::
    :label: LSAn0

    \begin{aligned}
        (1 + \Phi_0) (u_{c,0} - u_{n,0})
        & + \Phi_0 \bigl[ n_0 (u_{ll,0} - u_{lc,0} ) +  (n_0 - g(s_0)) (u_{cc,0} - u_{cl,0}) \Bigr]
        \\
        & \quad \quad \quad -  \Phi_0 (u_{cc,0} - u_{cl,0}) b_0  =  0
    \end{aligned}


Notice  similarities and differences between the first-order
conditions for :math:`t \geq 1` and for :math:`t=0`.

An additional term is present in :eq:`LSAn0` except in  three special cases

* :math:`b_0 = 0`, or

* :math:`u_c` is constant (i.e., preferences are quasi-linear in consumption), or

* initial government assets are sufficiently large to finance all government
  purchases with interest earnings from those assets so that :math:`\Phi_0= 0`

Except in these special cases, the allocation and the labor tax rate as
functions of :math:`s_t` differ between dates :math:`t=0` and subsequent
dates :math:`t \geq 1`.

Naturally, the first-order conditions in this recursive formulation of the
Ramsey problem agree with the first-order conditions derived when we first
formulated the Ramsey plan in the space of sequences.



State Variable Degeneracy
^^^^^^^^^^^^^^^^^^^^^^^^^

Equations :eq:`LSAx0` and :eq:`LSAn0` imply that :math:`\Phi_0 = \Phi_1`
and that

.. math::
    :label: FONCx

    V_x(x_t, s_t) = \Phi_0


for all :math:`t \geq 1`.

When :math:`V` is concave in :math:`x`, this implies *state-variable degeneracy*
along a Ramsey plan in the sense that for :math:`t \geq 1`, :math:`x_t` will be
a time-invariant function of :math:`s_t`.

Given :math:`\Phi_0`, this function mapping :math:`s_t` into :math:`x_t`
can be expressed as a vector :math:`\vec x` that solves equation :eq:`Bellman2cons2`
for :math:`n` and :math:`c` as functions of :math:`g` that are associated
with :math:`\Phi = \Phi_0`.

Manifestations of Time Inconsistency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the marginal utility adjusted level of government debt :math:`x_t`
is a key state variable for the continuation Ramsey planners at
:math:`t \geq 1`, it is not a state variable at time :math:`0`.

The time :math:`0` Ramsey planner faces :math:`b_0`, not :math:`x_0 = u_{c,0} b_0`, as a state variable.

The discrepancy in state variables faced by the time :math:`0` Ramsey planner and the time
:math:`t \geq 1` continuation Ramsey planners captures the differing
obligations and incentives faced by the time :math:`0` Ramsey planner
and the time :math:`t \geq 1` continuation Ramsey planners.

* The time :math:`0` Ramsey planner is obligated to honor government
  debt :math:`b_0` measured in time :math:`0` consumption goods.

* The time :math:`0` Ramsey planner can manipulate the *value* of government
  debt as measured by :math:`u_{c,0} b_0`.

* In contrast, time :math:`t \geq 1` continuation Ramsey planners are
  obligated *not* to alter values of debt, as measured by
  :math:`u_{c,t} b_t`, that they inherit from a preceding Ramsey planner or
  continuation Ramsey planner.

When government expenditures :math:`g_t` are a time-invariant function
of a Markov state :math:`s_t`, a Ramsey plan and associated Ramsey
allocation feature marginal utilities of consumption :math:`u_c(s_t)`
that, given :math:`\Phi`, for :math:`t \geq 1` depend only on
:math:`s_t`, but that for :math:`t=0` depend on :math:`b_0` as well.

This means that :math:`u_c(s_t)` will be a time-invariant function of
:math:`s_t` for :math:`t \geq 1`, but except when :math:`b_0 = 0`, a
different function for :math:`t=0`.

This in turn means that prices of
one-period Arrow securities :math:`p_{t+1}(s_{t+1} | s_t) = p(s_{t+1}|s_t)`
will be the *same* time-invariant functions of :math:`(s_{t+1}, s_t)`
for :math:`t \geq 1`, but a different function :math:`p_0(s_1|s_0)` for
:math:`t=0`, except when :math:`b_0=0`.

The differences between these
time :math:`0` and time :math:`t \geq 1` objects reflect
the Ramsey planner’s incentive to manipulate Arrow security prices and,
through them, the value of initial government debt :math:`b_0`.

Recursive Implementation
^^^^^^^^^^^^^^^^^^^^^^^^



The above steps are implemented in a class called `RecursiveAllocation`

.. literalinclude:: /_static/lecture_specific/opt_tax_recur/recursive_allocation.py





Examples
========

Anticipated One-Period War
^^^^^^^^^^^^^^^^^^^^^^^^^^

This example illustrates in a simple setting how a Ramsey planner manages risk.

Government expenditures are known for sure in all periods except one

* For :math:`t<3` and :math:`t > 3` we assume that :math:`g_t = g_l = 0.1`.

* At :math:`t = 3` a war occurs with probability 0.5.

  *  If there is war, :math:`g_3 = g_h = 0.2`

  *  If there is no war :math:`g_3 = g_l = 0.1`

We define the components of the state vector as the following six  :math:`(t,g)`
pairs: :math:`(0,g_l),(1,g_l),(2,g_l),(3,g_l),(3,g_h), (t\geq 4,g_l)`.

We think of these  6 states as corresponding to :math:`s=1,2,3,4,5,6`.

The transition matrix is

.. math::

    \Pi = \left(\begin{matrix}0 & 1 & 0 & 0   & 0   & 0\\
                              0 & 0 & 1 & 0   & 0   & 0\\
                              0 & 0 & 0 & 0.5 & 0.5 & 0\\
                              0 & 0 & 0 & 0   & 0   & 1\\
                              0 & 0 & 0 & 0   & 0   & 1\\
                              0 & 0 & 0 & 0   & 0   & 1\end{matrix}\right)


Government expenditures at  each state are

.. math::

    g = \left(\begin{matrix} 0.1\\0.1\\0.1\\0.1\\0.2\\0.1 \end{matrix}\right)


We assume that the  representative agent has  utility function

.. math::

    u(c,n) = {\frac{c^{1-\sigma}}{1-\sigma}} - {\frac{n^{1+\gamma}}{1+\gamma}}


and set  :math:`\sigma = 2`, :math:`\gamma = 2`, and the  discount factor :math:`\beta = 0.9`.

Note: For convenience in terms of matching our code, we have expressed
utility as a function of :math:`n` rather than leisure :math:`l`.

This utility function is implemented in the class `CRRAutility`

.. literalinclude:: /_static/lecture_specific/opt_tax_recur/crra_utility.py

We set initial government debt :math:`b_0 = 1`.

We can now plot the Ramsey tax  under both realizations of time :math:`t = 3` government expenditures

* black when :math:`g_3 = .1`, and

* red when :math:`g_3 = .2`



.. code-block:: python3

    time_π = np.array([[0, 1, 0,   0,   0,  0],
                       [0, 0, 1,   0,   0,  0],
                       [0, 0, 0, 0.5, 0.5,  0],
                       [0, 0, 0,   0,   0,  1],
                       [0, 0, 0,   0,   0,  1],
                       [0, 0, 0,   0,   0,  1]])

    time_G = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1])
    # Θ can in principle be random
    time_Θ = np.ones(6)
    time_example = CRRAutility(π=time_π, G=time_G, Θ=time_Θ)

    # Solve sequential problem
    time_allocation = SequentialAllocation(time_example)
    sHist_h = np.array([0, 1, 2, 3, 5, 5, 5])
    sHist_l = np.array([0, 1, 2, 4, 5, 5, 5])
    sim_seq_h = time_allocation.simulate(1, 0, 7, sHist_h)
    sim_seq_l = time_allocation.simulate(1, 0, 7, sHist_l)

    # Government spending paths
    sim_seq_l[4] = time_example.G[sHist_l]
    sim_seq_h[4] = time_example.G[sHist_h]

    # Output paths
    sim_seq_l[5] = time_example.Θ[sHist_l] * sim_seq_l[1]
    sim_seq_h[5] = time_example.Θ[sHist_h] * sim_seq_h[1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    titles = ['Consumption', 'Labor Supply', 'Government Debt',
              'Tax Rate', 'Government Spending', 'Output']

    for ax, title, sim_l, sim_h in zip(axes.flatten(),
            titles, sim_seq_l, sim_seq_h):
        ax.set(title=title)
        ax.plot(sim_l, '-ok', sim_h, '-or', alpha=0.7)
        ax.grid()

    plt.tight_layout()
    plt.show()




**Tax smoothing**

* the tax rate is constant for all  :math:`t\geq 1`

      * For :math:`t \geq 1, t \neq 3`, this is a consequence of :math:`g_t`
        being the same at all those dates.

      * For :math:`t = 3`, it is a consequence of the special one-period utility
        function that we have assumed.

      * Under  other one-period utility functions, the time :math:`t=3` tax rate
        could be either higher or lower than for dates :math:`t \geq 1, t \neq 3`.

* the tax rate is the same at :math:`t=3` for both the high :math:`g_t` outcome and the low :math:`g_t` outcome

We have assumed that at :math:`t=0`, the government owes positive debt :math:`b_0`.

It sets the time :math:`t=0` tax rate partly with an eye to reducing the value :math:`u_{c,0} b_0` of :math:`b_0`.

It does this by increasing consumption at time :math:`t=0` relative to
consumption in later periods.

This has the consequence of *raising* the time :math:`t=0` value of the  gross
interest  rate for risk-free loans between periods :math:`t` and :math:`t+1`, which  equals

.. math::

    R_t = \frac{u_{c,t}}{\beta\mathbb E_{t}[u_{c,t+1}]}


A tax policy that makes  time :math:`t=0` consumption be  higher than time :math:`t=1` consumption evidently increases the
risk-free rate one-period interest rate, :math:`R_t`, at :math:`t=0`.

Raising the time :math:`t=0` risk-free interest rate makes time :math:`t=0`
consumption goods cheaper relative to consumption goods at later dates, thereby
lowering the value :math:`u_{c,0} b_0` of initial government debt :math:`b_0`.

We see this in a figure below that plots the time path for the risk-free interest
rate under both realizations of the time :math:`t=3` government expenditure shock.

The following plot illustrates how the government lowers the interest rate at
time 0 by raising consumption



.. code-block:: python3

    fix, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('Gross Interest Rate')
    ax.plot(sim_seq_l[-1], '-ok', sim_seq_h[-1], '-or', alpha=0.7)
    ax.grid()
    plt.show()



Government Saving
^^^^^^^^^^^^^^^^^

At time  :math:`t=0` the government evidently *dissaves* since :math:`b_1> b_0`.

    * This is a consequence of it setting a *lower* tax rate at :math:`t=0`,
      implying more consumption at :math:`t=0`.

At time :math:`t=1`, the government evidently *saves* since it has set the tax
rate sufficiently high to allow it to set :math:`b_2 < b_1`.

    * Its motive for doing this is that it anticipates a likely war at :math:`t=3`.

At time :math:`t=2` the government trades state-contingent Arrow securities
to hedge against  war at :math:`t=3`.

   * It purchases a security that pays off when :math:`g_3 = g_h`.

   * It sells a security that  pays off when :math:`g_3 = g_l`.

   * These purchases are designed in such a way that regardless of whether or
     not there is a war at :math:`t=3`, the government will begin  period
     :math:`t=4` with the *same* government debt.

   * The time :math:`t=4` debt level can be serviced with revenues from the
     constant tax rate set at times :math:`t\geq 1`.

At times :math:`t \geq 4` the government rolls over its debt, knowing that the
tax rate is set at level required to service the interest payments
on the debt and government expenditures.

Time 0 Manipulation of Interest Rate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have seen that when :math:`b_0>0`, the Ramsey plan sets the time :math:`t=0`
tax rate partly with an eye toward raising a risk-free interest
rate for one-period loans between times :math:`t=0` and :math:`t=1`.

By raising this interest rate, the plan makes time :math:`t=0` goods cheap
relative to consumption goods at later times.

By doing this, it lowers the value of time :math:`t=0` debt that it has inherited
and must finance.


Time 0 and Time-Inconsistency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the  preceding  example,  the Ramsey tax rate at time 0 differs from its value  at time 1.

To explore what is going on here, let's simplify things by removing  the possibility of war at time :math:`t=3`.

The Ramsey problem then includes no randomness because  :math:`g_t = g_l` for all :math:`t`.

The figure below plots the Ramsey tax rates and gross interest rates at time
:math:`t=0` and time :math:`t\geq1` as functions of the initial government debt
(using the sequential allocation solution and a CRRA utility function defined
above)



.. code-block:: python3

    tax_sequence = SequentialAllocation(CRRAutility(G=0.15,
                                                    π=np.ones((1, 1)),
                                                    Θ=np.ones(1)))

    n = 100
    tax_policy = np.empty((n, 2))
    interest_rate = np.empty((n, 2))
    gov_debt = np.linspace(-1.5, 1, n)

    for i in range(n):
        tax_policy[i] = tax_sequence.simulate(gov_debt[i], 0, 2)[3]
        interest_rate[i] = tax_sequence.simulate(gov_debt[i], 0, 3)[-1]

    fig, axes = plt.subplots(2, 1, figsize=(10,8), sharex=True)
    titles = ['Tax Rate', 'Gross Interest Rate']

    for ax, title, plot in zip(axes, titles, [tax_policy, interest_rate]):
        ax.plot(gov_debt, plot[:, 0], gov_debt, plot[:, 1], lw=2)
        ax.set(title=title, xlim=(min(gov_debt), max(gov_debt)))
        ax.grid()

    axes[0].legend(('Time $t=0$', 'Time $t \geq 1$'))
    axes[1].set_xlabel('Initial Government Debt')

    fig.tight_layout()
    plt.show()



The figure indicates  that if the government enters with  positive debt, it sets
a tax rate at :math:`t=0` that is less than all later tax rates.

By setting  a lower tax rate at :math:`t = 0`, the government raises consumption,
which reduces the *value* :math:`u_{c,0} b_0` of its  initial debt.

It does this by increasing :math:`c_0` and thereby lowering :math:`u_{c,0}`.

Conversely, if  :math:`b_{0} < 0`, the Ramsey planner sets the tax rate at :math:`t=0`
higher than in  subsequent periods.

A side effect of lowering time :math:`t=0` consumption  is that it raises  the
one-period  interest rate at time 0 above that of subsequent periods.

There are only two values of initial government debt at which  the tax rate is
constant for all :math:`t \geq 0`.

The first is :math:`b_{0} = 0`

   * Here the government can't use the :math:`t=0` tax rate  to alter  the
     value of the initial debt.

The second occurs when the government enters with sufficiently large  assets
that the Ramsey planner can achieve first best and sets :math:`\tau_t = 0`
for all :math:`t`.

It is only for these two values of initial government debt that the Ramsey
plan is time-consistent.

Another way of saying this is that, except for these two values of initial
government debt, a continuation of a Ramsey plan is not a Ramsey plan.

To illustrate this, consider a Ramsey planner who starts with an initial
government debt  :math:`b_1` associated with one of the Ramsey plans computed above.

Call :math:`\tau_1^R` the time :math:`t=0` tax rate chosen by the Ramsey planner
confronting this value for initial government debt government.

The figure below shows both  the tax rate  at time 1 chosen by our original
Ramsey planner and what a new Ramsey planner would choose for its
time :math:`t=0` tax rate



.. code-block:: python3

    tax_sequence = SequentialAllocation(CRRAutility(G=0.15,
                                                    π=np.ones((1, 1)),
                                                    Θ=np.ones(1)))

    n = 100
    tax_policy = np.empty((n, 2))
    τ_reset = np.empty((n, 2))
    gov_debt = np.linspace(-1.5, 1, n)

    for i in range(n):
        tax_policy[i] = tax_sequence.simulate(gov_debt[i], 0, 2)[3]
        τ_reset[i] = tax_sequence.simulate(gov_debt[i], 0, 1)[3]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gov_debt, tax_policy[:, 1], gov_debt, τ_reset, lw=2)
    ax.set(xlabel='Initial Government Debt', title='Tax Rate',
           xlim=(min(gov_debt), max(gov_debt)))
    ax.legend((r'$\tau_1$', r'$\tau_1^R$'))
    ax.grid()

    fig.tight_layout()
    plt.show()



The tax rates in the figure are equal  for only two values of initial government debt.



Tax Smoothing and non-CRRA Preferences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The complete tax smoothing for :math:`t \geq 1` in the preceding example is a
consequence of our having assumed CRRA preferences.

To see what is driving this outcome, we begin by noting that the Ramsey tax rate for :math:`t\geq 1` is
a time-invariant function :math:`\tau(\Phi,g)` of the Lagrange multiplier on
the implementability constraint and government expenditures.

For CRRA preferences, we can exploit the relations :math:`U_{cc}c = -\sigma U_c`
and :math:`U_{nn} n = \gamma U_n` to derive

.. math::

    \frac{(1+(1-\sigma)\Phi)U_c}{(1+(1-\gamma)\Phi)U_n} = 1


from the first-order conditions.

This equation immediately implies that the tax rate is constant.

For other  preferences, the tax rate may not be constant.

For example, let the period utility function be

.. math::

    u(c,n) = \log(c) + 0.69 \log(1-n)

We will create a new class `LogUtility` to represent this utility function

.. literalinclude:: /_static/lecture_specific/opt_tax_recur/log_utility.py

Also, suppose that :math:`g_t` follows a two-state IID process with equal
probabilities attached to :math:`g_l` and :math:`g_h`.

To compute the tax rate, we will use both the sequential and recursive approaches described above.

The figure below plots a sample path of the Ramsey tax rate



.. code-block:: python3

    log_example = LogUtility()
    # Solve sequential problem
    seq_log = SequentialAllocation(log_example)

    # Initialize grid for value function iteration and solve
    μ_grid = np.linspace(-0.6, 0.0, 200)
    # Solve recursive problem
    bel_log = RecursiveAllocation(log_example, μ_grid)

    T = 20
    sHist = np.array([0, 0, 0, 0, 0, 0, 0,
                      0, 1, 1, 0, 0, 0, 1,
                      1, 1, 1, 1, 1, 0])

    # Simulate
    sim_seq = seq_log.simulate(0.5, 0, T, sHist)
    sim_bel = bel_log.simulate(0.5, 0, T, sHist)

    # Government spending paths
    sim_seq[4] = log_example.G[sHist]
    sim_bel[4] = log_example.G[sHist]

    # Output paths
    sim_seq[5] = log_example.Θ[sHist] * sim_seq[1]
    sim_bel[5] = log_example.Θ[sHist] * sim_bel[1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    titles = ['Consumption', 'Labor Supply', 'Government Debt',
              'Tax Rate', 'Government Spending', 'Output']

    for ax, title, sim_s, sim_b in zip(axes.flatten(), titles, sim_seq, sim_bel):
        ax.plot(sim_s, '-ob', sim_b, '-xk', alpha=0.7)
        ax.set(title=title)
        ax.grid()

    axes.flatten()[0].legend(('Sequential', 'Recursive'))
    fig.tight_layout()
    plt.show()



As should be expected, the recursive and sequential solutions produce almost
identical allocations.

Unlike outcomes with CRRA preferences, the tax rate is not perfectly smoothed.

Instead, the government raises the tax rate when :math:`g_t` is high.


Further Comments
^^^^^^^^^^^^^^^^

A :doc:`related lecture <amss>` describes an extension of the Lucas-Stokey model
by  Aiyagari, Marcet, Sargent, and Seppälä (2002) :cite:`aiyagari2002optimal`.

In th AMSS economy, only a risk-free bond is traded.

That lecture compares the recursive representation of the Lucas-Stokey model
presented in this lecture with  one for an AMSS economy.

By comparing these recursive formulations, we shall glean a sense in which the
dimension of the state is lower in the Lucas Stokey model.

Accompanying that difference in dimension will be  different dynamics of government debt.
