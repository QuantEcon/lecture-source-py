.. _opt_tax_AMSS:

.. include:: /_static/includes/header.raw

*****************************************************
Optimal Taxation without State-Contingent Debt
*****************************************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
==========

Let's start with following imports

.. code-block:: ipython

    import numpy as np
    from scipy.optimize import root, fmin_slsqp
    from scipy.interpolate import UnivariateSpline
    from quantecon import MarkovChain
    import matplotlib.pyplot as plt
    %matplotlib inline


In :doc:`an earlier lecture<opt_tax_recur>`, we described a model of
optimal taxation with state-contingent debt due to
Robert E. Lucas, Jr.,  and Nancy Stokey  :cite:`LucasStokey1983`.

Aiyagari, Marcet, Sargent, and Seppälä :cite:`aiyagari2002optimal`  (hereafter, AMSS)
studied optimal taxation in a model without state-contingent debt.

In this lecture, we

* describe assumptions and equilibrium concepts

* solve the model

* implement the model numerically

* conduct some policy experiments

* compare outcomes with those in a corresponding complete-markets model

We begin with an introduction to the model.




Competitive Equilibrium with Distorting Taxes
===================================================


Many but not all features of the economy are identical to those of :doc:`the Lucas-Stokey economy <opt_tax_recur>`.

Let's start with things that are identical.

For :math:`t \geq 0`, a history of the state is represented by :math:`s^t = [s_t, s_{t-1}, \ldots, s_0]`.

Government purchases :math:`g(s)` are an exact time-invariant function of :math:`s`.

Let :math:`c_t(s^t)`,  :math:`\ell_t(s^t)`, and :math:`n_t(s^t)` denote consumption,
leisure, and labor supply, respectively, at history :math:`s^t` at time :math:`t`.


Each period a representative  household is endowed with one unit of time that can be divided between  leisure
:math:`\ell_t` and labor :math:`n_t`:

.. math::
    :label: feas1_amss

    n_t(s^t) + \ell_t(s^t) = 1


Output equals :math:`n_t(s^t)` and can be divided between consumption :math:`c_t(s^t)` and :math:`g(s_t)`

.. math::
    :label: TSs_techr_amss

    c_t(s^t) + g(s_t) = n_t(s^t)


Output is not storable.


The technology pins down a pre-tax wage rate to unity for all :math:`t, s^t`.

A representative  household’s preferences over :math:`\{c_t(s^t), \ell_t(s^t)\}_{t=0}^\infty` are ordered by

.. math::
    :label: TS_prefr_amss

    \sum_{t=0}^\infty \sum_{s^t} \beta^t \pi_t(s^t) u[c_t(s^t), \ell_t(s^t)]


where

* :math:`\pi_t(s^t)` is a joint probability distribution over the sequence :math:`s^t`, and

* the utility function :math:`u` is  increasing, strictly concave, and three times  continuously differentiable in both arguments.


The government imposes a flat rate tax :math:`\tau_t(s^t)` on labor income at time :math:`t`, history :math:`s^t`.

Lucas and Stokey assumed that there are complete markets in one-period Arrow securities; also see :doc:`smoothing models<smoothing>`.

It is at this point that AMSS :cite:`aiyagari2002optimal` modify the Lucas and Stokey economy.

AMSS allow the government to issue only one-period risk-free debt each period.

Ruling out complete markets in this way is a step in the direction of making total tax collections behave more like that prescribed in :cite:`Barro1979` than they do in :cite:`LucasStokey1983`.


Risk-free One-Period Debt Only
--------------------------------

In period :math:`t` and history :math:`s^t`, let

* :math:`b_{t+1}(s^t)` be the amount of the time :math:`t+1` consumption good that at time :math:`t` the government promised to pay

* :math:`R_t(s^t)` be the gross interest rate on  risk-free one-period debt between periods :math:`t` and :math:`t+1`

* :math:`T_t(s^t)` be a non-negative lump-sum transfer to the representative household [#fn_a]_

That :math:`b_{t+1}(s^t)` is the same for all realizations of :math:`s_{t+1}` captures its *risk-free* character.

The market value at time :math:`t` of government debt maturing at time :math:`t+1` equals :math:`b_{t+1}(s^t)` divided by :math:`R_t(s^t)`.

The government’s budget constraint in period :math:`t` at history :math:`s^t` is

.. math::
    :label: TS_gov_wo

    \begin{aligned}
    b_t(s^{t-1})
        & =    \tau^n_t(s^t) n_t(s^t) - g_t(s_t) - T_t(s^t) +
                       {b_{t+1}(s^t) \over R_t(s^t )}
        \\
        & \equiv z(s^t) + {b_{t+1}(s^t) \over R_t(s^t )},
    \end{aligned}


where :math:`z(s^t)` is the net-of-interest government surplus.

To rule out Ponzi schemes, we assume that the government is subject to a **natural debt limit** (to be discussed in a forthcoming lecture).

The consumption Euler equation for a representative household able to trade only one-period risk-free debt
with one-period gross interest rate :math:`R_t(s^t)` is

.. math::

    {1 \over R_t(s^t)}
    = \sum_{s^{t+1}\vert s^t} \beta  \pi_{t+1}(s^{t+1} | s^t)
                            { u_c(s^{t+1}) \over u_c(s^{t}) }


Substituting this expression into the government’s budget constraint :eq:`TS_gov_wo`
yields:

.. math::
    :label: TS_gov_wo2

    b_t(s^{t-1}) =  z(s^t) + \beta  \sum_{s^{t+1}\vert s^t}  \pi_{t+1}(s^{t+1} | s^t)
                           { u_c(s^{t+1}) \over u_c(s^{t}) } \; b_{t+1}(s^t)

Components of :math:`z(s^t)` on the right side depend on :math:`s^t`, but the left side is required to depend on :math:`s^{t-1}` only.

**This is what it means for one-period government debt to be risk-free**.

Therefore, the sum on the right side of equation :eq:`TS_gov_wo2` also has to depend only on :math:`s^{t-1}`.

This requirement will give rise to **measurability constraints** on the Ramsey allocation to be discussed soon.

If we replace :math:`b_{t+1}(s^t)` on the right side of equation :eq:`TS_gov_wo2` by the right
side of next period’s budget constraint (associated with a
particular realization :math:`s_{t}`) we get

.. math::

    b_t(s^{t-1}) =  z(s^t) + \sum_{s^{t+1}\vert s^t} \beta  \pi_{t+1}(s^{t+1} | s^t)
                           { u_c(s^{t+1}) \over u_c(s^{t}) }
    \, \left[z(s^{t+1}) + {b_{t+2}(s^{t+1}) \over R_{t+1}(s^{t+1})}\right]


After making similar repeated substitutions for all future occurrences of
government indebtedness, and by invoking the natural debt limit, we
arrive at:

.. math::
    :label: TS_gov_wo3

    \begin{aligned}
    b_t(s^{t-1})
        &=  \sum_{j=0}^\infty \sum_{s^{t+j} | s^t} \beta^j  \pi_{t+j}(s^{t+j} | s^t)
                  { u_c(s^{t+j}) \over u_c(s^{t}) } \;z(s^{t+j})
            \end{aligned}


Now let's

* substitute the resource constraint into the net-of-interest government surplus, and

* use the household’s first-order condition :math:`1-\tau^n_t(s^t)= u_{\ell}(s^t) /u_c(s^t)` to eliminate the labor tax rate

so that we can express the net-of-interest government surplus :math:`z(s^t)` as

.. math::
    :label: AMSS_44_2

    z(s^t)
        = \left[1 - {u_{\ell}(s^t) \over u_c(s^t)}\right] \left[c_t(s^t)+g_t(s_t)\right]
            -g_t(s_t) - T_t(s^t)\,.

If we substitute the  appropriate versions of the right side of :eq:`AMSS_44_2` for :math:`z(s^{t+j})` into equation :eq:`TS_gov_wo3`,
we obtain a sequence of *implementability constraints* on a Ramsey allocation in an AMSS economy.

Expression :eq:`TS_gov_wo3` at time :math:`t=0` and initial state :math:`s^0`
was also  an *implementability constraint* on a Ramsey allocation in a Lucas-Stokey economy:

.. math::
    :label: TS_gov_wo4

    b_0(s^{-1}) = \mathbb E_0 \sum_{j=0}^\infty \beta^j
                   { u_c(s^{j}) \over u_c(s^{0}) } \;z(s^{j})

Indeed, it was the *only* implementability constraint there.

But now we also have a large number of additional implementability constraints

.. math::
    :label: TS_gov_wo4a

     b_t(s^{t-1}) =  \mathbb E_t \sum_{j=0}^\infty \beta^j
                  { u_c(s^{t+j}) \over u_c(s^{t}) } \;z(s^{t+j})

Equation :eq:`TS_gov_wo4a` must hold for each :math:`s^t` for each :math:`t \geq 1`.

Comparison with Lucas-Stokey Economy
-------------------------------------

The expression on the right side of :eq:`TS_gov_wo4a` in the Lucas-Stokey (1983) economy would  equal the present value of a continuation stream of government surpluses evaluated at what would be competitive equilibrium Arrow-Debreu prices at date :math:`t`.

In the Lucas-Stokey economy, that present value is measurable with respect to :math:`s^t`.

In the AMSS economy, the restriction that government debt be risk-free imposes that that same present value must be measurable with respect to :math:`s^{t-1}`.

In a language used in the literature on incomplete markets models, it can be said that the AMSS model requires that at each :math:`(t, s^t)` what would be the present value of continuation government surpluses in the Lucas-Stokey model must belong to  the **marketable subspace** of the AMSS model.


Ramsey Problem Without State-contingent Debt
-----------------------------------------------

After we have substituted the resource constraint into the utility function, we can express the Ramsey problem as being to choose an allocation that solves

.. math::

    \max_{\{c_t(s^t),b_{t+1}(s^t)\}}
    \mathbb E_0 \sum_{t=0}^\infty \beta^t
                            u\left(c_t(s^t),1-c_t(s^t)-g_t(s_t)\right)


where the maximization is subject to

.. math::
    :label: AMSS_44

    \mathbb E_{0} \sum_{j=0}^\infty \beta^j
          { u_c(s^{j}) \over u_c(s^{0}) } \;z(s^{j}) \geq b_0(s^{-1})


and

.. math::
    :label: AMSS_46

    \mathbb E_{t} \sum_{j=0}^\infty \beta^j
        { u_c(s^{t+j}) \over u_c(s^{t}) } \;
        z(s^{t+j}) = b_t(s^{t-1})
          \quad \forall \,  s^t


given :math:`b_0(s^{-1})`.



Lagrangian Formulation
^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\gamma_0(s^0)` be a non-negative Lagrange multiplier on constraint :eq:`AMSS_44`.

As in the Lucas-Stokey economy, this multiplier is strictly positive when the government must resort to
distortionary taxation; otherwise it equals zero.

A consequence of the assumption that there are no markets in state-contingent securities  and that a market exists only in a risk-free security is that we have to attach stochastic processes :math:`\{\gamma_t(s^t)\}_{t=1}^\infty` of
Lagrange multipliers to the implementability constraints :eq:`AMSS_46`.

Depending on how the constraints  bind, these multipliers can be positive or negative:

.. math::

    \begin{aligned}
       \gamma_t(s^t)
       &\;\geq\; (\leq)\;\, 0 \quad \text{if the constraint binds in this direction }
       \\
       & \mathbb E_{t} \sum_{j=0}^\infty \beta^j
        { u_c(s^{t+j}) \over u_c(s^{t}) } \;z(s^{t+j}) \;\geq \;(\leq)\;\, b_t(s^{t-1})
    \end{aligned}


A negative multiplier :math:`\gamma_t(s^t)<0` means that if we could
relax constraint :eq:`AMSS_46`, we would like to *increase* the beginning-of-period
indebtedness for that particular realization of history :math:`s^t`.

That would let us reduce the beginning-of-period indebtedness for some other history [#fn_b]_.

These features flow from  the fact that the government cannot use state-contingent debt and therefore cannot allocate its indebtedness  efficiently across future states.

Some Calculations
------------------


It is helpful to apply two transformations to the Lagrangian.

Multiply constraint :eq:`AMSS_44` by :math:`u_c(s^0)` and the constraints :eq:`AMSS_46` by :math:`\beta^t u_c(s^{t})`.

Then a Lagrangian for the Ramsey problem can  be represented as

.. math::
    :label: AMSS_lagr;a

    \begin{aligned}
       J &= \mathbb E_{0} \sum_{t=0}^\infty \beta^t
                            \biggl\{ u\left(c_t(s^t), 1-c_t(s^t)-g_t(s_t)\right)\\
       &  \qquad + \gamma_t(s^t) \Bigl[ \mathbb E_{t} \sum_{j=0}^\infty \beta^j
             u_c(s^{t+j}) \,z(s^{t+j}) - u_c(s^{t}) \,b_t(s^{t-1}) \biggr\}
             \\
       &= \mathbb E_{0} \sum_{t=0}^\infty \beta^t
                             \biggl\{ u\left(c_t(s^t), 1-c_t(s^t)-g_t(s_t)\right)
            \\
       &  \qquad + \Psi_t(s^t)\, u_c(s^{t}) \,z(s^{t}) -
                       \gamma_t(s^t)\, u_c(s^{t}) \, b_t(s^{t-1})  \biggr\}
    \end{aligned}


where

.. math::
    :label: AMSS_lagr;

    \Psi_t(s^t)=\Psi_{t-1}(s^{t-1})+\gamma_t(s^t)
     \quad \text{and} \quad
    \Psi_{-1}(s^{-1})=0


In :eq:`AMSS_lagr;a`,  the second equality uses  the law of iterated expectations
and Abel’s summation formula (also called *summation by parts*, see
`this page <https://en.wikipedia.org/wiki/Abel%27s_summation_formula>`__).


First-order conditions with respect
to :math:`c_t(s^t)` can be expressed as

.. math::
    :label: AMSS_foc;a

    \begin{aligned}
      u_c(s^t)-u_{\ell}(s^t) &+ \Psi_t(s^t)\left\{ \left[
        u_{cc}(s^t) - u_{c\ell}(s^{t})\right]z(s^{t}) +
        u_{c}(s^{t})\,z_c(s^{t}) \right\}
        \\
        & \hspace{35mm} - \gamma_t(s^t)\left[
        u_{cc}(s^{t}) - u_{c\ell}(s^{t})\right]b_t(s^{t-1}) =0
    \end{aligned}


and with respect to :math:`b_t(s^t)` as

.. math::
    :label: AMSS_foc;b

    \mathbb E_{t} \left[\gamma_{t+1}(s^{t+1})\,u_c(s^{t+1})\right] = 0


If we substitute :math:`z(s^t)` from :eq:`AMSS_44_2` and its derivative
:math:`z_c(s^t)` into the first-order condition :eq:`AMSS_foc;a`, we  find  two
differences from the corresponding condition for the optimal allocation
in a Lucas-Stokey economy with state-contingent government debt.

1. The term involving :math:`b_t(s^{t-1})` in the first-order condition
   :eq:`AMSS_foc;a` does not appear in the corresponding expression
   for the Lucas-Stokey economy.

    * This term reflects the constraint that
      beginning-of-period government indebtedness must be the same across all
      realizations of next period’s state, a constraint that would  not be present if
      government debt could be state contingent.

2. The Lagrange multiplier :math:`\Psi_t(s^t)` in the first-order condition
   :eq:`AMSS_foc;a` may change over time in response to realizations of the state,
   while the multiplier :math:`\Phi` in the Lucas-Stokey economy is time-invariant.


We need some code from our :doc:`an earlier lecture <opt_tax_recur>`
on optimal taxation with state-contingent debt  sequential allocation implementation:

.. literalinclude:: /_static/lecture_specific/opt_tax_recur/sequential_allocation.py
    :class: collapse


To analyze the AMSS model, we find it useful to adopt a recursive formulation
using techniques like those in our lectures on :doc:`dynamic Stackelberg models <dyn_stack>` and :doc:`optimal taxation with state-contingent debt <opt_tax_recur>`.




Recursive Version of AMSS Model
==================================

We now describe a recursive formulation of the AMSS economy.

We have noted that from the point of view of the Ramsey planner, the restriction
to one-period risk-free securities

* leaves intact the single implementability constraint on allocations
  :eq:`TS_gov_wo4` from the Lucas-Stokey economy, but

* adds measurability constraints :eq:`TS_gov_wo3` on functions of tails of
  allocations at each time and history

We now explore how these constraints alter  Bellman equations for a time
:math:`0` Ramsey planner and for time :math:`t \geq 1`, history :math:`s^t`
continuation Ramsey planners.



Recasting State Variables
--------------------------

In the AMSS setting, the government faces a sequence of budget constraints

.. math::

    \tau_t(s^t) n_t(s^t) + T_t(s^t) +  b_{t+1}(s^t)/ R_t (s^t) =  g_t + b_t(s^{t-1})


where :math:`R_t(s^t)` is the gross risk-free rate of interest between :math:`t`
and :math:`t+1` at history :math:`s^t` and :math:`T_t(s^t)` are non-negative transfers.

Throughout this lecture, we shall set transfers to zero (for some issues about the limiting behavior of debt, this makes a possibly
important  difference from AMSS :cite:`aiyagari2002optimal`, who restricted transfers
to be non-negative).

In this case, the household faces a sequence of budget constraints

.. math::
    :label: eqn:AMSSapp1

    b_t(s^{t-1}) + (1-\tau_t(s^t)) n_t(s^t) = c_t(s^t) + b_{t+1}(s^t)/R_t(s^t)


The household’s first-order conditions are :math:`u_{c,t} = \beta R_t \mathbb E_t u_{c,t+1}`
and :math:`(1-\tau_t) u_{c,t} = u_{l,t}`.

Using these to eliminate :math:`R_t` and :math:`\tau_t` from  budget constraint
:eq:`eqn:AMSSapp1` gives

.. math::
    :label: eqn:AMSSapp2a

    b_t(s^{t-1}) + \frac{u_{l,t}(s^t)}{u_{c,t}(s^t)} n_t(s^t)
    = c_t(s^t) + {\frac{\beta (\mathbb E_t u_{c,t+1}) b_{t+1}(s^t)}{u_{c,t}(s^t)}}


or

.. math::
    :label: eqn:AMSSapp2

    u_{c,t}(s^t) b_t(s^{t-1}) + u_{l,t}(s^t) n_t(s^t)
    = u_{c,t}(s^t) c_t(s^t) + \beta (\mathbb E_t u_{c,t+1}) b_{t+1}(s^t)


Now define

.. math::
    :label: eqn:AMSSapp3

    x_t \equiv \beta b_{t+1}(s^t) \mathbb E_t u_{c,t+1} = u_{c,t} (s^t) {\frac{b_{t+1}(s^t)}{R_t(s^t)}}


and represent the household’s budget constraint at time :math:`t`,
history :math:`s^t` as

.. math::
    :label: eqn:AMSSapp4

    {\frac{u_{c,t} x_{t-1}}{\beta \mathbb E_{t-1} u_{c,t}}} = u_{c,t} c_t - u_{l,t} n_t + x_t


for :math:`t \geq 1`.

Measurability Constraints
--------------------------

Write equation :eq:`eqn:AMSSapp2` as

.. math::
    :label: eqn:AMSSapp2b

    b_t(s^{t-1})  = c_t(s^t) -  { \frac{u_{l,t}(s^t)}{u_{c,t}(s^t)}} n_t(s^t) +
    {\frac{\beta (\mathbb E_t u_{c,t+1}) b_{t+1}(s^t)}{u_{c,t}}}


The right side of equation :eq:`eqn:AMSSapp2b` expresses the time :math:`t` value of government debt
in terms of a linear combination of terms whose individual components
are measurable with respect to :math:`s^t`.

The sum  of terms on the right side  of equation :eq:`eqn:AMSSapp2b` must equal
:math:`b_t(s^{t-1})`.

That implies that it has to be *measurable* with respect to :math:`s^{t-1}`.

Equations :eq:`eqn:AMSSapp2b` are the *measurability constraints* that the AMSS model adds to the single time :math:`0` implementation
constraint imposed in the Lucas and Stokey model.

Two Bellman Equations
----------------------

Let :math:`\Pi(s|s_-)` be a Markov transition matrix whose entries tell probabilities of moving from state :math:`s_-` to state :math:`s` in one period.

Let

* :math:`V(x_-, s_-)` be the continuation value of a continuation
  Ramsey plan at :math:`x_{t-1} = x_-, s_{t-1} =s_-` for :math:`t \geq 1`

* :math:`W(b, s)` be the value of the Ramsey plan at time :math:`0` at
  :math:`b_0=b` and :math:`s_0 = s`

We distinguish between two types of planners:

For :math:`t \geq 1`, the value function for a **continuation Ramsey planner**
satisfies the Bellman equation

.. math::
    :label: eqn:AMSSapp5

    V(x_-,s_-) = \max_{\{n(s), x(s)\}} \sum_s \Pi(s|s_-) \left[ u(n(s) -
    g(s), 1-n(s)) + \beta V(x(s),s) \right]


subject to the following collection of implementability constraints, one
for each :math:`s \in {\cal S}`:

.. math::
    :label: eqn:AMSSapp6

    {\frac{u_c(s) x_- }{\beta \sum_{\tilde s} \Pi(\tilde s|s_-) u_c(\tilde s) }}
    = u_c(s) (n(s) - g(s)) - u_l(s) n(s) + x(s)


A continuation Ramsey planner at :math:`t \geq 1` takes
:math:`(x_{t-1}, s_{t-1}) = (x_-, s_-)` as given and before
:math:`s` is realized chooses
:math:`(n_t(s_t), x_t(s_t)) = (n(s), x(s))` for :math:`s \in  {\cal S}`.


The **Ramsey planner** takes :math:`(b_0, s_0)` as given and chooses :math:`(n_0, x_0)`.

The value function :math:`W(b_0, s_0)`   for the time :math:`t=0` Ramsey planner
satisfies the Bellman equation

.. math::
    :label: eqn:AMSSapp100

    W(b_0, s_0) = \max_{n_0, x_0} u(n_0 - g_0, 1-n_0) + \beta V(x_0,s_0)

where maximization is subject to

.. math::
    :label: eqn:AMMSSapp101

    u_{c,0} b_0 = u_{c,0} (n_0-g_0) - u_{l,0} n_0 + x_0


Martingale Supercedes State-Variable Degeneracy
------------------------------------------------

Let :math:`\mu(s|s_-) \Pi(s|s_-)` be a Lagrange multiplier on the constraint :eq:`eqn:AMSSapp6`
for state :math:`s`.

After forming an appropriate Lagrangian, we find that the continuation Ramsey planner’s first-order
condition with respect to :math:`x(s)` is

.. math::
    :label: eqn:AMSSapp7

    \beta V_x(x(s),s) = \mu(s|s_-)


Applying the envelope theorem to Bellman equation :eq:`eqn:AMSSapp5` gives

.. math::
    :label: eqn:AMSSapp8

    V_x(x_-,s_-) = \sum_s \Pi(s|s_-) \mu(s|s_-) {\frac{u_c(s)}{\beta \sum_{\tilde s}
    \Pi(\tilde s|s_-) u_c(\tilde s) }}


Equations :eq:`eqn:AMSSapp7` and :eq:`eqn:AMSSapp8` imply that

.. math::
    :label: eqn:AMSSapp9

    V_x(x_-, s_-) = \sum_{s} \left( \Pi(s|s_-) {\frac{u_c(s)}{\sum_{\tilde s}
    \Pi(\tilde s| s_-) u_c(\tilde s)}} \right) V_x(x(s), s)


Equation :eq:`eqn:AMSSapp9` states that :math:`V_x(x, s)` is a *risk-adjusted martingale*.

Saying that :math:`V_x(x, s)` is a risk-adjusted martingale  means  that
:math:`V_x(x, s)`  is a martingale with respect to the probability distribution
over :math:`s^t` sequences that are generated by the *twisted* transition probability matrix:

.. math::

    \check \Pi(s|s_-) \equiv \Pi(s|s_-) {\frac{u_c(s)}{\sum_{\tilde s}
    \Pi(\tilde s| s_-) u_c(\tilde s)}}


**Exercise**: Please verify that :math:`\check \Pi(s|s_-)` is a valid Markov
transition density, i.e., that its elements are all non-negative and
that for each :math:`s_-`, the sum over :math:`s` equals unity.



Absence of State Variable Degeneracy
------------------------------------

Along a Ramsey plan, the state variable :math:`x_t = x_t(s^t, b_0)`
becomes a function of the history :math:`s^t` and initial
government debt :math:`b_0`.

In :doc:`Lucas-Stokey model<opt_tax_recur>`, we
found that

* a counterpart to :math:`V_x(x,s)` is time-invariant and equal to
  the Lagrange multiplier on the Lucas-Stokey implementability constraint

* time invariance of :math:`V_x(x,s)`  is the source of a key
  feature of the Lucas-Stokey model, namely, state variable degeneracy
  (i.e., :math:`x_t` is an exact function of :math:`s_t`)

That :math:`V_x(x,s)` varies over time according to a twisted martingale
means that there is no state-variable degeneracy in the AMSS model.

In the AMSS model, both :math:`x` and :math:`s` are needed to describe the state.

This property of the AMSS model  transmits a twisted martingale
component to consumption, employment, and the tax rate.




Digression on Non-negative Transfers
------------------------------------

Throughout this lecture, we have imposed that transfers :math:`T_t = 0`.

AMSS :cite:`aiyagari2002optimal` instead imposed a nonnegativity
constraint :math:`T_t\geq 0` on transfers.

They also considered a special case of quasi-linear preferences,
:math:`u(c,l)= c + H(l)`.

In this case, :math:`V_x(x,s)\leq 0` is a non-positive martingale.

By the *martingale convergence theorem*  :math:`V_x(x,s)` converges almost surely.

Furthermore, when the Markov chain :math:`\Pi(s| s_-)` and the government
expenditure function :math:`g(s)` are such that :math:`g_t` is perpetually
random, :math:`V_x(x, s)` almost surely converges to zero.

For quasi-linear preferences, the first-order condition with respect to :math:`n(s)` becomes

.. math::

    (1-\mu(s|s_-) ) (1 - u_l(s)) + \mu(s|s_-) n(s) u_{ll}(s) =0


When :math:`\mu(s|s_-) = \beta V_x(x(s),x)` converges to zero, in the limit
:math:`u_l(s)= 1 =u_c(s)`, so that :math:`\tau(x(s),s) =0`.

Thus, in the limit, if :math:`g_t` is perpetually random,  the government
accumulates sufficient assets to finance all expenditures from earnings on those
assets, returning any excess revenues to the household as non-negative lump-sum transfers.


Code
-----

The recursive formulation is implemented as follows

.. literalinclude:: /_static/lecture_specific/amss/recursive_allocation.py


Examples
================

We now turn to some examples.



We will first build some useful functions for solving the model

.. literalinclude:: /_static/lecture_specific/amss/utilities.py




Anticipated One-Period War
----------------------------------

In our lecture on :doc:`optimal taxation with state contingent debt <opt_tax_recur>`
we studied how the government manages uncertainty in a simple setting.

As in that lecture, we assume the one-period utility function

.. math::

    u(c,n) = {\frac{c^{1-\sigma}}{1-\sigma}} - {\frac{n^{1+\gamma}}{1+\gamma}}


.. note::
    For convenience in  matching our computer code, we have expressed
    utility as a function of :math:`n` rather than leisure :math:`l`.


We consider the same government expenditure process studied in the lecture on
:doc:`optimal taxation with state contingent debt <opt_tax_recur>`.

Government expenditures are known for sure in all periods except one.

* For :math:`t<3` or :math:`t > 3` we assume that :math:`g_t = g_l = 0.1`.

* At :math:`t = 3` a war occurs with probability 0.5.

  *  If there is war, :math:`g_3 = g_h = 0.2`.

  *  If there is no war :math:`g_3 = g_l = 0.1`.

A useful trick is to define  components of the state vector as the following six
:math:`(t,g)` pairs:

.. math::

    (0,g_l), (1,g_l), (2,g_l), (3,g_l), (3,g_h), (t\geq 4,g_l)


We think of these 6 states as corresponding to :math:`s=1,2,3,4,5,6`.

The transition matrix is

.. math::

    P = \begin{pmatrix}
      0 & 1 & 0 & 0   & 0   & 0\\
      0 & 0 & 1 & 0   & 0   & 0\\
      0 & 0 & 0 & 0.5 & 0.5 & 0\\
      0 & 0 & 0 & 0   & 0   & 1\\
      0 & 0 & 0 & 0   & 0   & 1\\
      0 & 0 & 0 & 0   & 0   & 1
    \end{pmatrix}


The government expenditure at  each state is

.. math::

    g = \left(\begin{matrix} 0.1\\0.1\\0.1\\0.1\\0.2\\0.1 \end{matrix}\right)


We assume the same utility parameters as in the :doc:`Lucas-Stokey economy <opt_tax_recur>`.

This utility function is implemented in the following class.

.. literalinclude:: /_static/lecture_specific/opt_tax_recur/crra_utility.py


The following figure plots the Ramsey plan under both complete and incomplete
markets for both possible realizations of the state at time :math:`t=3`.

Optimal policies when  the government has  access to state contingent debt are
represented by black lines, while the optimal policies when there is only a risk-free bond are in red.

Paths with circles are histories in which there is peace, while those with
triangle denote war.



.. code-block:: python3

    # Initialize μgrid for value function iteration
    μ_grid = np.linspace(-0.7, 0.01, 200)

    time_example = CRRAutility()

    time_example.π = np.array([[0, 1, 0,   0,   0,  0],
                               [0, 0, 1,   0,   0,  0],
                               [0, 0, 0, 0.5, 0.5,  0],
                               [0, 0, 0,   0,   0,  1],
                               [0, 0, 0,   0,   0,  1],
                               [0, 0, 0,   0,   0,  1]])

    time_example.G = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1])
    time_example.Θ = np.ones(6)  # Θ can in principle be random

    time_example.transfers = True                                  # Government can use transfers
    time_sequential = SequentialAllocation(time_example)           # Solve sequential problem
    time_bellman = RecursiveAllocationAMSS(time_example, μ_grid)   # Solve recursive problem

    sHist_h = np.array([0, 1, 2, 3, 5, 5, 5])
    sHist_l = np.array([0, 1, 2, 4, 5, 5, 5])

    sim_seq_h = time_sequential.simulate(1, 0, 7, sHist_h)
    sim_bel_h = time_bellman.simulate(1, 0, 7, sHist_h)
    sim_seq_l = time_sequential.simulate(1, 0, 7, sHist_l)
    sim_bel_l = time_bellman.simulate(1, 0, 7, sHist_l)

    # Government spending paths
    sim_seq_l[4] = time_example.G[sHist_l]
    sim_seq_h[4] = time_example.G[sHist_h]
    sim_bel_l[4] = time_example.G[sHist_l]
    sim_bel_h[4] = time_example.G[sHist_h]

    # Output paths
    sim_seq_l[5] = time_example.Θ[sHist_l] * sim_seq_l[1]
    sim_seq_h[5] = time_example.Θ[sHist_h] * sim_seq_h[1]
    sim_bel_l[5] = time_example.Θ[sHist_l] * sim_bel_l[1]
    sim_bel_h[5] = time_example.Θ[sHist_h] * sim_bel_h[1]


    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    titles = ['Consumption', 'Labor Supply', 'Government Debt',
              'Tax Rate', 'Government Spending', 'Output']

    for ax, title, sim_l, sim_h, bel_l, bel_h in zip(axes.flatten(), titles,
                                                     sim_seq_l, sim_seq_h,
                                                     sim_bel_l, sim_bel_h):
        ax.plot(sim_l, '-ok', sim_h, '-^k', bel_l, '-or', bel_h, '-^r', alpha=0.7)
        ax.set(title=title)
        ax.grid()

    plt.tight_layout()
    plt.show()




How a Ramsey planner responds to  war depends on the structure of the asset market.

If it is able to trade state-contingent debt, then at time :math:`t=2`

* the government purchases an Arrow security that pays off when :math:`g_3 = g_h`

* the government sells an Arrow security that  pays off when :math:`g_3 = g_l`

* These purchases are designed in such a way that regardless of whether or not there is a war at :math:`t=3`, the government will begin  period :math:`t=4` with the *same* government debt

This pattern facilities smoothing tax rates across  states.

The government without state contingent debt cannot do this.

Instead, it must enter   time :math:`t=3` with the same level of debt falling due whether there is peace or war at :math:`t=3`.

It responds to this constraint by smoothing tax rates across time.

To finance a war it raises taxes and issues more debt.

To service the additional debt burden, it raises taxes in all future periods.

The absence of state contingent debt leads to an important difference in the
optimal tax policy.

When the Ramsey planner has access to state contingent debt, the optimal tax
policy is history independent

* the tax rate is a function  of the current level of government spending only,
  given the Lagrange multiplier on the implementability constraint

Without state contingent debt, the optimal tax rate is history dependent.

* A war at time :math:`t=3` causes a permanent increase in the tax rate.



Perpetual War Alert
^^^^^^^^^^^^^^^^^^^^

History dependence occurs more dramatically in a case in which the government
perpetually faces the prospect  of war.

This case was studied in the final example of the lecture on
:doc:`optimal taxation with state-contingent debt <opt_tax_recur>`.

There, each period the government faces a constant probability, :math:`0.5`, of war.

In addition, this example features the following preferences

.. math::

    u(c,n) = \log(c) + 0.69 \log(1-n)

In accordance, we will re-define our utility function.

.. literalinclude:: /_static/lecture_specific/opt_tax_recur/log_utility.py


With these preferences, Ramsey tax rates will vary even in the Lucas-Stokey
model with state-contingent debt.

The figure below plots optimal tax policies for both the economy with
state contingent debt (circles) and the economy with only a risk-free bond
(triangles).



.. code-block:: python3

    log_example = LogUtility()
    log_example.transfers = True                        # Government can use transfers
    log_sequential = SequentialAllocation(log_example)  # Solve sequential problem
    log_bellman = RecursiveAllocationAMSS(log_example, μ_grid)

    T = 20
    sHist = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                      0, 0, 0, 1, 1, 1, 1, 1, 1, 0])

    # Simulate
    sim_seq = log_sequential.simulate(0.5, 0, T, sHist)
    sim_bel = log_bellman.simulate(0.5, 0, T, sHist)

    titles = ['Consumption', 'Labor Supply', 'Government Debt',
              'Tax Rate', 'Government Spending', 'Output']

    # Government spending paths
    sim_seq[4] = log_example.G[sHist]
    sim_bel[4] = log_example.G[sHist]

    # Output paths
    sim_seq[5] = log_example.Θ[sHist] * sim_seq[1]
    sim_bel[5] = log_example.Θ[sHist] * sim_bel[1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for ax, title, seq, bel in zip(axes.flatten(), titles, sim_seq, sim_bel):
        ax.plot(seq, '-ok', bel, '-^b')
        ax.set(title=title)
        ax.grid()

    axes[0, 0].legend(('Complete Markets', 'Incomplete Markets'))
    plt.tight_layout()
    plt.show()





When the government experiences a prolonged period of peace, it is able to reduce
government debt and set permanently lower tax rates.

However, the government  finances a long war by borrowing and raising taxes.

This results in a drift away from  policies with state contingent debt that
depends on the history of shocks.

This is even more evident in the following figure that plots the evolution of
the two policies over 200 periods.



.. code-block:: python3

    T = 200  # Set T to 200 periods
    sim_seq_long = log_sequential.simulate(0.5, 0, T)
    sHist_long = sim_seq_long[-3]
    sim_bel_long = log_bellman.simulate(0.5, 0, T, sHist_long)

    titles = ['Consumption', 'Labor Supply', 'Government Debt',
              'Tax Rate', 'Government Spending', 'Output']

    # Government spending paths
    sim_seq_long[4] = log_example.G[sHist_long]
    sim_bel_long[4] = log_example.G[sHist_long]

    # Output paths
    sim_seq_long[5] = log_example.Θ[sHist_long] * sim_seq_long[1]
    sim_bel_long[5] = log_example.Θ[sHist_long] * sim_bel_long[1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for ax, title, seq, bel in zip(axes.flatten(), titles, sim_seq_long, sim_bel_long):
        ax.plot(seq, '-k', bel, '-.b', alpha=0.5)
        ax.set(title=title)
        ax.grid()

    axes[0, 0].legend(('Complete Markets','Incomplete Markets'))
    plt.tight_layout()
    plt.show()





.. rubric:: Footnotes

.. [#fn_a]
    In an allocation that solves the Ramsey problem and that levies distorting
    taxes on labor, why would the government ever want to hand revenues back
    to the private sector? It would not in an economy with state-contingent debt, since
    any such allocation could be improved by lowering distortionary taxes
    rather than handing out lump-sum transfers. But, without state-contingent
    debt there can be circumstances when a government would like to make
    lump-sum transfers to the private sector.

.. [#fn_b]
    From the first-order conditions for the Ramsey
    problem, there exists another realization :math:`\tilde s^t` with
    the same history up until the previous period, i.e., :math:`\tilde s^{t-1}=
    s^{t-1}`, but where the multiplier on constraint :eq:`AMSS_46` takes  a positive value, so
    :math:`\gamma_t(\tilde s^t)>0`.
