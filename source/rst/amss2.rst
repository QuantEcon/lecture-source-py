.. _amss2:

.. include:: /_static/includes/header.raw

***********************************************************
Fluctuating Interest Rates Deliver Fiscal Insurance
***********************************************************


.. contents:: :depth: 2


**Co-authors: Anmol Bhandari and David Evans**

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
=============

This lecture extends our investigations of how optimal policies for levying a flat-rate tax on labor income and  issuing government debt depend
on whether there are complete  markets for debt.

A Ramsey allocation and Ramsey policy in the AMSS :cite:`aiyagari2002optimal` model described in :doc:`optimal taxation without state-contingent debt<amss>` generally differs
from a Ramsey allocation and Ramsey policy in the  Lucas-Stokey :cite:`LucasStokey1983` model described in :doc:`optimal taxation with state-contingent debt<opt_tax_recur>`.

This is because the implementability restriction that a competitive equilibrium with a distorting tax  imposes on  allocations in the Lucas-Stokey model is just one among a set of
implementability conditions imposed in  the AMSS model.

These additional constraints require that time :math:`t` components of a Ramsey allocation
for the AMSS model be **measurable** with respect to time :math:`t-1` information.

The  measurability constraints imposed by the AMSS model are inherited from the   restriction that  only one-period risk-free bonds
can be traded.

Differences between the  Ramsey allocations in the two models   indicate that at least some of the measurability constraints of the AMSS model of
:doc:`optimal taxation without state-contingent debt<amss>` are violated at the Ramsey allocation of a corresponding  :cite:`LucasStokey1983` model with state-contingent debt.

Another way to say this is that differences between the Ramsey allocations of the two models indicate that some of the measurability constraints of the
AMSS model are violated at the  Ramsey allocation of the Lucas-Stokey model.

Nonzero Lagrange multipliers on those constraints make the Ramsey allocation for the AMSS model differ from the Ramsey allocation for the Lucas-Stokey model.

This lecture studies a special  AMSS model in which

*  The exogenous state variable :math:`s_t` is governed by  a finite-state Markov chain.

*  With an arbitrary budget-feasible initial level of government debt, the measurability  constraints

   - bind for many periods, but :math:`\ldots`.

   - eventually, they stop binding evermore, so :math:`\ldots`.

   - in the tail of the Ramsey plan, the Lagrange multipliers :math:`\gamma_t(s^t)` on the AMSS implementability constraints :eq:`TS_gov_wo4`  converge to zero.

*  After the implementability constraints :eq:`TS_gov_wo4` no longer bind in the tail of the AMSS Ramsey plan

   - history dependence of the AMSS state variable :math:`x_t` vanishes and  :math:`x_t` becomes a time-invariant function of the Markov state :math:`s_t`.

   - the par value of government debt becomes **constant over time** so that :math:`b_{t+1}(s^t) = \bar b` for :math:`t \geq T` for a sufficiently large :math:`T`.

   - :math:`\bar b <0`, so that the tail of the Ramsey plan instructs  the government always to make a constant par value of risk-free one-period loans to the private sector.

   - the one-period gross interest rate :math:`R_t(s^t)` on risk-free debt  converges to a time-invariant function of the Markov state :math:`s_t`.

*  For a **particular** :math:`b_0 < 0` (i.e., a positive level of initial government **loans** to the private sector), the measurability constraints **never** bind.

* In this special case

   - the **par value** :math:`b_{t+1}(s_t) = \bar b`  of government debt at time :math:`t` and Markov state :math:`s_t`  is constant across time and states,
     but :math:`\ldots`.

   - the **market value** :math:`\frac{\bar b}{R_t(s_t)}` of government debt at time :math:`t`  varies as a time-invariant function of the Markov state :math:`s_t`.

   - fluctuations in the interest rate make gross earnings on government debt :math:`\frac{\bar b}{R_t(s_t)}` fully insure the gross-of-gross-interest-payments government budget against fluctuations in government expenditures.

   - the state variable :math:`x` in a recursive representation of a Ramsey plan is a time-invariant function of the Markov state for :math:`t \geq 0`.

*  In this special case, the Ramsey allocation in the AMSS model agrees with that in a :cite:`LucasStokey1983` model in which
   the same amount of state-contingent debt falls due in all states tomorrow

   - it is a situation in which  the Ramsey planner loses nothing from not being  able to  purchase state-contingent debt and being restricted to exchange only risk-free debt  debt.

* This outcome emerges only when we initialize government debt at a particular :math:`b_0 < 0`.


In a nutshell, the reason for this striking outcome is that at a particular level of risk-free government **assets**, fluctuations in the one-period risk-free interest
rate provide the government with complete insurance against stochastically varying government expenditures.

Let's start with some imports:

.. code-block:: ipython

    import matplotlib.pyplot as plt
    %matplotlib inline
    from scipy.optimize import fsolve, fmin


Forces at Work
===============

The forces  driving asymptotic  outcomes here are examples of dynamics present in a more general class  incomplete markets models analyzed in :cite:`BEGS1` (BEGS).

BEGS provide conditions under which government debt under a Ramsey plan converges to an invariant distribution.

BEGS  construct approximations to that asymptotically invariant  distribution  of government debt under a  Ramsey plan.

BEGS also compute an approximation to a Ramsey plan's rate of convergence  to that limiting invariant distribution.

We  shall use the BEGS approximating limiting distribution and the approximating  rate of convergence   to help interpret  outcomes here.

For a long time, the Ramsey plan puts a nontrivial martingale-like component into the par value of  government debt as part of the way that the Ramsey plan imperfectly
smooths distortions from the labor tax rate across  time and Markov states.

But BEGS show that binding implementability constraints slowly push government debt in a direction designed to let the government use fluctuations in equilibrium interest
rate  rather than fluctuations in  par values of debt to insure against shocks to government expenditures.

- This is a **weak** (but unrelenting) force that, starting from an initial debt level, for a long time is dominated by the stochastic martingale-like component of debt
  dynamics that the Ramsey planner uses to facilitate imperfect tax-smoothing across time and states.

- This weak force slowly drives the par value of government **assets** to a **constant** level at which the government can completely insure against government expenditure shocks while
  shutting down the stochastic component of debt dynamics.

- At that point, the tail of the par value of government debt becomes a trivial martingale: it is constant over time.



Logical Flow of Lecture
=========================

We present ideas  in the following order

* We describe a two-state  AMSS economy and generate a long simulation starting from a positive  initial government debt.

* We observe that in a long simulation starting from positive government debt, the par value of  government debt eventually converges to a constant :math:`\bar b`.

* In fact, the par value of government debt  converges to the same constant level :math:`\bar b` for alternative realizations of the Markov government expenditure process and for alternative settings of initial government
  debt :math:`b_0`.

* We reverse engineer a particular value of initial government debt :math:`b_0` (it turns out to be negative) for which the  continuation debt moves
  to :math:`\bar b` immediately.

* We note that for this particular initial debt :math:`b_0`, the Ramsey allocations  for the AMSS economy and the Lucas-Stokey model are identical

  - we verify that the LS Ramsey planner chooses to purchase **identical** claims to time :math:`t+1` consumption for all Markov states tomorrow for each Markov state today.

* We compute the BEGS approximations to check how accurately they describe the dynamics of the long-simulation.

Equations from Lucas-Stokey (1983) Model
------------------------------------------

Although we are studying an AMSS :cite:`aiyagari2002optimal` economy,  a Lucas-Stokey :cite:`LucasStokey1983` economy plays
an important  role in the reverse-engineering calculation to be described below.

For that reason, it is helpful  to have readily available some key equations underlying a Ramsey plan for the Lucas-Stokey economy.

Recall first-order conditions for a Ramsey allocation for the Lucas-Stokey economy.

For :math:`t \geq 1`, these take the form

.. math::
    :label: TS_barg10a

    \begin{aligned}
      (1+\Phi) &u_c(c,1-c-g) + \Phi \bigl[c u_{cc}(c,1-c-g) -
        (c+g) u_{\ell c}(c,1-c-g) \bigr]
        \\
        &= (1+\Phi) u_{\ell}(c,1-c-g) + \Phi \bigl[c u_{c\ell}(c,1-c-g) -
        (c+g) u_{\ell \ell}(c,1-c-g)  \bigr]
    \end{aligned}

There is one such equation for each value of the Markov state :math:`s_t`.

In addition, given an initial Markov state, the time :math:`t=0` quantities :math:`c_0` and :math:`b_0` satisfy

.. math::
    :label: TS_barg11b

    \begin{aligned}
          (1+\Phi) &u_c(c,1-c-g) + \Phi \bigl[c u_{cc}(c,1-c-g) -
            (c+g) u_{\ell c}(c,1-c-g) \bigr]
            \\
            &= (1+\Phi) u_{\ell}(c,1-c-g) + \Phi \bigl[c u_{c\ell}(c,1-c-g) -
            (c+g) u_{\ell \ell}(c,1-c-g)  \bigr] + \Phi (u_{cc} - u_{c,\ell}) b_0
    \end{aligned}

In addition, the time :math:`t=0` budget constraint is satisfied at :math:`c_0` and initial government debt
:math:`b_0`:

.. math::
    :label: eqn_AMSS2_10

    b_0 + g_0 = \tau_0 (c_0 + g_0) + \frac{\bar b}{R_0}

where :math:`R_0` is the gross interest rate for the Markov state :math:`s_0` that is assumed to prevail at time :math:`t =0`
and :math:`\tau_0` is the time :math:`t=0` tax rate.


In equation :eq:`eqn_AMSS2_10`, it is understood that

.. math::
    :nowrap:

    \begin{aligned}
    \tau_0 = 1 - \frac{u_{l,0}}{u_{c,0}} \\
    R_0^{-1} =  \beta  \sum_{s=1}^S \Pi(s | s_0) \frac{u_c(s)}{u_{c,0}}
    \end{aligned}


It is useful to transform  some of the above equations to forms that are more natural for analyzing the
case of a CRRA utility specification that we shall use in our example economies.

Specification with CRRA Utility
--------------------------------



As in lectures :doc:`optimal taxation without state-contingent debt<amss>` and :doc:`optimal taxation with state-contingent debt<opt_tax_recur>`,
we assume that the representative agent has utility function

.. math::

    u(c,n) = {\frac{c^{1-\sigma}}{1-\sigma}} - {\frac{n^{1+\gamma}}{1+\gamma}}


and set  :math:`\sigma = 2`, :math:`\gamma = 2`, and the  discount factor :math:`\beta = 0.9`.

We eliminate leisure from the model and continue to assume that

.. math::

    c_t + g_t = n_t

The analysis of Lucas and Stokey prevails once we make the following replacements

.. math::

        \begin{aligned}
        u_\ell(c, \ell) &\sim - u_n(c, n) \\
        u_c(c,\ell) &\sim u_c(c,n) \\
        u_{\ell,\ell}(c,\ell) &\sim u_{nn}(c,n) \\
        u_{c,c}(c,\ell)& \sim u_{c,c}(c,n) \\
        u_{c,\ell} (c,\ell) &\sim 0
        \end{aligned}

With these understandings, equations :eq:`TS_barg10a` and :eq:`TS_barg11b` simplify in the case of the CRRA utility function.

They become

.. math::
    :label: amss2_TS_barg10

    (1+\Phi) [u_c(c) + u_n(c+g)] + \Phi[c u_{cc}(c) + (c+g) u_{nn}(c+g)] = 0

and

.. math::
    :label: amss2_TS_barg11

    (1+\Phi) [u_c(c_0) + u_n(c_0+g_0)] + \Phi[c_0 u_{cc}(c_0) + (c_0+g_0) u_{nn}(c_0+g_0)] - \Phi u_{cc}(c_0) b_0 = 0

In equation :eq:`amss2_TS_barg10`, it is understood that :math:`c` and :math:`g` are each functions of the Markov state :math:`s`.

The CRRA utility function is represented in the following class.

.. literalinclude:: /_static/lecture_specific/opt_tax_recur/crra_utility.py


Example Economy
==================

We set the following parameter values.

The Markov state :math:`s_t` takes two values, namely,  :math:`0,1`.

The initial Markov state is :math:`0`.

The Markov transition matrix is :math:`.5 I` where :math:`I` is a :math:`2 \times 2` identity matrix, so the :math:`s_t` process is IID.

Government expenditures :math:`g(s)` equal :math:`.1` in Markov state :math:`0` and :math:`.2` in Markov state :math:`1`.

We set preference parameters as follows:

.. math::

    \begin{aligned}
    \beta & = .9 \cr
    \sigma & = 2  \cr
    \gamma & = 2
    \end{aligned}

Here are several classes that do most of the work for us.

The code is  mostly taken or adapted from the earlier lectures :doc:`optimal taxation without state-contingent debt<amss>` and
:doc:`optimal taxation with state-contingent debt<opt_tax_recur>`.


.. literalinclude:: /_static/lecture_specific/opt_tax_recur/sequential_allocation.py
    :class: collapse

.. literalinclude:: /_static/lecture_specific/amss/recursive_allocation.py
    :class: collapse


.. literalinclude:: /_static/lecture_specific/amss/utilities.py
    :class: collapse


Reverse Engineering Strategy
=============================

We can reverse engineer a value :math:`b_0` of initial debt due   that renders the AMSS measurability constraints not binding from time :math:`t =0` onward.

We accomplish this by recognizing that if the AMSS measurability constraints never bind, then the AMSS allocation and Ramsey plan is equivalent
with that for a Lucas-Stokey economy in which for each period :math:`t \geq 0`, the government promises to pay the **same** state-contingent
amount  :math:`\bar b` in each state tomorrow.

This insight tells us to find a :math:`b_0` and other fundamentals for the Lucas-Stokey :cite:`LucasStokey1983` model that make the Ramsey planner
want to borrow the same value :math:`\bar b` next period for all states and all dates.

We accomplish this by using various equations for the Lucas-Stokey :cite:`LucasStokey1983` model
presented in :doc:`optimal taxation with state-contingent debt<opt_tax_recur>`.

We use the following steps.


**Step 1:**  Pick an initial :math:`\Phi`.

**Step 2:** Given that :math:`\Phi`, jointly solve two versions of equation :eq:`amss2_TS_barg10` for :math:`c(s), s=1, 2` associated with the two values
for :math:`g(s), s=1,2`.

**Step 3:**  Solve the following equation for :math:`\vec x`

.. math::
    :label: LSA_xsola

    \vec x= (I - \beta \Pi )^{-1} [ \vec u_c (\vec n-\vec g) - \vec u_l \vec n]

**Step 4:** After solving for :math:`\vec x`, we can find :math:`b(s_t|s^{t-1})` in Markov
state :math:`s_t=s` from :math:`b(s) = {\frac{x(s)}{u_c(s)}}` or the matrix equation

.. math::
    :label: amss2_LSA_bsol

    \vec b = {\frac{ \vec x }{\vec u_c}}

**Step 5:** Compute :math:`J(\Phi) = (b(1) - b(2))^2`.

**Step 6:** Put steps 2 through 6 in a function minimizer and find a :math:`\Phi` that minimizes :math:`J(\Phi)`.


**Step 7:** At the value of :math:`\Phi` and the value of :math:`\bar b` that emerged from step 6, solve equations
:eq:`amss2_TS_barg11` and :eq:`eqn_AMSS2_10` jointly for :math:`c_0, b_0`.


Code for Reverse Engineering
===============================================

Here is code to do the calculations for us.

.. code-block:: python3

    u = CRRAutility()

    def min_Φ(Φ):

        g1, g2 = u.G  # Government spending in s=0 and s=1

        # Solve Φ(c)
        def equations(unknowns, Φ):
            c1, c2 = unknowns
            # First argument of .Uc and second argument of .Un are redundant

            # Set up simultaneous equations
            eq = lambda c, g: (1 + Φ) * (u.Uc(c, 1) - -u.Un(1, c + g)) + \
                                Φ * ((c + g) * u.Unn(1, c + g) + c * u.Ucc(c, 1))

            # Return equation evaluated at s=1 and s=2
            return np.array([eq(c1, g1), eq(c2, g2)]).flatten()

        global c1                 # Update c1 globally
        global c2                 # Update c2 globally

        c1, c2 = fsolve(equations, np.ones(2), args=(Φ))

        uc = u.Uc(np.array([c1, c2]), 1)                            # uc(n - g)
        # ul(n) = -un(c + g)
        ul = -u.Un(1, np.array([c1 + g1, c2 + g2])) * [c1 + g1, c2 + g2]
        # Solve for x
        x = np.linalg.solve(np.eye((2)) - u.β * u.π, uc * [c1, c2] - ul)

        global b                 # Update b globally
        b = x / uc
        loss = (b[0] - b[1])**2

        return loss

    Φ_star = fmin(min_Φ, .1, ftol=1e-14)


To recover and print out :math:`\bar b`

.. code-block:: python3

    b_bar = b[0]
    b_bar


To complete the reverse engineering exercise by jointly determining :math:`c_0, b_0`,  we
set up a function that returns two simultaneous equations.

.. code-block:: python3

    def solve_cb(unknowns, Φ, b_bar, s=1):

        c0, b0 = unknowns

        g0 = u.G[s-1]

        R_0 = u.β * u.π[s] @ [u.Uc(c1, 1) / u.Uc(c0, 1), u.Uc(c2, 1) / u.Uc(c0, 1)]
        R_0 = 1 / R_0

        τ_0 = 1 + u.Un(1, c0 + g0) / u.Uc(c0, 1)

        eq1 = τ_0 * (c0 + g0) + b_bar / R_0 - b0 - g0
        eq2 = (1 + Φ) * (u.Uc(c0, 1)  + u.Un(1, c0 + g0)) \
               + Φ * (c0 * u.Ucc(c0, 1) + (c0 + g0) * u.Unn(1, c0 + g0)) \
               - Φ * u.Ucc(c0, 1) * b0

        return np.array([eq1, eq2], dtype='float64')

To solve the equations for :math:`c_0, b_0`, we use SciPy's fsolve function

.. code-block:: python3

    c0, b0 = fsolve(solve_cb, np.array([1., -1.], dtype='float64'), 
                    args=(Φ_star, b[0], 1), xtol=1.0e-12)
    c0, b0

Thus, we have reverse engineered an initial :math:`b0 = -1.038698407551764` that ought to render the AMSS measurability constraints slack.


Short Simulation for Reverse-engineered: Initial Debt
=====================================================

The following graph shows simulations of outcomes for both a Lucas-Stokey economy and for an AMSS economy starting from initial government
debt equal to :math:`b_0 = -1.038698407551764`.

These graphs report outcomes for both the Lucas-Stokey economy with complete markets and the AMSS economy with one-period risk-free debt only.

.. code-block:: python3

    μ_grid = np.linspace(-0.09, 0.1, 100)

    log_example = CRRAutility()

    log_example.transfers = True                    # Government can use transfers
    log_sequential = SequentialAllocation(log_example)  # Solve sequential problem
    log_bellman = RecursiveAllocationAMSS(log_example, μ_grid,
                                          tol_diff=1e-10, tol=1e-12)

    T = 20
    sHist = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                      0, 0, 0, 1, 1, 1, 1, 1, 1, 0])


    sim_seq = log_sequential.simulate(-1.03869841, 0, T, sHist)
    sim_bel = log_bellman.simulate(-1.03869841, 0, T, sHist)

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

The Ramsey allocations and Ramsey outcomes are **identical** for the Lucas-Stokey and AMSS economies.

This outcome confirms the success of our reverse-engineering exercises.

Notice how for :math:`t \geq 1`, the tax rate is a constant - so is the par value of government debt.

However, output and labor supply are both nontrivial time-invariant functions of the Markov state.


Long Simulation
=================

The following graph shows the par value of government debt and the flat rate tax on labor income  for a long simulation for our sample economy.

For the **same** realization of a government expenditure path, the graph reports outcomes for two economies

- the gray lines are for the Lucas-Stokey economy with complete markets

- the blue lines are for the AMSS economy with risk-free one-period debt only

For both economies, initial government debt due at time :math:`0` is :math:`b_0 = .5`.

For the Lucas-Stokey complete markets economy, the government debt plotted is :math:`b_{t+1}(s_{t+1})`.

- Notice that this is a time-invariant function of the Markov state from the beginning.

For the AMSS incomplete markets economy, the government debt plotted is :math:`b_{t+1}(s^t)`.

- Notice that this is a martingale-like random process that eventually seems to converge to a constant :math:`\bar b \approx - 1.07`.

- Notice that the limiting value :math:`\bar b < 0` so that asymptotically the government makes a constant level of risk-free loans to the public\.

- In the simulation displayed as well as  other simulations we have run, the par value of government debt converges to about :math:`1.07` afters between 1400 to 2000 periods.

For the AMSS incomplete markets economy, the marginal tax rate on labor income  :math:`\tau_t` converges to a constant

- labor supply and output each converge to time-invariant functions of the Markov state

.. code-block:: python3

    T = 2000  # Set T to 200 periods

    sim_seq_long = log_sequential.simulate(0.5, 0, T)
    sHist_long = sim_seq_long[-3]
    sim_bel_long = log_bellman.simulate(0.5, 0, T, sHist_long)

    titles = ['Government Debt', 'Tax Rate']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for ax, title, id in zip(axes.flatten(), titles, [2, 3]):
        ax.plot(sim_seq_long[id], '-k', sim_bel_long[id], '-.b', alpha=0.5)
        ax.set(title=title)
        ax.grid()

    axes[0].legend(('Complete Markets', 'Incomplete Markets'))
    plt.tight_layout()
    plt.show()


Remarks about Long Simulation
-------------------------------

As remarked above, after :math:`b_{t+1}(s^t)` has converged to a constant, the measurability constraints in the AMSS model cease to bind

- the associated Lagrange multipliers on those implementability constraints converge to zero

This leads us to seek an initial value of government debt :math:`b_0` that renders the measurability constraints slack from time :math:`t=0` onward

- a tell-tale sign of this situation is that the Ramsey planner in a corresponding Lucas-Stokey economy would instruct the government to issue a
  constant level of government debt :math:`b_{t+1}(s_{t+1})` across the two Markov states

We  now describe how to find such an initial level of government debt.


BEGS Approximations of Limiting Debt and Convergence Rate
===========================================================

It is useful to link the outcome of our reverse engineering exercise to limiting approximations constructed by :cite:`BEGS1`.

:cite:`BEGS1` used a slightly different notation to represent a generalization of the AMSS model.

We'll introduce a version of their notation so that readers can quickly relate notation that appears in their key formulas to the notation
that we have used.



BEGS work with objects :math:`B_t, {\mathcal B}_t, {\mathcal R}_t, {\mathcal X}_t` that are related to our notation by


.. math::

     \begin{aligned}
     {\mathcal R}_t & = \frac{u_{c,t}}{u_{c,t-1}} R_{t-1}  = \frac{u_{c,t}}{ \beta E_{t-1} u_{c,t}} \\
     B_t & = \frac{b_{t+1}(s^t)}{R_t(s^t)} \\
     b_t(s^{t-1}) & = {\mathcal R}_{t-1} B_{t-1} \\
     {\mathcal B}_t & = u_{c,t} B_t = (\beta E_t u_{c,t+1}) b_{t+1}(s^t) \\
     {\mathcal X}_t & = u_{c,t} [g_t - \tau_t n_t]
     \end{aligned}


In terms of their notation, equation (44) of :cite:`BEGS1` expresses the time :math:`t` state :math:`s` government budget constraint as

.. math::
    :label: eq_fiscal_risk

    {\mathcal B}(s) = {\mathcal R}_\tau(s, s_{-}) {\mathcal B}_{-} + {\mathcal X}_{\tau(s)} (s)

where the dependence on :math:`\tau` is to remind us that these objects depend on the tax rate and :math:`s_{-}` is last period's Markov state.

BEGS interpret random variations in the right side of :eq:`eq_fiscal_risk` as a measure of **fiscal risk** composed of

- interest-rate-driven fluctuations in time :math:`t` effective payments due on the government portfolio, namely,
  :math:`{\mathcal R}_\tau(s, s_{-}) {\mathcal B}_{-}`,  and

- fluctuations in the effective government deficit :math:`{\mathcal X}_t`

Asymptotic Mean
----------------

BEGS give conditions under which the ergodic mean of :math:`{\mathcal B}_t` is

.. math::
   :label: prelim_formula

   {\mathcal B}^* = - \frac{\rm cov^{\infty}(\mathcal R, \mathcal X)}{\rm var^{\infty}(\mathcal R)}

where the superscript :math:`\infty` denotes a moment taken with respect to an ergodic distribution.

Formula :eq:`prelim_formula` presents :math:`{\mathcal B}^*` as a regression coefficient of :math:`{\mathcal X}_t` on :math:`{\mathcal R}_t` in the ergodic
distribution.

This regression coefficient emerges as the minimizer for a variance-minimization problem:

.. math::
   :label: eq_criterion_fiscal

   {\mathcal B}^* = {\rm argmin}_{\mathcal B}  {\rm var} ({\mathcal R} {\mathcal B} + {\mathcal X})

The minimand in criterion :eq:`eq_criterion_fiscal` is the  measure of fiscal risk associated with a given tax-debt policy that appears on the right side
of equation :eq:`eq_fiscal_risk`.


Expressing formula :eq:`prelim_formula` in terms of  our notation tells us that :math:`\bar b` should approximately equal

.. math::
  :label: key_formula

  \hat b = \frac{\mathcal B^*}{\beta E_t u_{c,t+1}}

Rate of Convergence
--------------------

BEGS also derive the following  approximation to the rate of convergence to :math:`{\mathcal B}^{*}` from an arbitrary initial condition.

 .. math::
    :label: rate_of_convergence

    \frac{ E_t  ( {\mathcal B}_{t+1} - {\mathcal B}^{*} )} { ( {\mathcal B}_{t} - {\mathcal B}^{*} )} \approx \frac{1}{1 + \beta^2 {\rm var} ({\mathcal R} )}

(See the equation above equation (47) in :cite:`BEGS1`)




Formulas and Code Details
--------------------------

For our example, we describe some code that we use to compute the steady state mean and the rate of convergence to it.

The  values of :math:`\pi(s)` are 0.5, 0.5.

We can then construct :math:`{\mathcal X}(s), {\mathcal R}(s), u_c(s)` for our two states using  the definitions above.

We can then construct :math:`\beta E_{t-1} u_c = \beta \sum_s u_c(s) \pi(s)`, :math:`{\rm cov}({\mathcal R}(s), \mathcal{X}(s))` and
:math:`{\rm var}({\mathcal R}(s))` to be plugged into formula :eq:`key_formula`.

We also want to  compute :math:`{\rm var}({\mathcal X})`.

To compute the variances and covariance, we use the following standard formulas.

Temporarily let :math:`x(s), s =1,2` be an arbitrary random variables.

Then we define

.. math::

   \begin{aligned}
   \mu_x & = \sum_s x(s) \pi(s) \\
   {\rm var}(x) &= \left(\sum_s \sum_s x(s)^2 \pi(s) \right) - \mu_x^2 \\
   {\rm cov}(x,y)  & = \left(\sum_s x(s) y(s) \pi(s) \right) - \mu_x \mu_y
   \end{aligned}

After we compute these moments, we  compute the BEGS approximation to the asymptotic mean :math:`\hat b` in formula :eq:`key_formula`.

After that, we move on to compute :math:`{\mathcal B}^*` in formula :eq:`prelim_formula`.

We'll also evaluate  the BEGS criterion :eq:`eq_fiscal_risk` at the limiting value :math:`{\mathcal B}^*`

.. math::
    :label: eqn_Jcriterion

    J ( {\mathcal B}^*)=  {\rm var}(\mathcal{R}) \left( {\mathcal B}^* \right)^2 + 2 {\mathcal B}^* {\rm cov}(\mathcal{R},\mathcal{X}) + {\rm var}(\mathcal X)



Here are some functions that we'll use to compute key objects that we want

.. code-block:: python3

        def mean(x):
            '''Returns mean for x given initial state'''
            x = np.array(x)
            return x @ u.π[s]

        def variance(x):
            x = np.array(x)
            return x**2 @ u.π[s] - mean(x)**2

        def covariance(x, y):
            x, y = np.array(x), np.array(y)
            return x * y @ u.π[s] - mean(x) * mean(y)


Now let's form the two random variables :math:`{\mathcal R}, {\mathcal X}` appearing in the BEGS approximating formulas

.. code-block:: python3

    u = CRRAutility()

    s = 0
    c = [0.940580824225584, 0.8943592757759343]  # Vector for c
    g = u.G       # Vector for g
    n = c + g     # Total population
    τ = lambda s: 1 + u.Un(1, n[s]) / u.Uc(c[s], 1)

    R_s = lambda s: u.Uc(c[s], n[s]) / (u.β * (u.Uc(c[0], n[0]) * u.π[0, 0] \
                    + u.Uc(c[1], n[1]) * u.π[1, 0]))
    X_s = lambda s: u.Uc(c[s], n[s]) * (g[s] - τ(s) * n[s])

    R = [R_s(0), R_s(1)]
    X = [X_s(0), X_s(1)]

    print(f"R, X = {R}, {X}")

Now let's compute the ingredient of the approximating limit and the approximating rate of convergence

.. code-block:: python3

    bstar = -covariance(R, X) / variance(R)
    div = u.β * (u.Uc(c[0], n[0]) * u.π[s, 0] + u.Uc(c[1], n[1]) * u.π[s, 1])
    bhat = bstar / div
    bhat


Print out :math:`\hat b` and :math:`\bar b`

.. code-block:: python3

    bhat, b_bar

So we have

.. code-block:: python3

    bhat - b_bar

These outcomes show that :math:`\hat b` does a remarkably good job of approximating :math:`\bar b`.


Next, let's compute the BEGS fiscal criterion that :math:`\hat b` is minimizing

.. code-block:: python3

    Jmin = variance(R) * bstar**2 + 2 * bstar * covariance(R, X) + variance(X)
    Jmin

This is *machine zero*, a verification that :math:`\hat b` succeeds in minimizing the nonnegative fiscal cost criterion :math:`J ( {\mathcal B}^*)` defined in
BEGS and in equation :eq:`eqn_Jcriterion` above.


Let's push our luck and compute the mean reversion speed in the formula above equation (47) in :cite:`BEGS1`.

.. code-block:: python3

    den2 = 1 + (u.β**2) * variance(R)
    speedrever = 1/den2
    print(f'Mean reversion speed = {speedrever}')


Now let's compute the implied meantime to get to within 0.01 of the limit


.. code-block:: python3

    ttime = np.log(.01) / np.log(speedrever)
    print(f"Time to get within .01 of limit = {ttime}")

The slow rate of convergence and the implied time of getting within one percent of the limiting value do a good job of approximating
our long simulation above.
