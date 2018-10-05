.. _hist_dep:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3

******************************************
:index:`History Dependent Public Policies`
******************************************

.. contents:: :depth: 2

.. TODO: Still need to change some of references to matlab programs to python programs, but some are done.  Where should we put these python files - currently they are in /_static/temp/EvansSargent.

Overview
============

This lecture describes history-dependent public policies and some of their representations

History dependent policies are decision rules that depend on the entire past history of the state variables

History dependent policies naturally emerge in :doc:`Ramsey problems <lqramsey>`

A Ramsey planner (typically interpreted as a government) devises a plan  of actions at time :math:`t=0` to follow at all future dates and for all contingencies

In order to make a plan, he takes as given Euler equations expressing private agents' first-order necessary conditions

He also takes into account that his *future* actions affect earlier decisions by private agents, an avenue opened up by the maintained assumption of *rational expectations*

Another setting in which history dependent policies naturally emerge is where instead of a Ramsey planner there is a *sequence* of government administrators whose time :math:`t` member takes as given the policies used by its successors

We study these ideas in  the context of a model in which a benevolent tax authority is forced

* to raise a prescribed present value of revenues

* to do so by imposing a distorting flat rate tax on the output of a competitive representative firm

The firm faces costs of adjustment and lives within a competitive equilibrium, which in turn imposes restrictions on the tax authority [#fn_a]_


References
-----------

The presentation below is based on a recent paper by Evans and Sargent :cite:`evans2011history`

Regarding techniques, we will make use of the methods described in

#. the :doc:`linear regulator lecture<lqcontrol>`

#. the  :doc:`solving LQ dynamic Stackelberg problems lecture<dyn_stack>` 

.. :doc:`Stackelberg LQ models <lqstackelberg>`.

.. TODO: Need to update the link above when the LQ Stackelberg lecture is released.



Two Sources of History Dependence
==================================

.. index::
    single: History Dependent Public Policies; Timing Protocols

We compare two timing protocols

#. An infinitely lived benevolent tax authority solves a Ramsey problem

#. There is a sequence of tax authorities, each choosing only a time :math:`t` tax rate

Under both timing protocols, optimal tax policies are *history-dependent*

But  history dependence captures  different economic forces across the two timing protocols

In the first timing protocol, history dependence expresses the *time-inconsistency of the Ramsey plan*

In the second timing protocol, history dependence reflects the unfolding of constraints that assure that a time :math:`t` government administrator wants to confirm the representative firm's expectations about government actions

We describe recursive representations of history-dependent tax policies under both timing protocols



Ramsey Timing Protocol
------------------------

.. index::
    single: History Dependent Public Policies; Ramsey Timing

The first timing protocol models a policy maker who can be said to 'commit', choosing a sequence of tax rates once-and-for-all at time :math:`0`


Sequence of Governments Timing Protocol
------------------------------------------

.. index::
    single: History Dependent Public Policies; Sequence of Governments Timing

For the second timing protocol we use the notion of a *sustainable plan* proposed in :cite:`chari1990sustainable`, also referred to as a *credible public policy* in :cite:`stokey1989reputation`

A key idea here is that history-dependent policies can be arranged so that, when regarded as a representative firm's forecasting functions, they confront policy makers with incentives to confirm them

We follow Chang :cite:`chang1998credible` in expressing such history-dependent plans recursively

Credibility considerations contribute an additional auxiliary state variable in the form of a promised value to the planner

It expresses how decisions must unfold to give the government the incentive to confirm private sector expectations when the government chooses sequentially

.. note::
      We occasionally hear confusion about the consequences of  recursive
      representations of government policies under our two timing protocols.
      It is incorrect to regard a recursive representation of the Ramsey plan
      as in any way 'solving a time-inconsistency problem'.  On the contrary,
      the evolution of the auxiliary state variable that augments the
      authentic ones under our first timing protocol ought to be viewed as
      *expressing* the time-inconsistency of a Ramsey plan.  Despite that, in
      literatures about practical monetary policy one sometimes hears
      interpretations that sell Ramsey plans in settings where our sequential
      timing protocol is the one that more accurately characterizes  decision
      making.  Please beware of discussions that toss around claims about
      credibility if you don't also see recursive representations of policies
      with the complete list of state variables appearing in our :cite:`chang1998credible` -like analysis that we present :ref:`below <sec:credible>`.



Competitive equilibrium
=======================

.. index::
    single: History Dependent Public Policies; Competitive Equilibrium

A representative competitive firm sells output :math:`q_t` at price
:math:`p_t` when market-wide output is :math:`Q_t`

The market as a whole faces a downward sloping inverse demand function

.. math::
    :label: ES_1

    p_t = A_0 - A_1 Q_t, \quad A_0 >0, A_1 >0


The representative firm

* has given initial condition :math:`q_0`

* endures quadratic adjustment costs :math:`\frac{d}{2} (q_{t+1} - q_t)^2`

* pays a flat rate tax :math:`\tau_t` per unit of output

* treats :math:`\{p_t, \tau_t\}_{t=0}^\infty` as exogenous

* chooses :math:`\{q_{t+1}\}_{t=0}^\infty` to maximize

.. math::
    :label: ES_2

    \sum_{t=0}^\infty \beta^t \bigl\{ p_t q_t - \frac{d}{2}(q_{t+1} - q_t)^2 - \tau_t q_t \bigr\}


Let :math:`u_t := q_{t+1} - q_t` be the firm's 'control variable' at time :math:`t`

First-order conditions for the representative firm's problem are

.. math::
    :label: ES_3

    u_t = \frac{\beta}{d} p_{t+1} + \beta u_{t+1} - \frac{\beta}{d} \tau_{t+1},
    \quad t = 0, 1, \ldots


To compute a competitive equilibrium, it is appropriate to take :eq:`ES_3`, eliminate :math:`p_t` in favor of :math:`Q_t` by using :eq:`ES_1`, and then set :math:`q_t = Q_t`

This last step *makes the representative firm be representative* [#fn_b]_

We arrive at

.. math::
    :label: ES_4

    u_t = \frac{\beta}{d} ( A_0 - A_1 Q_{t+1} ) + \beta u_{t+1} - \frac{\beta}{d} \tau_{t+1}


.. math::
    :label: ES_5

    Q_{t+1} = Q_t + u_t


**Notation:** For any scalar :math:`x_t`, let :math:`\vec x = \{x_t\}_{t=0}^\infty`

Given a tax sequence :math:`\{\tau_{t+1}\}_{t=0}^\infty`, a **competitive
equilibrium** is a price sequence :math:`\vec p` and an output sequence :math:`\vec Q` that satisfy :eq:`ES_1`,  :eq:`ES_4`, and :eq:`ES_5`

For any sequence :math:`\vec x = \{x_t\}_{t=0}^\infty`, the sequence :math:`\vec x_1 := \{x_t\}_{t=1}^\infty` is called the **continuation sequence** or simply the **continuation**

Note that a competitive equilibrium consists of a first period value :math:`u_0 = Q_1-Q_0` and a continuation competitive equilibrium with initial condition :math:`Q_1`

Also, a continuation of a competitive equilibrium is a competitive equilibrium

Following the lead of :cite:`chang1998credible`, we shall make extensive use of the following property:

* A continuation :math:`\vec \tau_1 = \{\tau_{t}\}_{t=1}^\infty` of a tax policy :math:`\vec \tau` influences :math:`u_0` via :eq:`ES_4` entirely through its impact on :math:`u_1`

A continuation competitive equilibrium can be indexed by a :math:`u_1` that satisfies :eq:`ES_4`

In the spirit of :cite:`kydland1980dynamic` , we shall use :math:`u_{t+1}` to describe what we shall call a **promised marginal value** that a competitive equilibrium offers to a representative firm [#fn_c]_

Define :math:`Q^t := [Q_0, \ldots, Q_t]`

A **history-dependent tax policy** is a sequence of functions :math:`\{\sigma_t\}_{t=0}^\infty` with :math:`\sigma_t` mapping :math:`Q^t` into a choice of :math:`\tau_{t+1}`

Below, we shall

* Study history-dependent tax policies that either solve a Ramsey plan or are credible

* Describe recursive representations of both types of history-dependent policies




:index:`Ramsey Problem`
=======================

The planner's objective is cast in terms of consumer surplus net of the firm's adjustment costs

Consumer surplus is

.. math::

    \int_0^Q ( A_0 - A_1 x) dx = A_0 Q - \frac{A_1}{2} Q^2


Hence the planner's one-period return function is

.. math::
    :label: ES_7

    A_0 Q_t - \frac{A_1}{2} Q_t^2 - \frac{d}{2} u_t^2


At time :math:`t=0`, a Ramsey planner faces the intertemporal budget constraint

.. math::
    :label: ES_6

    \sum_{t=1}^\infty \beta^t \tau_t Q_t = G_0


Note that :eq:`ES_6` forbids taxation of initial output :math:`Q_0`


The **Ramsey problem** is to choose a tax sequence :math:`\vec \tau_1` and a competitive equilibrium outcome :math:`(\vec Q, \vec u)` that maximize

.. math::
    :label: ES_Lagrange0

    \sum_{t=0}^\infty \beta^t
    \left\{
      A_0 Q_t - \frac{A_1}{2}Q_t^2 - \frac{d}{2} u_t^2
    \right\}


subject to :eq:`ES_6`

Thus, the Ramsey timing protocol is:

#. At time :math:`0`, knowing :math:`(Q_0, G_0)`, the Ramsey planner chooses :math:`\{\tau_{t+1}\}_{t=0}^\infty`

#. Given :math:`\bigl(Q_0, \{\tau_{t+1}\}_{t=0}^\infty\bigr)`, a competitive equilibrium outcome :math:`\{u_t, Q_{t+1}\}_{t=0}^\infty` emerges



.. note::
    In bringing out the timing protocol associated with a Ramsey plan, we run
    head on into a set of issues analyzed by Bassetto :cite:`bassetto2005equilibrium`.  This is because
    our definition of the Ramsey timing protocol doesn't completely describe all
    conceivable actions by the government and firms as time unfolds.
    For example, the definition is silent about how the government would
    respond if firms, for some unspecified reason, were to choose to deviate
    from the competitive equilibrium associated with the Ramsey plan,
    possibly prompting violation of government budget balance.  This is an example of the
    issues raised by :cite:`bassetto2005equilibrium`, who identifies a class of government
    policy problems whose proper formulation requires supplying a complete and
    coherent description of all actors' behavior across all possible
    histories. Implicitly, we are assuming that a more complete description of
    a government strategy could be specified that (a)
    agrees with ours along the Ramsey outcome, and (b) suffices uniquely to
    implement the Ramsey plan by deterring firms from taking actions that
    deviate from the Ramsey outcome path.




Computing a Ramsey Plan
-----------------------

.. index::
    single: Ramsey Problem; Computing

The planner chooses :math:`\{u_t\}_{t=0}^\infty, \{\tau_t\}_{t=1}^\infty` to maximize :eq:`ES_Lagrange0` subject to :eq:`ES_4`, :eq:`ES_5`, and :eq:`ES_6`

To formulate this problem as a Lagrangian, attach a Lagrange multiplier
:math:`\mu` to the budget constraint :eq:`ES_6`

Then the planner chooses :math:`\{u_t\}_{t=0}^\infty, \{\tau_t\}_{t=1}^\infty`
to maximize and the Lagrange multiplier :math:`\mu` to minimize

.. math::
    :label: ES_Lagrange1

    \sum_{t=0}^\infty
    \beta^t
    ( A_0 Q_t - \frac{A_1}{2}Q_t^2 - \frac{d}{2} u_t^2 ) +
      \mu \left[
              \sum_{t=0}^\infty \beta^t \tau_t Q_t -G_0 - \tau_0 Q_0
          \right]


subject to and :eq:`ES_4` and :eq:`ES_5`


The Ramsey problem is a special case of the linear quadratic dynamic Stackelberg problem analyzed in :doc:`this lecture<dyn_stack>` 

The key implementability conditions are :eq:`ES_4` for :math:`t \geq 0`

Holding fixed :math:`\mu` and :math:`G_0`, the Lagrangian for the planning problem can be abbreviated as

.. math::

    \max_{\{u_t, \tau_{t+1}\}}
    \sum_{t=0}^\infty
      \beta^t
      \left\{
           A_0 Q_t-\frac {A_1}2 Q_t^2-\frac d2 u_t^2+\mu \tau_t Q_t
      \right\}


Define

.. math::

     z_t :=
     \begin{bmatrix}
         1
         \\
         Q_t
         \\
         \tau_t
     \end{bmatrix}
     \quad \text{and} \quad
     y_t :=
     \begin{bmatrix}
         z_t
         \\
         u_t
    \end{bmatrix}
     = \begin{bmatrix}
         1
         \\
         Q_t
         \\
         \tau_t
         \\
         u_t
     \end{bmatrix}


Here the elements of :math:`z_t` are natural state variables and :math:`u_t` is a forward looking variable that we treat as a state variable for :math:`t \geq 1`

But :math:`u_0` is a choice variable for the Ramsey planner.

We include :math:`\tau_t` as a state variable for bookkeeping purposes: it helps to map the problem into a linear regulator problem with no cross products between states and controls

However, it will be a redundant state variable in the sense that the optimal tax :math:`\tau_{t+1}` will not depend on :math:`\tau_t`

The government chooses :math:`\tau_{t+1}` at time :math:`t` as a function of the time :math:`t` state

Thus, we can rewrite the Ramsey problem as

.. math::
    :label: ES_10

    \max_{\{y_t, \tau_{t+1}\}} -\sum_{t=0}^\infty \beta^t y_t' Ry_t


subject to :math:`z_0` given and the law of motion

.. math::
    :label: ES_11

    y_{t+1} = A y_t + B \tau_{t+1}


where

.. math::

    R =
    \begin{bmatrix}
         0 &-\frac{A_0}{2} & 0 & 0
         \\
         -\frac{A_0}{2} & \frac{A_1}{2} & \frac {-\mu}{2} & 0
         \\
         0 & \frac{-\mu}{2} & 0 & 0
         \\ 0 & 0 & 0 & \frac{d}{2}
    \end{bmatrix},
    \quad
    A =
    \begin{bmatrix}
         1 & 0 & 0 & 0
         \\
         0 & 1 & 0 & 1
         \\
         0 & 0 & 0 & 0
         \\
         -\frac{A_0}{d} & \frac{A_1}{d} & 0 & \frac{A_1}{d} + \frac{1}{\beta}
    \end{bmatrix},
    \quad
    B =
    \begin{bmatrix}
        0 \\ 0 \\ 1 \\ \frac{1}{d}
    \end{bmatrix}


.. _sec:twosub:

Two Subproblems
==================

.. index::
    single: Ramsey Problem; Two Subproblems

Working backwards, we first present the Bellman equation for the value function that takes both :math:`z_t` and :math:`u_t` as given. Then we present
a value function that takes only :math:`z_0` as given and is the indirect utility function that arises from choosing :math:`u_0` optimally.

Let :math:`v(Q_t, \tau_t, u_t)` be the optimum value function for the time :math:`t \geq 1` government administrator facing state
:math:`Q_t, \tau_t, u_t`.

Let :math:`w(Q_0)` be the value of the Ramsey plan starting from :math:`Q_0`     

Subproblem 1
--------------

Here the Bellman equation is

.. math::

    v(Q_t,\tau_t,u_t)
    =
    \max_{\tau_{t+1}}
        \left\{
            A_0 Q_t-\frac {A_1}2 Q_t^2-\frac d2 u_t^2+\mu\tau_tQ_t
                + \beta v(Q_{t+1},\tau_{t+1},u_{t+1})
        \right\}


where the maximization is subject to the constraints

.. math::

    Q_{t+1} = Q_t+u_t


and

.. math::

    u_{t+1}
    =
    -\frac{A_0}d+\frac{A_1}d Q_t
        +  \frac{A_1}d+\frac1\beta u_t+\frac1d \tau_{t+1}


Here we regard :math:`u_t` as a state

Subproblem 2
--------------

The subproblem 2 Bellman equation is

.. math::

    w(z_0) = \max_{u_0} v (Q_0,0, u_0)


Details 
--------

Define the state vector to be

.. math::

    y_t =
        \begin{bmatrix}
            1
            \\
            Q_t
            \\
            \tau_t
            \\
            u_t
        \end{bmatrix}
        =
        \begin{bmatrix}
            z_t
            \\
            u_t
        \end{bmatrix},


where :math:`z_t = \begin{bmatrix} 1 & Q_t & \tau_t\end{bmatrix}'` are authentic state variables and :math:`u_t` is a variable whose time :math:`0` value is a 'jump' variable but whose values for dates :math:`t \geq 1` will become state variables that encode history dependence in the Ramsey plan



.. math::
    :label: ES_KP

    v(y_t) = \max_{\tau_{t+1}} \left\{ -y_t'Ry_t+\beta v(y_{t+1}) \right\}


where the maximization is subject to the constraint

.. math::

    y_{t+1} = Ay_t+B\tau_{t+1}


and where

.. math::

    R = \begin{bmatrix} 0 & -\frac {A_0}2 & 0 & 0 \\ -\frac{A_0}2 & \frac{A_1}2 & \frac {-\mu}{2}&0\\ 0 & \frac{-\mu}{2}&0 & 0 \\ 0 & 0 & 0&\frac d2\end{bmatrix},
    \: A = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 1\\ 0 & 0 & 0 & 0 \\ -\frac{A_0}d & \frac{A_1}d & 0 & \frac{A_1}d+\frac1\beta\end{bmatrix}\text{,  and  }B =\begin{bmatrix} 0 \\ 0 \\ 1 \\ \frac1d \end{bmatrix}.


Functional equation :eq:`ES_KP` has solution

.. math::

    v(y_t) = -y_t'Py_t


where

* :math:`P` solves the algebraic matrix Riccati equation :math:`P = R+ \beta A'PA- \beta A'PB(B'PB)^{-1}B'PA`

* the optimal policy function is given by :math:`\tau_{t+1} = -F y_t` for :math:`F = (B'PB)^{-1}B'PA`

Now we turn to subproblem 1.  

Evidently the optimal choice of :math:`u_0` satisfies :math:`\frac{\partial v}{\partial u_0} =0`

If we partition :math:`P` as

.. math::

    P
    =
    \begin{bmatrix}
        P_{11}&P_{12}
        \\
        P_{21}&P_{22}
    \end{bmatrix}


then we have

.. math::

    0
    =
    \frac{\partial}{\partial u_0}
        \left(
            z_0'P_{11}z_0+z_0'P_{12}u_0+u_0'P_{21}z_0 +u_0' P_{22} u_0
        \right)
    =
    P_{12}'z_0+P_{21}z_0+2P_{22}u_0


which implies

.. math::
    :label: ES_u0

    u_0 = -P_{22}^{-1}P_{21}z_0


Thus, the Ramsey plan is

.. math::

    \tau_{t+1}
    =
    -F
    \begin{bmatrix}
        z_t
        \\
        u_t
    \end{bmatrix}
    \quad \text{and} \quad
    \begin{bmatrix}
        z_{t+1}
        \\
        u_{t+1}
    \end{bmatrix}
    =
    (A-BF)
    \begin{bmatrix}
        z_t
        \\
        u_t
    \end{bmatrix}


with initial state :math:`\begin{bmatrix} z_0 & -P_{22}^{-1}P_{21}z_0\end{bmatrix}'`





Recursive Representation
-------------------------

.. index::
    single: Ramsey Problem; Recursive Representation

An outcome of the preceding results is that the Ramsey plan can be represented recursively as the choice of an initial marginal utility (or rate of growth of output) according to a function

.. math::
    :label: ES_24

    u_0 = \upsilon(Q_0|\mu)


that obeys :eq:`ES_u0`  and the following updating equations for :math:`t\geq 0`:

.. math::
    :label: ES_25

    \tau_{t+1} = \tau(Q_t, u_t|\mu)


.. math::
    :label: ES_26

    Q_{t+1} = Q_t + u_t


.. math::
    :label: ES_27

    u_{t+1} = u(Q_t, u_t|\mu)


We have conditioned the functions :math:`\upsilon`, :math:`\tau`, and :math:`u` by :math:`\mu` to emphasize how the dependence of :math:`F` on :math:`G_0` appears indirectly through the Lagrange multiplier :math:`\mu`


An Example Calculation
-----------------------

We'll discuss how to compute :math:`\mu` :ref:`below <sec:computing_mu>` but first consider the following numerical example

We take the parameter set :math:`[A_0, A_1, d, \beta, Q_0] = [100, .05, .2, .95, 100]` and compute the Ramsey plan with the following piece of code



.. literalinclude:: /_static/code/hist_dep_policies/evans_sargent.py




The above code computes a number of sequences besides the Ramsey plan, some of which have already been discussed, while others will be described below

The next figure uses the program to compute and show the Ramsey plan for :math:`\tau` and the Ramsey outcome for :math:`(Q_t,u_t)`



.. code-block:: python3

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    labels = ["Output", "Tax rate", "First difference in output"]
    ylabels = ["$Q$", r"$\tau$", "$u$"]

    for y_i, ax, label, ylabel, in zip(y[1:], axes, labels, ylabels):
        ax.plot(np.arange(T), y_i, label=label, lw=2, alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.grid()
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Time", fontsize=16)

    plt.show()
    


From top to bottom, the panels show :math:`Q_t`, :math:`\tau_t` and :math:`u_t := Q_{t+1} - Q_t` over :math:`t=0, \ldots, 15`

The optimal decision rule is [#fn_e]_

.. math::
    :label: ES_tax_rule

    \tau_{t+1} = -248.0624 - 0.1242 Q_t - 0.3347 u_t


Notice how the Ramsey plan calls for a high tax at :math:`t=1` followed by a perpetual stream of lower taxes

Taxing heavily at first, less later expresses time-inconsistency of the optimal plan for :math:`\{\tau_{t+1}\}_{t=0}^\infty`

We'll characterize this formally after first discussing how to compute :math:`\mu`.




.. _sec:computing_mu:

Computing :math:`\mu`
---------------------------

Define the selector vectors :math:`e_\tau = \begin{bmatrix} 0 & 0 & 1 & 0 \end{bmatrix}'` and :math:`e_Q = \begin{bmatrix} 0 & 1 & 0 & 0 \end{bmatrix}'` and express :math:`\tau_t = e_\tau' y_t` and :math:`Q_t = e_Q' y_t`

Evidently :math:`Q_t \tau_t = y_t' e_Q e_\tau' y_t = y_t' S y_t` where :math:`S := e_Q e_\tau'`

We want to compute

.. math::

    T_0 = \sum_{t=1}^\infty \beta^t \tau_t Q_t  = \tau_1 Q_1 + \beta T_1


where :math:`T_1 = \sum_{t=2}^\infty \beta^{t-1} Q_t \tau_t`

The present values :math:`T_0` and :math:`T_1` are connected by

.. math::

    T_0 = \beta y_0' A_F' S A_F y_0 + \beta T_1


Guess a solution that takes the form :math:`T_t = y_t' \Omega y_t`, then find an :math:`\Omega` that satisfies

.. math::
    :label: ES_Lyapunov

    \Omega = \beta A_F' S A_F + \beta A_F' \Omega A_F


Equation :eq:`ES_Lyapunov` is a discrete Lyapunov equation that can be solved for :math:`\Omega` using QuantEcon's
`solve_discrete_lyapunov <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/matrix_eqn.py#L25>`_
function

The matrix :math:`F` and therefore the matrix :math:`A_F = A-BF` depend on :math:`\mu`

To find a :math:`\mu` that guarantees that :math:`T_0 = G_0` we proceed as follows:

#. Guess an initial :math:`\mu`, compute a tentative Ramsey plan and the implied :math:`T_0 = y_0' \Omega(\mu) y_0`

#. If :math:`T_0 > G_0`, lower :math:`\mu`; otherwise, raise :math:`\mu`

#. Continue iterating on step 3 until :math:`T_0 = G_0`






Time Inconsistency
==================

.. index::
    single: Ramsey Problem; Time Inconsistency

Recall that the Ramsey planner chooses :math:`\{u_t\}_{t=0}^\infty, \{\tau_t\}_{t=1}^\infty` to maximize

.. math::

    \sum_{t=0}^\infty \beta^t
      \left\{
          A_0 Q_t - \frac{A_1}{2}Q_t^2 - \frac{d}{2} u_t^2
      \right\}


subject to :eq:`ES_4`, :eq:`ES_5`, and :eq:`ES_6`

We express the outcome that a Ramsey plan is time-inconsistent the following way

**Proposition.** A continuation of a Ramsey plan is not a Ramsey plan

Let

.. math::
    :label: ES_Ramsey_value

    w(Q_0,u_0|\mu_0)
    = \sum_{t=0}^\infty \beta^t
     \left\{
         A_0 Q_t - \frac{A_1}{2}Q_t^2 - \frac{d}{2} u_t^2
     \right\}


where

* :math:`\{Q_t,u_t\}_{t=0}^\infty` are evaluated under the Ramsey plan whose recursive representation is given by :eq:`ES_25`, :eq:`ES_26`, :eq:`ES_27`

* :math:`\mu_0` is the value of the Lagrange multiplier that assures budget balance, computed as described :ref:`above <sec:computing_mu>`

Evidently, these continuation values satisfy the recursion

.. math::
    :label: ES_28a

    w(Q_t,u_t|\mu_0) = A_0 Q_{t} - \frac{A_1}{2} Q_{t}^2 - \frac{d}{2} u_{t}^2  + \beta w (Q_{t+1},u_{t+1}|\mu_0)


for all :math:`t \geq 0`, where :math:`Q_{t+1} = Q_t + u_t`

Under the timing protocol affiliated with the Ramsey plan, the planner is committed to the outcome of iterations on :eq:`ES_25`, :eq:`ES_26`, :eq:`ES_27`

In particular, when time :math:`t` comes, the Ramsey planner is committed to the value of :math:`u_t` implied by the Ramsey plan and receives continuation value :math:`w(Q_t,u_t,\mu_0)`

That the Ramsey plan is time-inconsistent can be seen by subjecting it to the following 'revolutionary' test

First, define continuation revenues :math:`G_t` that the government raises along the original Ramsey outcome by

.. math::
    :label: eqn:G_continuation

    G_t = \beta^{-t}(G_0-\sum_{s=1}^t\beta^s\tau_sQ_s)


where :math:`\{\tau_t, Q_t\}_{t=0}^\infty` is the original Ramsey outcome
[#fn_f]_

Then at time :math:`t \geq 1`,

#. take :math:`(Q_t, G_t)` inherited from the original Ramsey plan as initial conditions

#. invite a brand new Ramsey planner to compute a new Ramsey plan, solving for a new :math:`u_t`, to be called :math:`{\check u_t}`, and for a new :math:`\mu`, to be called :math:`{\check \mu_t}`

The revised Lagrange multiplier :math:`\check{\mu_t}` is chosen so that, under the new Ramsey plan, the government is able to raise enough continuation revenues :math:`G_t` given by :eq:`eqn:G_continuation`

Would this new Ramsey plan be a continuation of the original plan?

The answer is no because along a Ramsey plan, for :math:`t \geq 1`, in general it is true that

.. math::
    :label: ES_28

    w\bigl(Q_t, \upsilon(Q_t|\check{\mu})|\check{\mu}\bigr) > w(Q_t, u_t|\mu_0)


Inequality :eq:`ES_28` expresses a continuation Ramsey planner's incentive to deviate from a time :math:`0` Ramsey plan by

#. resetting :math:`u_t` according to :eq:`ES_24`

#.  adjusting the Lagrange multiplier on the continuation appropriately to account for tax revenues already collected [#fn_g]_

Inequality :eq:`ES_28` expresses the time-inconsistency of a Ramsey plan

A Simulation
-------------

To bring out the time inconsistency of the Ramsey plan, we compare

* the time :math:`t` values of :math:`\tau_{t+1}` under the original Ramsey plan with

* the value :math:`\check \tau_{t+1}` associated with a new Ramsey plan begun at time :math:`t` with initial conditions :math:`(Q_t, G_t)` generated by following the *original* Ramsey plan

Here again :math:`G_t := \beta^{-t}(G_0-\sum_{s=1}^t\beta^s\tau_sQ_s)`

The difference :math:`\Delta \tau_t := \check{\tau_t} -  \tau_t` is shown in the top panel of the following figure




.. code-block:: python3

    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    
    yvals = [τhatdif, uhatdif, μ, G]
    labels = ["Time inconsistency differential for tax rate", 
              "Time inconsistency differential for $u$",
              "Lagrange multiplier",
              "Government revenue"]
    ylabels = [r"$\Delta\tau$", r"$\Delta u$", r"$\mu$", "$G$"]
    
    for ax, y, label, ylabel in zip(axes, yvals, labels, ylabels):
        ax.plot(np.arange(T), y, label=label, lw=2, alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=16) 
        ax.set_xlim(0, 15)
        ax.grid()
        ax.legend(loc="upper left")
        
    axes[-1].set_xlabel("Time", fontsize=16)

    plt.tight_layout()
    plt.show()




In the second panel we compare the time :math:`t` outcome for :math:`u_t` under the original Ramsey plan with the time :math:`t` value of this new Ramsey problem starting from :math:`(Q_t, G_t)`

To compute :math:`u_t` under the new Ramsey plan, we use the following version of formula :eq:`ES_u0`:

.. math::

    \check{u_t} = - P_{22}^{-1} (\check\mu_{t}) P_{21}(\check\mu_t) z_t


Here :math:`z_t` is evaluated along the Ramsey outcome path, where we have included :math:`\check{\mu_t}` to emphasize the dependence of :math:`P` on the Lagrange multiplier :math:`\mu_0` [#fn_h]_

To compute :math:`u_t` along the Ramsey path, we just iterate the recursion starting :eq:`ES_27` from the initial :math:`Q_0` with :math:`u_0` being given by formula :eq:`ES_u0`

Thus the second panel indicates how far the reinitialized value :math:`\check{u_t}` value departs from the time :math:`t` outcome along the Ramsey plan

Note that the restarted plan raises the time :math:`t+1` tax and consequently lowers the time :math:`t` value of :math:`u_t`

Associated with the new Ramsey plan at :math:`t` is a value of the Lagrange multiplier on the continuation government budget constraint

This is the third panel of the figure

The fourth panel plots the required continuation revenues :math:`G_t` implied by the original Ramsey plan

These figures help us understand the time inconsistency of the Ramsey plan


Further Intuition
^^^^^^^^^^^^^^^^^^

One feature to note is the large difference between :math:`\check \tau_{t+1}` and :math:`\tau_{t+1}` in the top panel of the figure

If the government is able to reset to a new Ramsey plan at time :math:`t`, it chooses a significantly higher tax rate than if it were required to maintain the original Ramsey plan

The intuition here is that the government is required to finance a given present value of expenditures with distorting taxes :math:`\tau`

The quadratic adjustment costs prevent firms from reacting strongly to variations in the tax rate for next period, which tilts a time :math:`t` Ramsey planner toward using time :math:`t+1` taxes

As was noted before, this is evident in :ref:`the first figure <fig_ES_plot_1>`, where the government taxes the next period heavily and then falls back to a constant tax from then on

This can also been seen in the third panel of :ref:`the second figure <fig_ES_plot_2>`, where the government pays off a significant portion of the debt using the first period tax rate

The similarities between the graphs in the last two panels of :ref:`the second figure <fig_ES_plot_2>` reveals that there is a one-to-one mapping between :math:`G` and :math:`\mu`

The Ramsey plan can then only be time consistent if :math:`G_t` remains constant over time, which will not be true in general




.. _sec:credible:

Credible Policy
==================

.. index::
    single: Ramsey Problem; Credible Policy

We express the  theme of this section in the following: In general, a continuation of a Ramsey plan is not a Ramsey plan

This is sometimes summarized by saying that a Ramsey plan is not *credible*

On the other hand, a continuation of a credible plan is a credible plan

The literature on a credible public policy (:cite:`chari1990sustainable` and :cite:`stokey1989reputation`)  arranges strategies and incentives so that public policies can be implemented by a *sequence* of government decision makers instead of  a single Ramsey planner who chooses an entire sequence of history-dependent actions once and for all at time :math:`t=0`

Here  we confine ourselves to  sketching how recursive methods can be  used to characterize credible policies in  our model

A key reference on these topics is  :cite:`chang1998credible`

A credibility problem arises because we assume that the timing of decisions differs from those for a Ramsey problem


A **sequential timing protocol** is a protocol such that

#. At each :math:`t \geq 0`, given :math:`Q_t` and expectations about a continuation tax policy :math:`\{\tau_{s+1}\}_{s=t}^\infty` and a continuation price sequence :math:`\{p_{s+1}\}_{s=t}^\infty`, the representative firm chooses :math:`u_t`

#. At each :math:`t`, given :math:`(Q_t, u_t)`, a government chooses :math:`\tau_{t+1}`

Item (2) captures that taxes are now set sequentially, the time :math:`t+1` tax being set *after* the government has observed :math:`u_t`

Of course, the representative firm sets :math:`u_t` in light of its expectations of how the government will ultimately choose to set future taxes

A credible tax plan :math:`\{\tau_{s+1}\}_{s=t}^\infty`

* is anticipated by the representative firm, and

* is one that a time :math:`t` government chooses to confirm

We use the following recursion, closely related to but different from  :eq:`ES_28a`, to define the continuation value function for the government:

.. math::
    :label: foo1

    J_t = A_0 Q_{t} - \frac{A_1}{2} Q_{t}^2 - \frac{d}{2} u_{t}^2 + \beta J_{t+1} (\tau_{t+1},G_{t+1})


This differs from :eq:`ES_28a` because

* continuation values are now allowed to depend explicitly on values of the choice :math:`\tau_{t+1}`, and

* continuation government revenue to be raised :math:`G_{t+1}`  need not be ones called for by the prevailing government policy

Thus, deviations from that policy are allowed, an alteration that recognizes that :math:`\tau_t` is chosen sequentially

Express the government budget constraint as requiring that :math:`G_0` solves the difference equation

.. math::
    :label: ES_govt_budget_sequential

    G_t = \beta \tau_{t+1} Q_{t+1} + \beta G_{t+1}, \quad t \geq 0


subject to the terminal condition :math:`\lim_{t \rightarrow + \infty} \beta^t G_t= 0`

Because the government is choosing sequentially, it is convenient to

* take :math:`G_t` as a state variable at :math:`t` and

* to regard the time :math:`t` government as choosing :math:`(\tau_{t+1}, G_{t+1})` subject to constraint :eq:`ES_govt_budget_sequential`

To express the notion of a credible government plan concisely, we expand the strategy space by also adding :math:`J_t` itself as a state variable and allowing policies to take the following recursive forms [#fn_i]_

Regard :math:`J_0` as an a discounted present value promised to the Ramsey planner and take it as an initial condition.

Then after choosing :math:`u_0` according to

.. math::
    :label: ES_29a

    u_0 = \upsilon(Q_0, G_0, J_0),


choose subsequent taxes, outputs, *and* continuation values according to recursions that can be represented as

.. math::
    :label: ES_30

    \hat \tau_{t+1}
    = \tau(Q_t, u_t, G_t, J_t )


.. math::
    :label: ES_31

    u_{t+1}
    = \xi (Q_t, u_t, G_t, J_t,{\tau_{t+1}} )


.. math::
    :label: ES_32

    G_{t+1}
    = \beta^{-1} G_t -  \tau_{t+1} Q_{t+1}


.. math::
    :label: ES_33

    J_{t+1}(\tau_{t+1}, G_{t+1})
    = \nu(Q_t, u_t, G_{t+1}, J_t, \tau_{t+1} )


Here

* :math:`\hat \tau_{t+1}` is the time :math:`t+1` government action called for by the plan, while

* :math:`\tau_{t+1}` is possibly some one-time deviation that the time :math:`t+1` government contemplates and

* :math:`G_{t+1}` is the associated continuation tax collections

The plan is said to be **credible** if, for each :math:`t` and each state :math:`(Q_t, u_t, G_t, J_t)`, the plan satisfies the incentive constraint

.. math::
    :label: ES_34

    \begin{aligned}
    J_t
    & = A_0 Q_{t} - \frac{A_1}{2} Q_{t}^2 - \frac{d}{2} u_{t}^2
        + \beta J_{t+1} (\hat \tau_{t+1}, \hat G_{t+1})
    \\
    & \geq A_0 Q_{t} - \frac{A_1}{2} Q_{t}^2
        - \frac{d}{2} u_{t}^2 +  \beta J_{t+1} ( \tau_{t+1}, G_{t+1})
    \end{aligned}


for all tax rates :math:`\tau_{t+1} \in {\mathbb R}` available to the government

Here :math:`\hat G_{t+1} = \frac{G_t - \hat \tau_{t+1} Q_{t+1}}{\beta}`

* Inequality expresses that continuation values adjust to deviations in ways that discourage the government from deviating from the prescribed :math:`\hat \tau_{t+1}`

* Inequality :eq:`ES_34` indicates that *two* continuation values :math:`J_{t+1}` contribute to sustaining time :math:`t` promised value :math:`J_t`

  * :math:`J_{t+1} (\hat \tau_{t+1}, \hat G_{t+1})` is the continuation value when the government chooses to confirm the private sector's expectation, formed according to the decision rule :eq:`ES_30` [#fn_j]_
  * :math:`J_{t+1}(\tau_{t+1}, G_{t+1})` tells the continuation consequences should the government disappoint the private sector's expectations

The internal structure of a credible  plan deters deviations from it

That :eq:`ES_34` maps *two* continuation values :math:`J_{t+1}(\tau_{t+1},G_{t+1})` and :math:`J_{t+1}(\hat \tau_{t+1},\hat G_{t+1})` into one promised value :math:`J_t` reflects how a credible plan arranges a system of private sector expectations that induces the government to choose to confirm them

Chang :cite:`chang1998credible` builds on how inequality :eq:`ES_34` maps two continuation values into one

**Remark** Let :math:`{\mathcal J}` be the set of values associated with credible plans

Every value :math:`J \in {\mathcal J}` can be attained by a credible plan that has a recursive representation of form form :eq:`ES_30`, :eq:`ES_31`, :eq:`ES_32`

The set of values can be computed as the largest fixed point of an operator that maps sets of candidate values into sets of values

Given a value within this set, it is possible to construct a government strategy of the  recursive form :eq:`ES_30`, :eq:`ES_31`, :eq:`ES_32` that attains that value

In many cases, there is  a **set** of values and associated credible plans

In those cases where the Ramsey outcome is credible, a multiplicity of credible plans is a key part of the story because, as we have seen earlier, a continuation of a Ramsey plan is not a Ramsey plan

For it to be credible, a Ramsey outcome must be supported by a worse outcome associated with another plan, the prospect of reversion to which sustains the Ramsey outcome



Concluding remarks
==================

The term 'optimal policy', which pervades an important applied monetary economics literature, means different things under different timing protocols

Under the 'static' Ramsey timing protocol (i.e., choose a sequence once-and-for-all), we obtain a unique plan

Here the phrase 'optimal policy' seems to fit well, since the Ramsey planner optimally reaps early benefits from influencing the private sector's beliefs about the government's later actions

When we adopt the sequential timing protocol associated with credible public policies, 'optimal policy' is a more ambiguous description

There is a multiplicity of credible plans

True, the theory explains how it is optimal for the government to confirm the private sector's expectations about its actions along a credible plan

But some credible plans have very bad outcomes

These bad outcomes are central to the theory because it is the presence of bad credible plans that makes possible better ones by sustaining the low continuation values that appear in the second line of incentive constraint :eq:`ES_34`

Recently, many have taken for granted that 'optimal policy' means 'follow the Ramsey plan' [#fn_k]_

In pursuit of more attractive ways to describe a Ramsey plan when policy making is in practice done sequentially, some writers have repackaged a Ramsey plan in the following way

*  Take a Ramsey *outcome* - a sequence of endogenous variables under a Ramsey plan - and reinterpret it (or perhaps only a subset of its variables) as a *target path* of relationships among outcome variables to be assigned to a sequence of policy makers [#fn_l]_

*  If appropriate (infinite dimensional) invertibility conditions are satisfied, it can happen that following the Ramsey plan is the *only* way to hit the target path [#fn_m]_

*  The spirit of this work is to say, "in a democracy we are obliged to live with the sequential timing protocol, so let's constrain policy makers' objectives in ways that will force them to follow a Ramsey plan in spite of their benevolence" [#fn_n]_

*  By this slight of hand, we acquire a theory of an *optimal outcome target path*

This 'invertibility' argument leaves open two important loose ends:

#. implementation, and

#. time consistency

As for (1), repackaging a Ramsey plan (or the tail of a Ramsey plan) as a
target outcome sequence does not confront the delicate issue of *how* that
target path is to be implemented [#fn_o]_

As for (2), it is an interesting question whether the 'invertibility' logic
can repackage and conceal a Ramsey plan well enough to make policy makers
forget or ignore the benevolent intentions that give rise to the time
inconsistency of a Ramsey plan in the first place

To attain such an optimal output path, policy makers must forget their
benevolent intentions because there will inevitably occur temptations to
deviate from that target path, and the implied relationship among variables
like inflation, output, and interest rates along it

**Remark** The continuation of such an optimal target path is not an optimal target path




.. rubric:: Footnotes

.. [#fn_a]
   We could also call a competitive equilibrium a rational expectations
   equilibrium.

.. [#fn_b]
   It is important not to set :math:`q_t = Q_t` prematurely. To make the
   firm a price taker, this equality should be imposed *after* and not
   *before* solving the firm's optimization problem.

.. [#fn_c]
    We could instead, perhaps with more accuracy, define a promised marginal
    value as :math:`\beta (A_0 - A_1 Q_{t+1} ) - \beta \tau_{t+1} +
    u_{t+1}/\beta`, since this is the object to which the firm's
    first-order condition instructs it to equate to the marginal cost :math:`d
    u_t` of :math:`u_t = q_{t+1} - q_t`.
    This choice would align better with how Chang :cite:`chang1998credible` chose to express his
    competitive equilibrium recursively.  But given :math:`(u_t,
    Q_t)`, the representative firm knows :math:`(Q_{t+1},\tau_{t+1})`, so it
    is adequate to take :math:`u_{t+1}` as the intermediate variable that
    summarizes how :math:`\vec \tau_{t+1}` affects the firm's choice of
    :math:`u_t`.


.. [#fn_e]
   As promised, :math:`\tau_t` does not appear in the Ramsey planner's
   decision rule for :math:`\tau_{t+1}`.

.. [#fn_f]
   The continuation revenues :math:`G_t` are the time :math:`t` present
   value of revenues that must be raised to satisfy the original time
   :math:`0` government intertemporal budget constraint, taking into
   account the revenues already raised from :math:`s=1, \ldots, t` under
   the original Ramsey plan.

.. [#fn_g]
   For example, let the Ramsey plan yield time :math:`1` revenues
   :math:`Q_1 \tau_1`. Then at time :math:`1`, a continuation Ramsey
   planner would want to raise continuation revenues, expressed in units
   of time :math:`1` goods, of
   :math:`\tilde G_1 := \frac{G - \beta Q_1 \tau_1}{\beta}`. To
   finance the remainder revenues, the continuation Ramsey planner would
   find a continuation Lagrange multiplier :math:`\mu` by applying the
   three-step procedure from the previous section to revenue
   requirements :math:`\tilde G_1`.

.. [#fn_h]
   It can be verified that this formula puts non-zero weight only on the
   components :math:`1` and :math:`Q_t` of :math:`z_t`.

.. [#fn_i]
   This choice is the key to what :cite:`Ljungqvist2012` call 'dynamic programming squared'.

.. [#fn_j]
   Note the double role played by :eq:`ES_30`: as decision rule for the government
   and as the private sector's rule for forecasting government actions.

.. [#fn_k]
   It is possible to read :cite:`woodfordinterest` and
   :cite:`giannoni2010optimal` as making some carefully qualified statements
   of this type. Some of the qualifications can be interpreted as advice
   'eventually' to follow a tail of a Ramsey plan.

.. [#fn_l]
   In our model, the Ramsey outcome would be a path :math:`(\vec p, \vec Q)`.

.. [#fn_m]
   See :cite:`giannoni2010optimal`.

.. [#fn_n]
   Sometimes the analysis is framed in terms of following the Ramsey
   plan only from some future date :math:`T` onwards.

.. [#fn_o]
   See :cite:`bassetto2005equilibrium`  and :cite:`atkeson2010sophisticated`.
