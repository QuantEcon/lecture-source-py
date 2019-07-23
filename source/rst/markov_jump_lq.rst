.. _markov_jump_lq:

.. include:: /_static/includes/lecture_howto_py.raw

.. index::
    single: python


************************************************
Markov Jump Linear Quadratic Dynamic Programming
************************************************

.. contents:: :depth: 2

**Co-authors:** `Sebastian Graves <https://github.com/sebgraves>`__ and Zejin Shi


Overview
========

This lecture describes **Markov jump linear quadratic dynamic programming**, an extension of the method described in
:doc:`LQ dynamic programming <lqcontrol>`.

Markov jump linear quadratic dynamic programming is  described and analyzed in :cite:`do1999solutions` and the references cited there.

The method has been applied to problems in  macroeconomics and monetary economics by :cite:`svensson2008optimal` and :cite:`SvenssonWilliams2009`.

The periodic models of seasonality described in chapter 14 of :cite:`HS2013` are a special case of Markov jump linear quadratic problems.


**Markov jump linear quadratic dynamic programming** combines advantages
of

-  the computational simplicity of **linear quadratic dynamic
   programming**, with

-  the ability of **finite state Markov chains** to represent
   interesting patterns of random variation.

The idea is to replace the constant matrices that define a **linear quadratic dynamic programming problem** 
with :math:`N` sets of matrices that are fixed functions of
the state of an :math:`N` state Markov chain.

The state of the Markov chain together with   the continuous :math:`n \times 1` state vector :math:`x_t` form the state of the system.

For the class of infinite horizon problems being studied in this lecture, we obtain  :math:`N` interrelated
matrix Riccati equations that determine :math:`N` optimal value
functions and :math:`N` linear decision rules.

One of these value functions and one of these decision rules apply in each of the :math:`N` Markov states.

That is,  when the Markov state is in state :math:`j`, the value function and the decision rule
for state :math:`j` prevails.


Review of useful LQ dynamic programming formulas
================================================

To begin,it is handy to have the following reminder in mind.

A **linear quadratic dynamic programming problem** consists of a scalar
discount factor :math:`\beta \in (0,1)`, an :math:`n\times 1` state
vector :math:`x_t`, an initial condition for :math:`x_0`, a
:math:`k \times 1` control vector :math:`u_t`, a :math:`p \times 1`
random shock vector :math:`w_{t+1}` and the following two triples of
matrices:

-  A triple of matrices :math:`(R, Q, W)` defining a loss function

.. math::  r(x_t, u_t) = x_t' R x_t + u_t' Q u_t + 2 u_t' W x_t

-  a triple of matrices :math:`(A, B, C)` defining a state-transition
   law

.. math::  x_{t+1} = A x_t + B u_t + C w_{t+1}

The problem is

.. math::


   -x_0' P x_0 - \rho = \min_{\{u_t\}_{t=0}^\infty} E \sum_{t=0}^{\infty} \beta^t r(x_t, u_t)

subject to the transition law for the state.

The optimal decision rule for this problem has the form

.. math::  u_t = - F x_t

and the optimal value function is of the form

.. math::  -\left( x_t' P x_t  + \rho \right)

where :math:`P` solves the algebraic matrix Riccati equation

.. math::


   P = R+ \beta A' P A
             -(\beta B'  P A + W)' (Q + \beta B P B )^{-1} (\beta B P A + W)

and the constant :math:`\rho` satisfies

.. math::

   \rho = \beta
     \left( \rho + {\rm trace}(P C C') \right)

and the matrix :math:`F` in the decision rule for :math:`u_t` satisfies

.. math::


   F = (Q + \beta  B' P B)^{-1} (\beta (B' P A )+ W)


With the preceding formulas in mind, we are ready to approach Markov Jump linear quadratic dynamic programming.


Linked Ricatti equations for Markov LQ dynamic programming
==========================================================

The key idea is to make the matrices :math:`A, B, C, R, Q, W` fixed
functions of a finite state :math:`s` that is governed by an :math:`N`
state Markov chain.

This makes decision rules depend on the Markov
state, and so fluctuate through time in limited ways.

In particular, we use the following extension of a discrete-time linear
quadratic dynamic programming problem.

We let :math:`s(t) \equiv s_t \in [1, 2, \ldots, N]` be a time :math:`t` realization of an
:math:`N`-state Markov chain with transition matrix :math:`\Pi` having
typical element :math:`\Pi_{ij}`.

Here :math:`i` denotes today and
:math:`j` denotes tomorrow and

.. math::  \Pi_{ij} = {\rm Prob}(s_{t+1} = j |s_t = i)

We’ll switch between labeling today’s state as :math:`s(t)` and
:math:`i` and between labeling tomorrow’s state as :math:`s(t+1)` or
:math:`j`.

The decision-maker solves the minimization problem:

.. math::

  \min_{\{u_t\}_{t=0}^\infty} E \sum_{t=0}^{\infty} \beta^t r(x_t, s(t), u_t)


with

.. math::

  r(x_t, s(t), u_t) = -( x_t' R(s_t) x_t + u_t' Q(s_t) u_t + 2 u_t' W(s_t) x_t)


subject to linear laws of motion with matrices :math:`(A,B,C)` each
possibly dependent on the Markov-state-\ :math:`s_t`:

.. math::


    x_{t+1} = A(s_t) x_t + B(s_t) u_t + C(s_t) w_{t+1}

where :math:`\{w_{t+1}\}` is an i.i.d. stochastic process with
:math:`w_{t+1} \sim {\cal N}(0,I)`.

The optimal decision rule for this problem has the form

.. math::  u_t = - F(s_t) x_t

and the optimal value functions are of the form

.. math::  -\left( x_t' P(s_t) x_t  + \rho(s_t) \right)

or equivalently

.. math::  -x_t' P_i x_t - \rho_i

The optimal value functions :math:`- x' P_i x - \rho_i` for
:math:`i = 1, \ldots, n` satisfy the :math:`N`
interrelated Bellman equations

.. math::

    -x' P_i x - \rho_i  = \max_u - \biggl[ x'R_i x + u' Q_i u + 2 u' W_i x \\
                    \beta \sum_j \Pi_{ij}E ((A_i x + B_i u + C_i w)' P_j
                    (A_i x + B_i u + C_i w) x + \rho_j) \biggr]


The matrices :math:`P(s(t)) = P_i` and the scalars
:math:`\rho(s_t) = \rho_i, i = 1, \ldots`, n satisfy the following stacked system of
**algebraic matrix Riccati** equations:

.. math::


   P_i = R_i + \beta \sum_j A_i' P_j A_i
    \Pi_{ij}
             -\sum_j \Pi_{ij}[ (\beta B_i'  P_j A_i + W_i)' (Q + \beta B_i' P_j B_i)^{-1}
             (\beta B_i' P_j A_i + W_i)]

.. math::

   \rho_i = \beta
    \sum_j \Pi_{ij} ( \rho_j + {\rm trace}(P_j C_i C_i') )

and the :math:`F_i` in the optimal decision rules are

.. math::


   F_i = (Q_i + \beta \sum_j \Pi_{ij} B_i' P_j B_i)^{-1}
   (\beta \sum_j \Pi_{ij}(B_i' P_j A_i )+ W_i)



Applications
============

We now describe some Python code and a few examples that put the code to work.

To begin, we import these Python modules




.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    %matplotlib inline

.. code-block:: python3

    # set discount factor
    β = 0.95

Example 1
=========

This example is a version of a classic problem of optimally adjusting a variable :math:`k_t` to a target level in the face of costly adjustment.

This provides a  model of gradual adjustment.

Given :math:`k_{0}`, the objective function is

.. math::


   \max_{\left\{ k_{t}\right\} _{t=1}^{\infty}}E_{0}\sum_{t=0}^{\infty}\beta^{t}r\left(s_{t},k_{t}\right)

where the one-period payoff function is

.. math::


   r\left(s_{t},k_{t}\right)=f_{1}\left(s_{t}\right)k_{t}-f_{2}\left(s_{t}\right)k_{t}^{2}-d\left(s_{t}\right)\left(k_{t+1}-k_{t}\right)^{2},

:math:`E_0` is a mathematical expectation conditioned on time :math:`0`
information :math:`x_0, s_0`

and the transition law for continuous state variable :math:`k_t` is

.. math::


   k_{t+1}-k_{t}=u_{t}

We can think of :math:`k_t` as the decision-maker’s capital and
:math:`u_t` as costs of adjusting the level of capital.

We assume that :math:`f_{1}\left(s_{t}\right)>0`,
:math:`f_{2}\left(s_{t}\right)>0`, and :math:`d\left(s_{t}\right)>0`.

Denote the state transition matrix for Markov state
:math:`s_{t}\in\left\{ \bar{s}_{1},\bar{s}_{2}\right\}` as :math:`\Pi`:

.. math::


   \Pr \left(s_{t+1}=\bar{s}_{j}\mid s_{t}=\bar{s}_{i}\right)=\Pi_{ij}

Let :math:`x_{t}=\begin{bmatrix} k_{t}\\ 1 \end{bmatrix}`

We can represent the one-period payoff function
:math:`r\left(s_{t},k_{t}\right)` and the state-transition law as

.. math::


   r\left(s_{t},k_{t}\right)   =f_{1}\left(s_{t}\right)k_{t}-f_{2}\left(s_{t}\right)k_{t}^{2}-d\left(s_{t}\right)u_{t}{}^{2} \\
       =-\left(x_{t}^{\prime}\underset{\equiv R\left(s_{t}\right)}{\underbrace{\begin{bmatrix}
   f_{2}\left(s_{t}\right) & -\frac{f_{1}\left(s_{t}\right)}{2}\\
   -\frac{f_{1}\left(s_{t}\right)}{2} & 0
   \end{bmatrix}}}x_{t}+\underset{\equiv Q\left(s_{t}\right)}{\underbrace{d\left(s_{t}\right)}}u_{t}{}^{2}\right)

.. math::


   x_{t+1}=\begin{bmatrix}
   k_{t+1}\\
   1
   \end{bmatrix}=\underset{\equiv A\left(s_{t}\right)}{\underbrace{I_{2}}}x_{t}+\underset{\equiv B\left(s_{t}\right)}{\underbrace{\begin{bmatrix}
   1\\
   0
   \end{bmatrix}}}u_{t}

.. code-block:: python3

    def construct_arrays1(f1_vals=[1. ,1.],
                          f2_vals=[1., 1.],
                          d_vals=[1., 1.]):
        """
        Construct matrices that maps the problem described in example 1
        into a Markov jump linear quadratic dynamic programming problem.
        """

        # number of Markov states
        m = len(f1_vals)
        # number of state and control variables
        n, k = 2, 1

        # construct sets of matrices for each state
        As = [np.eye(n) for i in range(m)]
        Bs = [np.array([[1, 0]]).T for i in range(m)]

        Rs = np.zeros((m, n, n))
        Qs = np.zeros((m, k, k))

        for i in range(m):
            Rs[i, 0, 0] = f2_vals[i]
            Rs[i, 1, 0] = - f1_vals[i] / 2
            Rs[i, 0, 1] = - f1_vals[i] / 2

            Qs[i, 0, 0] = d_vals[i]

        Cs, Ns = None, None

        # compute the optimal k level of the payoff function in each state
        k_star = np.empty(m)
        for i in range(m):
            k_star[i] = f1_vals[i] / (2 * f2_vals[i])

        return Qs, Rs, Ns, As, Bs, Cs, k_star

The continuous part of the state :math:`x_t` consists of two variables,
namely, :math:`k_t` and a constant term.

.. code-block:: python3

    state_vec1 = ["k", "constant term"]

We start with a Markov transition matrix that makes the Markov state be
strictly periodic:

.. math::


   \Pi_{1}=\begin{bmatrix}
   0 & 1\\
   1 & 0
   \end{bmatrix},

We set :math:`f_1(s_t)` and :math:`f_2(s_t)` to be independent of the
Markov state :math:`s_t`

.. math::


   f_1(\bar{s_1})=f_1(\bar{s_2}) = 1,

.. math::


   f_2(\bar{s_1})=f_2(\bar{s_2}) = 1

In contrast to :math:`f_1(s_t)` and :math:`f_2(s_t)`, we make the
adjustment cost :math:`d(s_t)` vary across Markov states :math:`s_t`.

We set the adjustment cost to be lower in Markov state :math:`\bar{s}_2`

.. math::


   d(\bar{s_1})=1, d(\bar{s_2}) = 0.5

The following code forms a Markov switching LQ problem and computes the
optimal value functions and optimal decision rules for each Markov state

.. code-block:: python3

    # construct Markov transition matrix
    Π1 = np.array([[0., 1.],
                   [1., 0.]])

.. code-block:: python3

    # construct matrices
    Qs, Rs, Ns, As, Bs, Cs, k_star = construct_arrays1(d_vals=[1., 0.5])

.. code-block:: python3

    # construct a Markov Jump LQ problem
    ex1_a = qe.LQMarkov(Π1, Qs, Rs, As, Bs, Cs=Cs, Ns=Ns, beta=β)
    # solve for optimal value functions and decision rules
    ex1_a.stationary_values();

Let’s look at the value function matrices and the decision rules for
each Markov state

.. code-block:: python3

    # P(s)
    ex1_a.Ps

.. code-block:: python3

    # d(s) = 0, since there is no randomness
    ex1_a.ds

.. code-block:: python3

    # F(s)
    ex1_a.Fs

Now we’ll plot the decision rules and see if they make sense

.. code-block:: python3

    # Plot the optimal decision rules
    k_grid = np.linspace(0., 1., 100)
    # optimal choice in state s1
    u1_star = - ex1_a.Fs[0, 0, 1] - ex1_a.Fs[0, 0, 0] * k_grid
    # optimal choice in state s2
    u2_star = - ex1_a.Fs[1, 0, 1] - ex1_a.Fs[1, 0, 0] * k_grid

    fig, ax = plt.subplots()
    ax.plot(k_grid, k_grid + u1_star, label="$\overline{s}_1$ (high)")
    ax.plot(k_grid, k_grid + u2_star, label="$\overline{s}_2$ (low)")

    # the optimal k*
    ax.scatter([0.5, 0.5], [0.5, 0.5], marker="*")
    ax.plot([k_star[0], k_star[0]], [0., 1.0], '--')

    # 45 degree line
    ax.plot([0., 1.], [0., 1.], '--', label="45 degree line")

    ax.set_xlabel("$k_t$")
    ax.set_ylabel("$k_{t+1}$")
    ax.legend()
    plt.show()

The above graph plots $k_{t+1}= k_t + u_t = k_t - F x_t $ as an affine
(i.e., linear in :math:`k_t` plus a constant) function of :math:`k_t`
for both Markov states :math:`s_t`.

It also plots the 45 degree line.

Notice that the two :math:`s_t`-dependent *closed loop* functions that
determine :math:`k_{t+1}` as functions of :math:`k_t` share the same
rest point (also called a fixed point) at :math:`k_t = 0.5`.

Evidently, the optimal decision rule in Markov state :math:`\bar{s}_2`,
in which the adjustment cost is lower, makes :math:`k_{t+1}` a flatter
function of :math:`k_t` in Markov state :math:`\bar{s}_2`.

This happens because when :math:`k_t` is not at its fixed point,
:math:`\left| u_t(\bar{s}_2)\right| > \left| u_t(\bar{s}_1)\right|`, so
that the decision-maker adjusts toward the fixed point faster when
the Markov state :math:`s_t` takes a value that makes it cheaper.

.. code-block:: python3

    # compute time series
    T = 20
    x0 = np.array([[0., 1.]]).T
    x_path = ex1_a.compute_sequence(x0, ts_length=T)[0]

    fig, ax = plt.subplots()
    ax.plot(range(T), x_path[0, :-1])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$k_t$")
    ax.set_title("Optimal path of $k_t$")
    plt.show()

Now we’ll depart from the preceding transition matrix that made the
Markov state be strictly periodic.

We’ll begin with symmetric transition matrices of the form

.. math::


   \Pi_{2}=\begin{bmatrix}
   1-\lambda & \lambda\\
   \lambda & 1-\lambda
   \end{bmatrix}.

.. code-block:: python3

    λ = 0.8 # high λ
    Π2 = np.array([[1-λ, λ],
                   [λ, 1-λ]])

    ex1_b = qe.LQMarkov(Π2, Qs, Rs, As, Bs, Cs=Cs, Ns=Ns, beta=β)
    ex1_b.stationary_values();
    ex1_b.Fs

.. code-block:: python3

    λ = 0.2 # low λ
    Π2 = np.array([[1-λ, λ],
                   [λ, 1-λ]])

    ex1_b = qe.LQMarkov(Π2, Qs, Rs, As, Bs, Cs=Cs, Ns=Ns, beta=β)
    ex1_b.stationary_values();
    ex1_b.Fs

We can plot optimal decision rules associated with different
:math:`\lambda` values.

.. code-block:: python3

    λ_vals = np.linspace(0., 1., 10)
    F1 = np.empty((λ_vals.size, 2))
    F2 = np.empty((λ_vals.size, 2))

    for i, λ in enumerate(λ_vals):
        Π2 = np.array([[1-λ, λ],
                       [λ, 1-λ]])

        ex1_b = qe.LQMarkov(Π2, Qs, Rs, As, Bs, Cs=Cs, Ns=Ns, beta=β)
        ex1_b.stationary_values();
        F1[i, :] = ex1_b.Fs[0, 0, :]
        F2[i, :] = ex1_b.Fs[1, 0, :]

.. code-block:: python3

    for i, state_var in enumerate(state_vec1):
        fig, ax = plt.subplots()
        ax.plot(λ_vals, F1[:, i], label="$\overline{s}_1$", color="b")
        ax.plot(λ_vals, F2[:, i], label="$\overline{s}_2$", color="r")

        ax.set_xlabel("$\lambda$")
        ax.set_ylabel("$F(s_t)$")
        ax.set_title(f"Coefficient on {state_var}")
        ax.legend()
        plt.show()

Notice how the decision rules’ constants and slopes behave as functions
of :math:`\lambda`.

Evidently, as the Markov chain becomes *more nearly periodic* (i.e., as
:math:`\lambda \rightarrow 1`), the dynamic program adjusts capital
faster in the low adjustment cost Markov state to take advantage of what
is only temporarily a more favorable time to invest.

Now let’s study situations in which the Markov transition matrix
:math:`\Pi` is asymmetric

.. math::


   \Pi_{3}=\begin{bmatrix}
   1-\lambda & \lambda\\
   \delta & 1-\delta
   \end{bmatrix}.

.. code-block:: python3

    λ, δ = 0.8, 0.2
    Π3 = np.array([[1-λ, λ],
                   [δ, 1-δ]])

    ex1_b = qe.LQMarkov(Π3, Qs, Rs, As, Bs, Cs=Cs, Ns=Ns, beta=β)
    ex1_b.stationary_values();
    ex1_b.Fs

We can plot optimal decision rules for different :math:`\lambda` and
:math:`\delta` values.

.. code-block:: python3

    λ_vals = np.linspace(0., 1., 10)
    δ_vals = np.linspace(0., 1., 10)

    λ_grid = np.empty((λ_vals.size, δ_vals.size))
    δ_grid = np.empty((λ_vals.size, δ_vals.size))
    F1_grid = np.empty((λ_vals.size, δ_vals.size, len(state_vec1)))
    F2_grid = np.empty((λ_vals.size, δ_vals.size, len(state_vec1)))

    for i, λ in enumerate(λ_vals):
        λ_grid[i, :] = λ
        δ_grid[i, :] = δ_vals
        for j, δ in enumerate(δ_vals):
            Π3 = np.array([[1-λ, λ],
                           [δ, 1-δ]])

            ex1_b = qe.LQMarkov(Π3, Qs, Rs, As, Bs, Cs=Cs, Ns=Ns, beta=β)
            ex1_b.stationary_values();
            F1_grid[i, j, :] = ex1_b.Fs[0, 0, :]
            F2_grid[i, j, :] = ex1_b.Fs[1, 0, :]

.. code-block:: python3

    for i, state_var in enumerate(state_vec1):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # high adjustment cost, blue surface
        ax.plot_surface(λ_grid, δ_grid, F1_grid[:, :, i], color="b")
        # low adjustment cost, red surface
        ax.plot_surface(λ_grid, δ_grid, F2_grid[:, :, i], color="r")
        ax.set_xlabel("$\lambda$")
        ax.set_ylabel("$\delta$")
        ax.set_zlabel("$F(s_t)$")
        ax.set_title(f"coefficient on {state_var}")
        plt.show()

The following code defines a wrapper function that computes optimal
decision rules for cases with different Markov transition matrices

.. code-block:: python3

    def run(construct_func, vals_dict, state_vec):
        """
        A Wrapper function that repeats the computation above
        for different cases
        """

        Qs, Rs, Ns, As, Bs, Cs, k_star = construct_func(**vals_dict)

        # symmetric Π
        # notice that pure periodic transition is a special case
        # when λ=1
        print("symmetric Π case:\n")
        λ_vals = np.linspace(0., 1., 10)
        F1 = np.empty((λ_vals.size, len(state_vec)))
        F2 = np.empty((λ_vals.size, len(state_vec)))

        for i, λ in enumerate(λ_vals):
            Π2 = np.array([[1-λ, λ],
                           [λ, 1-λ]])

            mplq = qe.LQMarkov(Π2, Qs, Rs, As, Bs, Cs=Cs, Ns=Ns, beta=β)
            mplq.stationary_values();
            F1[i, :] = mplq.Fs[0, 0, :]
            F2[i, :] = mplq.Fs[1, 0, :]

        for i, state_var in enumerate(state_vec):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(λ_vals, F1[:, i], label="$\overline{s}_1$", color="b")
            ax.plot(λ_vals, F2[:, i], label="$\overline{s}_2$", color="r")

            ax.set_xlabel("$\lambda$")
            ax.set_ylabel("$F(\overline{s}_t)$")
            ax.set_title(f"coefficient on {state_var}")
            ax.legend()
            plt.show()

        # plot optimal k*(s_t) and k that optimal policies are targeting
        # only for example 1
        if state_vec == ["k", "constant term"]:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(2):
                F = [F1, F2][i]
                c = ["b", "r"][i]
                ax.plot([0, 1], [k_star[i], k_star[i]], "--", color=c, label="$k^*(\overline{s}_"+str(i+1)+")$")
                ax.plot(λ_vals, - F[:, 1] / F[:, 0], color=c, label="$k^{target}(\overline{s}_"+str(i+1)+")$")

            # plot a vertical line at λ=0.5
            ax.plot([0.5, 0.5], [min(k_star), max(k_star)], "-.")

            ax.set_xlabel("$\lambda$")
            ax.set_ylabel("$k$")
            ax.set_title("Optimal k levels and k targets")
            ax.text(0.5, min(k_star)+(max(k_star)-min(k_star))/20, "$\lambda=0.5$")
            ax.legend(bbox_to_anchor=(1., 1.))
            plt.show()

        # asymmetric Π
        print("asymmetric Π case:\n")
        δ_vals = np.linspace(0., 1., 10)

        λ_grid = np.empty((λ_vals.size, δ_vals.size))
        δ_grid = np.empty((λ_vals.size, δ_vals.size))
        F1_grid = np.empty((λ_vals.size, δ_vals.size, len(state_vec)))
        F2_grid = np.empty((λ_vals.size, δ_vals.size, len(state_vec)))

        for i, λ in enumerate(λ_vals):
            λ_grid[i, :] = λ
            δ_grid[i, :] = δ_vals
            for j, δ in enumerate(δ_vals):
                Π3 = np.array([[1-λ, λ],
                               [δ, 1-δ]])

                mplq = qe.LQMarkov(Π3, Qs, Rs, As, Bs, Cs=Cs, Ns=Ns, beta=β)
                mplq.stationary_values();
                F1_grid[i, j, :] = mplq.Fs[0, 0, :]
                F2_grid[i, j, :] = mplq.Fs[1, 0, :]

        for i, state_var in enumerate(state_vec):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(λ_grid, δ_grid, F1_grid[:, :, i], color="b")
            ax.plot_surface(λ_grid, δ_grid, F2_grid[:, :, i], color="r")
            ax.set_xlabel("$\lambda$")
            ax.set_ylabel("$\delta$")
            ax.set_zlabel("$F(\overline{s}_t)$")
            ax.set_title(f"coefficient on {state_var}")
            plt.show()

To illustrate the code with another example, we shall set
:math:`f_2(s_t)` and :math:`d(s_t)` as constant functions and

.. math::


   f_1(\bar{s}_1) = 0.5, f_1(\bar{s}_2) = 1

Thus, the sole role of the Markov jump state :math:`s_t` is to identify
times in which capital is very productive and other times in which it is
less productive.

The example below reveals much about the structure of the optimum
problem and optimal policies.

Only :math:`f_1(s_t)` varies with :math:`s_t`.

So there are different :math:`s_t`-dependent optimal static :math:`k`
level in different states :math:`k^*(s_t)=\frac{f_1(s_t)}{2 f_2(s_t)}`,
values of :math:`k` that maximize one-period payoff functions in each
state.

We denote a target :math:`k` level as :math:`k^{target}(s_t)`, the fixed
point of the optimal policies in each state, given the value of
:math:`\lambda`.

We call :math:`k^{target}(s_t)` a “target” because in each Markov state
:math:`s_t`, optimal policies are contraction mappings and will push
:math:`k_t` towards a fixed point :math:`k^{target}(s_t)`.

When :math:`\lambda \rightarrow 0`, each Markov state becomes close to
absorbing state and consequently
:math:`k^{target}(s_t) \rightarrow k^*(s_t)`.

But when :math:`\lambda \rightarrow 1`, the Markov transition matrix
becomes more nearly periodic, so the optimum decision rules target more
at the optimal k level in the other state in order to enjoy higher
expected payoff in the next period.

The switch happens at :math:`\lambda = 0.5` when both states are equally
likely to be reached.

Below we plot an additional figure that shows optimal :math:`k` levels
in the two states Markov jump state and also how the targeted :math:`k`
levels change as :math:`\lambda` changes.

.. code-block:: python3

    run(construct_arrays1, {"f1_vals":[0.5, 1.]}, state_vec1)



Set :math:`f_1(s_t)` and :math:`d(s_t)` as constant functions and

.. math::


   f_2(\bar{s}_1) = 0.5, f_2(\bar{s}_2) = 1

.. code-block:: python3

    run(construct_arrays1, {"f2_vals":[0.5, 1.]}, state_vec1)

Example 2
=========

We now add to the example 1 setup another state variable :math:`w_t`
that follows the evolution law

.. math::


   w_{t+1}=\alpha_{0}\left(s_{t}\right)+\rho\left(s_{t}\right)w_{t}+\sigma\left(s_{t}\right)\epsilon_{t+1},\quad\epsilon_{t+1}\sim N\left(0,1\right)

We think of :math:`w_t` as a rental rate or tax rate that the decision
maker pays each period for :math:`k_t`.

To capture this idea, we add to the decision-maker’s one-period payoff
function the product of :math:`w_t` and :math:`k_t`

.. math::


   r\left(s_{t},k_{t},w_{t}\right)=f_{1}\left(s_{t}\right)k_{t}-f_{2}\left(s_{t}\right)k_{t}^{2}-d\left(s_{t}\right)\left(k_{t+1}-k_{t}\right)^{2}-w_{t}k_{t},

We now let the continuous part of the state at time :math:`t` be
:math:`x_{t}=\begin{bmatrix} k_{t}\\ 1\\ w_{t} \end{bmatrix}`
and continue to set the control :math:`u_{t}=k_{t+1}-k_{t}`.

We can write the one-period payoff function
:math:`r\left(s_{t},k_{t},w_{t}\right)` and the state-transition law as

.. math::


   r\left(s_{t},k_{t},w_{t}\right) =f_{1}\left(s_{t}\right)k_{t}-f_{2}\left(s_{t}\right)k_{t}^{2}-d\left(s_{t}\right)\left(k_{t+1}-k_{t}\right)^{2}-w_{t}k_{t} \\
       =-\left(x_{t}^{\prime}\underset{\equiv R\left(s_{t}\right)}{\underbrace{\begin{bmatrix}
   f_{2}\left(s_{t}\right) & -\frac{f_{1}\left(s_{t}\right)}{2} & \frac{1}{2}\\
   -\frac{f_{1}\left(s_{t}\right)}{2} & 0 & 0\\
   \frac{1}{2} & 0 & 0
   \end{bmatrix}}}x_{t}+\underset{\equiv Q\left(s_{t}\right)}{\underbrace{d\left(s_{t}\right)}}u_{t}^{2}\right),

and

.. math::


   x_{t+1}=\begin{bmatrix}
   k_{t+1}\\
   1\\
   w_{t+1}
   \end{bmatrix}=\underset{\equiv A\left(s_{t}\right)}{\underbrace{\begin{bmatrix}
   1 & 0 & 0\\
   0 & 1 & 0\\
   0 & \alpha_{0}\left(s_{t}\right) & \rho\left(s_{t}\right)
   \end{bmatrix}}}x_{t}+\underset{\equiv B\left(s_{t}\right)}{\underbrace{\begin{bmatrix}
   1\\
   0\\
   0
   \end{bmatrix}}}u_{t}+\underset{\equiv C\left(s_{t}\right)}{\underbrace{\begin{bmatrix}
   0\\
   0\\
   \sigma\left(s_{t}\right)
   \end{bmatrix}}}\epsilon_{t+1}

.. code-block:: python3

    def construct_arrays2(f1_vals=[1. ,1.],
                          f2_vals=[1., 1.],
                          d_vals=[1., 1.],
                          α0_vals=[1., 1.],
                          ρ_vals=[0.9, 0.9],
                          σ_vals=[1., 1.]):
        """
        Construct matrices that maps the problem described in example 2
        into a Markov jump linear quadratic dynamic programming problem.
        """

        m = len(f1_vals)
        n, k, j = 3, 1, 1

        Rs = np.zeros((m, n, n))
        Qs = np.zeros((m, k, k))
        As = np.zeros((m, n, n))
        Bs = np.zeros((m, n, k))
        Cs = np.zeros((m, n, j))

        for i in range(m):
            Rs[i, 0, 0] = f2_vals[i]
            Rs[i, 1, 0] = - f1_vals[i] / 2
            Rs[i, 0, 1] = - f1_vals[i] / 2
            Rs[i, 0, 2] = 1/2
            Rs[i, 2, 0] = 1/2

            Qs[i, 0, 0] = d_vals[i]

            As[i, 0, 0] = 1
            As[i, 1, 1] = 1
            As[i, 2, 1] = α0_vals[i]
            As[i, 2, 2] = ρ_vals[i]

            Bs[i, :, :] = np.array([[1, 0, 0]]).T

            Cs[i, :, :] = np.array([[0, 0, σ_vals[i]]]).T

        Ns = None
        k_star = None

        return Qs, Rs, Ns, As, Bs, Cs, k_star

.. code-block:: python3

    state_vec2 = ["k", "constant term", "w"]



Only :math:`d(s_t)` depends on :math:`s_t`.

.. code-block:: python3

    run(construct_arrays2, {"d_vals":[1., 0.5]}, state_vec2)



Only :math:`f_1(s_t)` depends on :math:`s_t`.

.. code-block:: python3

    run(construct_arrays2, {"f1_vals":[0.5, 1.]}, state_vec2)



Only :math:`f_2(s_t)` depends on :math:`s_t`.

.. code-block:: python3

    run(construct_arrays2, {"f2_vals":[0.5, 1.]}, state_vec2)



Only :math:`\alpha_0(s_t)` depends on :math:`s_t`.

.. code-block:: python3

    run(construct_arrays2, {"α0_vals":[0.5, 1.]}, state_vec2)



Only :math:`\rho(s_t)` depends on :math:`s_t`.

.. code-block:: python3

    run(construct_arrays2, {"ρ_vals":[0.5, 0.9]}, state_vec2)



Only :math:`\sigma(s_t)` depends on :math:`s_t`.

.. code-block:: python3

    run(construct_arrays2, {"σ_vals":[0.5, 1.]}, state_vec2)





More examples
=============

The following  lectures describe how Markov jump  linear quadratic dynamic programming  can be used to extend the :cite:`Barro1979` model
of optimal tax-smoothing and government debt in several interesting directions

  1. :doc:`How to Pay for a War: Part 1 <tax_smoothing_1>`

  2. :doc:`How to Pay for a War: Part 2 <tax_smoothing_2>`

  3. :doc:`How to Pay for a War: Part 3 <tax_smoothing_3>`
