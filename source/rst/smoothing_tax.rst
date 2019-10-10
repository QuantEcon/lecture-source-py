.. _smoothing_tax:

.. include:: /_static/includes/header.raw

.. highlight:: python3


******************************************************************
Tax Smoothing with Complete and Incomplete Markets
******************************************************************


.. index::
    single: Smoothing; Tax

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture uses the  library:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
========
This lecture describes two types of tax-smoothing models that are counterparts to the consumption-smoothing models in :doc:`smoothing`.

* one is in the **complete markets** tradition of Lucas and Stokey :cite:`LucasStokey1983`.

* the other is in the **incomplete markets** tradition  of Hall :cite:`Hall1978` and Barro :cite:`Barro1979`.

*Complete markets* allow a  government to buy or sell claims contingent on all possible states of the world.

*Incomplete markets* allow a  government to buy or sell only a limited set of securities, often only a single risk-free security.

Barro :cite:`Barro1979`  worked in an incomplete markets tradition by assuming
that the only asset that can be traded is a risk-free one period bond.

Hall assumed an exogenous stochastic process of nonfinancial income and
an exogenous gross interest rate on one period risk-free debt that equals
:math:`\beta^{-1}`, where :math:`\beta \in (0,1)` is also a consumer's
intertemporal discount factor.

Barro :cite:`Barro1979` made an analogous assumption about the risk-free interest
rate in a tax-smoothing model that turns out to have the same mathematical structure as Hall's
consumption-smoothing model.

To get Barro's model from Hall's, all we have to do is to rename variables

We maintain Hall and Barro's assumption about the interest rate when we describe an
incomplete markets version of our model.

In addition, we extend their assumption about the interest rate to an appropriate counterpart to create a "complete markets" model in the style of
Lucas and Stokey :cite:`LucasStokey1983`.

While in :doc:`smoothing` we  focus on  consumption-smoothing
versions of these models, in this lecture we  study the  tax-smoothing
interpretation.

It is convenient that for each version of a consumption-smoothing model, there is a tax-smoothing counterpart obtained simply by

*  relabeling consumption as tax collections 

*  relabeling a consumer's one-period utility function as a government's one-period loss function from collecting taxes that impose deadweight welfare losses

*  relabeling a consumer's  nonfinancial income as a government's purchases 

*  relabeling a consumer's *debt* as a government's *assets* 


Convenient Isomorphism
-----------------------

We can convert  the consumption-smoothing models in lecture :doc:`smoothing` into  tax-smoothing models by setting 
:math:`c_t = T_t` and :math:`G_t = y_t`, where :math:`T_t` is total tax
collections and :math:`\{G_t\}` is an exogenous government expenditures
process.

For elaborations on this theme, please see :doc:`perm_income_cons` and later parts of this lecture.


We'll spend most of this lecture studying the finite-state Markov specification, 
but will also  treat the linear state space specification.

Let's start with some standard imports:

.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt
    %matplotlib inline
    import scipy.linalg as la



Relationship to Other lectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linear-quadratic versions of the Lucas-Stokey tax-smoothing model are described in :doc:`lqramsey`, which can be viewed a warm-up for a model of tax smoothing described in :doc:`opt_tax_recur`.

* In :doc:`lqramsey` and :doc:`opt_tax_recur`, the government  recognizes that its decisions affect prices.

So these later lectures are partly about how a government  optimally  manipulate prices of government debt, albeit indirectly via the effects that distorting
taxes have on equilibrium prices and allocations



Link to History
^^^^^^^^^^^^^^^^^^

For those who love history, President Thomas Jefferson's Secretary of Treasury Albert Gallatin (1807) :cite:`Gallatin` advocated what 
amounts to Barro's model :cite:`Barro1979`



To exploit the isomorphism between consumption-smoothing and tax-smoothing models, we bring in code from
:doc:`smoothing` 

Code
----

Here's some code that, among other things, contains a function called `consumption_complete()`.

This function computes :math:`\{ b(i) \}_{i=1}^{N}, \bar c` as outcomes given a set of parameters for the general case with :math:`N` Markov states
under the assumption of complete markets

.. code-block:: python3

    class ConsumptionProblem:
        """
        The data for a consumption problem, including some default values.
        """

        def __init__(self,
                     β=.96,
                     y=[2, 1.5],
                     b0=3,
                     P=[[.8, .2],
                        [.4, .6]],
                     init=0):
            """
            Parameters
            ----------

            β : discount factor
            y : list containing the two income levels
            b0 : debt in period 0 (= initial state debt level)
            P : 2x2 transition matrix
            init : index of initial state s0
            """
            self.β = β
            self.y = np.asarray(y)
            self.b0 = b0
            self.P = np.asarray(P)
            self.init = init

        def simulate(self, N_simul=80, random_state=1):
            """
            Parameters
            ----------

            N_simul : number of periods for simulation
            random_state : random state for simulating Markov chain
            """
            # For the simulation define a quantecon MC class
            mc = qe.MarkovChain(self.P)
            s_path = mc.simulate(N_simul, init=self.init, random_state=random_state)

            return s_path


    def consumption_complete(cp):
        """
        Computes endogenous values for the complete market case.

        Parameters
        ----------

        cp : instance of ConsumptionProblem

        Returns
        -------

            c_bar : constant consumption
            b : optimal debt in each state

        associated with the price system

            Q = β * P
        """
        β, P, y, b0, init = cp.β, cp.P, cp.y, cp.b0, cp.init   # Unpack

        Q = β * P                               # assumed price system

        # construct matrices of augmented equation system
        n = P.shape[0] + 1

        y_aug = np.empty((n, 1))
        y_aug[0, 0] = y[init] - b0
        y_aug[1:, 0] = y

        Q_aug = np.zeros((n, n))
        Q_aug[0, 1:] = Q[init, :]
        Q_aug[1:, 1:] = Q

        A = np.zeros((n, n))
        A[:, 0] = 1
        A[1:, 1:] = np.eye(n-1)

        x = np.linalg.inv(A - Q_aug) @ y_aug

        c_bar = x[0, 0]
        b = x[1:, 0]

        return c_bar, b


    def consumption_incomplete(cp, s_path):
        """
        Computes endogenous values for the incomplete market case.

        Parameters
        ----------

        cp : instance of ConsumptionProblem
        s_path : the path of states
        """
        β, P, y, b0 = cp.β, cp.P, cp.y, cp.b0  # Unpack

        N_simul = len(s_path)

        # Useful variables
        n = len(y)
        y.shape = (n, 1)
        v = np.linalg.inv(np.eye(n) - β * P) @ y

        # Store consumption and debt path
        b_path, c_path = np.ones(N_simul+1), np.ones(N_simul)
        b_path[0] = b0

        # Optimal decisions from (12) and (13)
        db = ((1 - β) * v - y) / β

        for i, s in enumerate(s_path):
            c_path[i] = (1 - β) * (v - b_path[i] * np.ones((n, 1)))[s, 0]
            b_path[i + 1] = b_path[i] + db[s, 0]

        return c_path, b_path[:-1], y[s_path]






Revisiting the consumption-smoothing model 
---------------------------------------------

It is convenient to remind ourselves of outcomes for the consumption-smoothing model from :doc:`smoothing` by reminding ourselves again that 
the code above also contains a function called `consumption_incomplete()` that uses :eq:`cs_12` and :eq:`cs_13` to

*  simulate paths of :math:`y_t, c_t, b_{t+1}`

*  plot these against values of :math:`\bar c, b(s_1), b(s_2)` found in a corresponding  complete markets economy

Let's try this, using the same parameters in both complete and incomplete markets economies

.. code-block:: python3

    cp = ConsumptionProblem()
    s_path = cp.simulate()
    N_simul = len(s_path)

    c_bar, debt_complete = consumption_complete(cp)

    c_path, debt_path, y_path = consumption_incomplete(cp, s_path)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].set_title('Consumption paths')
    ax[0].plot(np.arange(N_simul), c_path, label='incomplete market')
    ax[0].plot(np.arange(N_simul), c_bar * np.ones(N_simul), label='complete market')
    ax[0].plot(np.arange(N_simul), y_path, label='income', alpha=.6, ls='--')
    ax[0].legend()
    ax[0].set_xlabel('Periods')

    ax[1].set_title('Debt paths')
    ax[1].plot(np.arange(N_simul), debt_path, label='incomplete market')
    ax[1].plot(np.arange(N_simul), debt_complete[s_path], label='complete market')
    ax[1].plot(np.arange(N_simul), y_path, label='income', alpha=.6, ls='--')
    ax[1].legend()
    ax[1].axhline(0, color='k', ls='--')
    ax[1].set_xlabel('Periods')

    plt.show()


In the graph on the left, for the same sample path of nonfinancial
income :math:`y_t`, notice that

*  consumption is constant when there are complete markets, but  takes a random walk in the incomplete markets version of the model.

*  the consumer's debt oscillates between two values that are functions
   of the Markov state in the complete markets model, while the
   consumer's debt drifts in a "unit root" fashion in the incomplete
   markets economy.






Relabeling variables to get tax-smoothing interpretations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


We can  relabel variables to acquire tax-smoothing interpretations of the complete markets and incomplete markets consumption-smoothing models




.. code-block:: python3

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].set_title('Tax collection paths')
    ax[0].plot(np.arange(N_simul), c_path, label='incomplete market')
    ax[0].plot(np.arange(N_simul), c_bar * np.ones(N_simul), label='complete market')
    ax[0].plot(np.arange(N_simul), y_path, label='govt expenditures', alpha=.6, ls='--')
    ax[0].legend()
    ax[0].set_xlabel('Periods')
    ax[0].set_ylim([1.4, 2.1])

    ax[1].set_title('Government assets paths')
    ax[1].plot(np.arange(N_simul), debt_path, label='incomplete market')
    ax[1].plot(np.arange(N_simul), debt_complete[s_path], label='complete market')
    ax[1].plot(np.arange(N_simul), y_path, label='govt expenditures', ls='--')
    ax[1].legend()
    ax[1].axhline(0, color='k', ls='--')
    ax[1].set_xlabel('Periods')

    plt.show()






Example: Tax Smoothing with Complete Markets
============================================

It is instructive  to focus on a simple tax-smoothing example with complete markets.

This example will illustrate how, in a complete markets model like that of Lucas and Stokey :cite:`LucasStokey1983`, the government purchases
insurance from the private sector.

Purchasing insurance  protects the government against having to  raise taxes too high when  emergencies make governments high.

We assume that government expenditures take one of two values :math:`G_1 < G_2`, where Markov state :math:`1` means "peace" and Markov state :math:`2` means "war".

The government budget constraint in Markov state :math:`i` is

.. math::

    T_i + b_i = G_i + \sum_j Q_{ij} b_j


where

.. math::

    Q_{ij} = \beta P_{ij}


is the price of one unit of goods when tomorrow's Markov  state is  :math:`j` and when
today's Markov state is :math:`i`

:math:`b_i` is the quantity of the government's
level of *assets* in Markov state :math:`i`.

That is, :math:`b_i` equals  one-period state-contingent claims owed to the government that fall due at time :math:`t`.

Thus, if :math:`b_i < 0`, it means the government **owes** :math:`-b_i` to the private sector when the economy arrives in Markov state :math:`i`.

In our examples below, this  happens when in a previous war-time  period the government has sold an Arrow securities paying off :math:`- b_i`
in peacetime Markov state :math:`i`




Returns on state-contingent debt
=================================

The *ex post* one-period gross return on the portfolio of government assets  held from state :math:`i` at time :math:`t`
to state :math:`j` at time :math:`t+1`  is

.. math::
    R(j | i) = \frac{b(j) }{ \sum_{j'=1}^N Q_{ij'} b(j') }

where :math:`\sum_{j'=1}^N Q_{ij'} b(j')` is the total government expenditure on one-period state-contingent claims in state :math:`i` at time :math:`t`.

The cumulative return earned from putting :math:`1` unit of time :math:`t` goods into the government portfolio of state-contingent securities at
time :math:`t` and then rolling over the proceeds into the government portfolio each period thereafter is

.. math::
    R^T(s_{t+T}, s_{t+T-1}, \ldots, s_t) \equiv R(s_{t+1} | s_t) R (s_{t+2} | s_{t+1} )
    \cdots R(s_{t+T} | s_{t+T-1} )

Below we define two functions that calculate these return rates.

**Convention:**  When :math:`P_{ij}=0`,  we arbitrarily set :math:`R(j | i)` to be :math:`0`.



.. code-block:: python3

    def ex_post_gross_return(b, cp):
        """
        calculate the ex post one-period gross return on the portfolio
        of government assets, given b and Q.
        """
        Q = cp.β * cp.P

        values = Q @ b

        n = len(b)
        R = np.zeros((n, n))

        for i in range(n):
            ind = cp.P[i, :] != 0
            R[i, ind] = b[ind] / values[i]

        return R

    def cumulative_return(s_path, R):
        """
        compute cumulative return from holding 1 unit market portfolio
        of government bonds, given some simulated state path.
        """
        T = len(s_path)

        RT_path = np.empty(T)
        RT_path[0] = 1
        RT_path[1:] = np.cumprod([R[s_path[t], s_path[t+1]] for t in range(T-1)])

        return RT_path

As above, we'll assume that the initial Markov state is state :math:`1`, which means we start from a state of peace.

The government  then experiences 3 time periods of war and come back to peace again.

The history of states is therefore :math:`\{ peace, war, war, war, peace \}`.

In addition, as indicated above, to simplify our example, we'll set the government's initial
asset level to :math:`1`, so that :math:`b_1 = 1`.

Here's our code to compute a quantitative example  intialized to have government assets being one in an initial peace time state:

.. code-block:: python3

    # Parameters
    β = .96

    # change notation y to g in the tax-smoothing example
    g = [1, 2]
    b0 = 1
    P = np.array([[.8, .2],
                  [.4, .6]])

    cp = ConsumptionProblem(β, g, b0, P)
    Q = β * P

    # change notation c_bar to T_bar in the tax-smoothing example
    T_bar, b = consumption_complete(cp)
    R = ex_post_gross_return(b, cp)
    s_path = [0, 1, 1, 1, 0]
    RT_path = cumulative_return(s_path, R)

    print(f"P \n {P}")
    print(f"Q \n {Q}")
    print(f"Govt expenditures in peace and war = {g}")
    print(f"Constant tax collections = {T_bar}")
    print(f"Govt debts in two states = {-b}")

    msg = """
    Now let's check the government's budget constraint in peace and war.
    Our assumptions imply that the government always purchases 0 units of the
    Arrow peace security.
    """
    print(msg)

    AS1 = Q[0, :] @ b
    # spending on Arrow security
    # since the spending on Arrow peace security is not 0 anymore after we change b0 to 1
    print(f"Spending on Arrow security in peace = {AS1}")
    AS2 = Q[1, :] @ b
    print(f"Spending on Arrow security in war = {AS2}")

    print("")
    # tax collections minus debt levels
    print("Government tax collections minus debt levels in peace and war")
    TB1 = T_bar + b[0]
    print(f"T+b in peace = {TB1}")
    TB2 = T_bar + b[1]
    print(f"T+b in war = {TB2}")

    print("")
    print("Total government spending in peace and war")
    G1 = g[0] + AS1
    G2 = g[1] + AS2
    print(f"Peace = {G1}")
    print(f"War = {G2}")

    print("")
    print("Let's see ex-post and ex-ante returns on Arrow securities")

    Π = np.reciprocal(Q)
    exret = Π
    print(f"Ex-post returns to purchase of Arrow securities = \n {exret}")
    exant = Π * P
    print(f"Ex-ante returns to purchase of Arrow securities \n {exant}")

    print("")
    print("The Ex-post one-period gross return on the portfolio of government assets")
    print(R)

    print("")
    print("The cumulative return earned from holding 1 unit market portfolio of government bonds")
    print(RT_path[-1])


Explanation
-----------

In this example, the government always purchase :math:`1` units of the
Arrow security that pays off in peace time (Markov state :math:`1`).

And it purchases a higher amount of the security that pays off in war
time (Markov state :math:`2`).

We recommend plugging the quantities computed above into the government
budget constraints in the two Markov states and staring.

This is an example in which

*  during peacetime, the government purchases *insurance* against the possibility that war breaks out next period

*  during wartime, the government purchases *insurance* against the possibility that war continues another period

*  the return  on the insurance against war is low so long as peace continues

*  the return  on the insurance against war  is high when war breaks out or continues

*  given the history of states that  we assumed, the value of one unit of the portfolio of government assets will double in the end because of high returns during wartime.

*Exercise:* try changing the Markov transition matrix so that

.. math::

    P = \begin{bmatrix}
            1 & 0 \\
           .2 & .8
        \end{bmatrix}


Also, start the system in Markov state :math:`2` (war) with initial
government assets :math:`- 10`, so that the government starts the
war in debt and :math:`b_2 = -10`.

We provide further examples of tax-smoothing models with a finite Markov state in the lecture :doc:`More Finite Markov Chain Tax-Smoothing Examples <smoothing_tax>`.






More Finite Markov Chain Tax-Smoothing Examples
=================================================

For thinking about some episodes in the fiscal history of the United States, we find it interesting to study a few more examples that we now present.



Here we give more examples of tax-smoothing models with both complete and incomplete markets in an :math:`N` state Markov setting.

These examples differ in how Markov states are jumping between peace and war.

To wrap the procedure of solving models, relabeling the graph so that we record government *debt* rather than *assets*,
and displaying the results, we define a new class below.

.. code-block:: python3

    class TaxSmoothingExample:
        """
        construct a tax-smoothing example, by relabeling consumption problem class.
        """
        def __init__(self, g, P, b0, states, β=.96,
                     init=0, s_path=None, N_simul=80, random_state=1):

            self.states = states # state names

            # if the path of states is not specified
            if s_path is None:
                self.cp = ConsumptionProblem(β, g, b0, P, init=init)
                self.s_path = self.cp.simulate(N_simul=N_simul, random_state=random_state)
            # if the path of states is specified
            else:
                self.cp = ConsumptionProblem(β, g, b0, P, init=s_path[0])
                self.s_path = s_path

            # solve for complete market case
            self.T_bar, self.b = consumption_complete(self.cp)
            self.debt_value = - (β * P @ self.b).T

            # solve for incomplete market case
            self.T_path, self.asset_path, self.g_path = \
                consumption_incomplete(self.cp, self.s_path)

            # calculate returns on state-contingent debt
            self.R = ex_post_gross_return(self.b, self.cp)
            self.RT_path = cumulative_return(self.s_path, self.R)

        def display(self):

            # plot graphs
            N = len(self.T_path)

            plt.figure()
            plt.title('Tax collection paths')
            plt.plot(np.arange(N), self.T_path, label='incomplete market')
            plt.plot(np.arange(N), self.T_bar * np.ones(N), label='complete market')
            plt.plot(np.arange(N), self.g_path, label='govt expenditures', alpha=.6, ls='--')
            plt.legend()
            plt.xlabel('Periods')
            plt.show()

            plt.title('Government debt paths')
            plt.plot(np.arange(N), -self.asset_path, label='incomplete market')
            plt.plot(np.arange(N), -self.b[self.s_path], label='complete market')
            plt.plot(np.arange(N), self.g_path, label='govt expenditures', ls='--')
            plt.plot(np.arange(N), self.debt_value[self.s_path], label="today's value of debts")
            plt.legend()
            plt.axhline(0, color='k', ls='--')
            plt.xlabel('Periods')
            plt.show()

            fig, ax = plt.subplots()
            ax.set_title('Cumulative return path (complete market)')
            line1 = ax.plot(np.arange(N), self.RT_path)[0]
            c1 = line1.get_color()
            ax.set_xlabel('Periods')
            ax.set_ylabel('Cumulative return', color=c1)

            ax_ = ax.twinx()
            ax_._get_lines.prop_cycler = ax._get_lines.prop_cycler
            line2 = ax_.plot(np.arange(N), self.g_path, ls='--')[0]
            c2 = line2.get_color()
            ax_.set_ylabel('Government expenditures', color=c2)

            plt.show()

            # plot detailed information
            Q = self.cp.β * self.cp.P

            print(f"P \n {self.cp.P}")
            print(f"Q \n {Q}")
            print(f"Govt expenditures in {', '.join(self.states)} = {self.cp.y.flatten()}")
            print(f"Constant tax collections = {self.T_bar}")
            print(f"Govt debt in {len(self.states)} states = {-self.b}")

            print("")
            print(f"Government tax collections minus debt levels in {', '.join(self.states)}")
            for i in range(len(self.states)):
                TB = self.T_bar + self.b[i]
                print(f"  T+b in {self.states[i]} = {TB}")

            print("")
            print(f"Total government spending in {', '.join(self.states)}")
            for i in range(len(self.states)):
                G = self.cp.y[i, 0] + Q[i, :] @ self.b
                print(f"  {self.states[i]} = {G}")

            print("")
            print("Let's see ex-post and ex-ante returns on Arrow securities \n")

            print(f"Ex-post returns to purchase of Arrow securities:")
            for i in range(len(self.states)):
                for j in range(len(self.states)):
                    if Q[i, j] != 0.:
                        print(f"  π({self.states[j]}|{self.states[i]}) = {1/Q[i, j]}")

            print("")
            exant = 1 / self.cp.β
            print(f"Ex-ante returns to purchase of Arrow securities = {exant}")

            print("")
            print("The Ex-post one-period gross return on the portfolio of government assets")
            print(self.R)

            print("")
            print("The cumulative return earned from holding 1 unit market portfolio of government bonds")
            print(self.RT_path[-1])

Parameters
----------

.. code-block:: python3

    γ = .1
    λ = .1
    ϕ = .1
    θ = .1
    ψ = .1
    g_L = .5
    g_M = .8
    g_H = 1.2
    β = .96

Example 1
---------

This example is designed to produce some stylized versions of tax, debt, and deficit paths followed by the United States during and
after the Civil War and also during and after World War I.

We set the Markov chain to have three states

.. math::
    P =
    \begin{bmatrix}
        1 - \lambda & \lambda  & 0    \cr
        0           & 1 - \phi & \phi \cr
        0           & 0        & 1
    \end{bmatrix}

where the government expenditure vector  :math:`g = \begin{bmatrix} g_L & g_H & g_M \end{bmatrix}` where :math:`g_L < g_M < g_H`.

We set :math:`b_0 = 1` and assume that the initial Markov state is state :math:`1` so that the system starts off in peace.

These parameters have government expenditure beginning at a low level, surging during the war, then decreasing after the war to a level
that exceeds its prewar level.

(This type of  pattern occurred in the US Civil War and World War I experiences.)

.. code-block:: python3

    g_ex1 = [g_L, g_H, g_M]
    P_ex1 = np.array([[1-λ, λ,  0],
                      [0, 1-ϕ,  ϕ],
                      [0,   0,  1]])
    b0_ex1 = 1
    states_ex1 = ['peace', 'war', 'postwar']

.. code-block:: python3

    ts_ex1 = TaxSmoothingExample(g_ex1, P_ex1, b0_ex1, states_ex1, random_state=1)
    ts_ex1.display()

.. code-block:: python3

    # The following shows the use of the wrapper class when a specific state path is given
    s_path = [0, 0, 1, 1, 2]
    ts_s_path = TaxSmoothingExample(g_ex1, P_ex1, b0_ex1, states_ex1, s_path=s_path)
    ts_s_path.display()

Example 2
---------

This example captures a peace followed by a war, eventually followed by a  permanent peace .

Here we set

.. math::
    P =
    \begin{bmatrix}
        1    & 0        & 0      \cr
        0    & 1-\gamma & \gamma \cr
        \phi & 0        & 1-\phi
    \end{bmatrix}

where the government expenditure vector :math:`g = \begin{bmatrix} g_L & g_L & g_H \end{bmatrix}` where :math:`g_L < g_H`.

We assume :math:`b_0 = 1` and that the initial Markov state is state :math:`2` so that the system starts off in a temporary peace.

.. code-block:: python3

    g_ex2 = [g_L, g_L, g_H]
    P_ex2 = np.array([[1,   0,    0],
                      [0, 1-γ,    γ],
                      [ϕ,   0, 1-ϕ]])
    b0_ex2 = 1
    states_ex2 = ['peace', 'temporary peace', 'war']

.. code-block:: python3

    ts_ex2 = TaxSmoothingExample(g_ex2, P_ex2, b0_ex2, states_ex2, init=1, random_state=1)
    ts_ex2.display()

Example 3
---------

This example features a situation in which one of the states is a war state with no hope of peace next period, while another state
is a war state with a positive probability of peace next period.

The Markov chain is:

.. math::
    P =
    \begin{bmatrix}
   		1 - \lambda & \lambda  & 0      & 0         \cr
        0           & 1 - \phi & \phi   & 0         \cr
        0           & 0        & 1-\psi & \psi      \cr
        \theta      & 0        & 0      & 1 - \theta
    \end{bmatrix}

with government expenditure levels for the four states being
:math:`\begin{bmatrix} g_L & g_L & g_H & g_H \end{bmatrix}` where :math:`g_L < g_H`.


We start with :math:`b_0 = 1` and :math:`s_0 = 1`.

.. code-block:: python3

	g_ex3 = [g_L, g_L, g_H, g_H]
	P_ex3 = np.array([[1-λ,  λ,   0,    0],
	                  [0,  1-ϕ,   ϕ,     0],
	                  [0,    0,  1-ψ,    ψ],
	                  [θ,    0,    0,  1-θ ]])
	b0_ex3 = 1
	states_ex3 = ['peace1', 'peace2', 'war1', 'war2']

.. code-block:: python3

	ts_ex3 = TaxSmoothingExample(g_ex3, P_ex3, b0_ex3, states_ex3, random_state=1)
	ts_ex3.display()


Example 4
---------



Here the Markov chain is:

.. math::
	P =
    \begin{bmatrix}
   		1 - \lambda & \lambda  & 0      & 0          & 0      \cr
		0           & 1 - \phi & \phi   & 0          & 0      \cr
        0           & 0        & 1-\psi & \psi       & 0      \cr
        0           & 0        & 0      & 1 - \theta & \theta \cr
        0           & 0        & 0      & 0          & 1
    \end{bmatrix}

with government expenditure levels for the five states being
:math:`\begin{bmatrix} g_L & g_L & g_H & g_H & g_L \end{bmatrix}` where :math:`g_L < g_H`.

We ssume that :math:`b_0 = 1` and :math:`s_0 = 1`.

.. code-block:: python3

	g_ex4 = [g_L, g_L, g_H, g_H, g_L]
	P_ex4 = np.array([[1-λ,  λ,   0,     0,    0],
	                  [0,  1-ϕ,   ϕ,     0,    0],
	                  [0,    0,  1-ψ,    ψ,    0],
	                  [0,    0,    0,   1-θ,   θ],
	                  [0,    0,    0,     0,   1]])
	b0_ex4 = 1
	states_ex4 = ['peace1', 'peace2', 'war1', 'war2', 'permanent peace']

.. code-block:: python3

	ts_ex4 = TaxSmoothingExample(g_ex4, P_ex4, b0_ex4, states_ex4, random_state=1)
	ts_ex4.display()

Example 5
---------

The  example captures a case when  the system follows a deterministic path from peace to war, and back to peace again.

Since there is no randomness, the outcomes in complete markets setting should be the same as in incomplete markets setting.

The Markov chain is:

.. math::
    P =
    \begin{bmatrix}
   		0 & 1 & 0 & 0 & 0 & 0 & 0 \cr
        0 & 0 & 1 & 0 & 0 & 0 & 0 \cr
        0 & 0 & 0 & 1 & 0 & 0 & 0 \cr
        0 & 0 & 0 & 0 & 1 & 0 & 0 \cr
        0 & 0 & 0 & 0 & 0 & 1 & 0 \cr
        0 & 0 & 0 & 0 & 0 & 0 & 1 \cr
        0 & 0 & 0 & 0 & 0 & 0 & 1 \cr
    \end{bmatrix}

with government expenditure levels for the seven states being
:math:`\begin{bmatrix} g_L & g_L & g_H & g_H &  g_H & g_H & g_L \end{bmatrix}` where
:math:`g_L < g_H`. Assume :math:`b_0 = 1` and :math:`s_0 = 1`.

.. code-block:: python3

	g_ex5 = [g_L, g_L, g_H, g_H, g_H, g_H, g_L]
	P_ex5 = np.array([[0, 1, 0, 0, 0, 0, 0],
	                  [0, 0, 1, 0, 0, 0, 0],
	                  [0, 0, 0, 1, 0, 0, 0],
	                  [0, 0, 0, 0, 1, 0, 0],
	                  [0, 0, 0, 0, 0, 1, 0],
	                  [0, 0, 0, 0, 0, 0, 1],
	                  [0, 0, 0, 0, 0, 0, 1]])
	b0_ex5 = 1
	states_ex5 = ['peace1', 'peace2', 'war1', 'war2', 'war3', 'permanent peace']

.. code-block:: python3

	ts_ex5 = TaxSmoothingExample(g_ex5, P_ex5, b0_ex5, states_ex5, N_simul=7, random_state=1)
	ts_ex5.display()



Tax-smoothing interpretation of continuous-state Gaussian model
----------------------------------------------------------------

In the tax-smoothing interpretation of the  complete markets consumption-smoothing model with a continuous state space that we presented in 
the lecture :doc:`consumption smoothing with complete and incomplete markets<smoothing>`, we simply relabel variables.

Thus,  a government  faces a sequence of budget constraints

.. math::

    T_t + b_t = g_t + \beta \mathbb E_t b_{t+1}, \quad t \geq 0

where :math:`T_t` is tax revenues, :math:`b_t` are receipts at :math:`t` from contingent claims that the government had *purchased* at time :math:`t-1`,
and

.. math::

    \mathbb E_t b_{t+1} \equiv \int q_{t+1}(x_{t+1} | x_t) b_{t+1}(x_{t+1}) d x_{t+1}

is the value of time :math:`t+1` state-contingent claims purchased  by the government  at time :math:`t`


As above with the consumption-smoothing model, we can solve the time :math:`t` budget constraint forward to obtain

.. math::

    b_t = \mathbb E_t  \sum_{j=0}^\infty \beta^j (g_{t+j} - T_{t+j} )

which can be rearranged to become

.. math::

    \mathbb E_t  \sum_{j=0}^\infty \beta^j g_{t+j}  = b_t + \mathbb E_t \sum_{j=0}^\infty \beta^j T_{t+j}

which states that the present value of government purchases equals the value of government assets at :math:`t` plus the present value tax receipts.

With these relabelings, examples presented in :doc:`consumption smoothing with complete and incomplete markets<smoothing>` can be interpreted as tax-smoothing models.




Government Manipulation of Arrow Securities Prices
--------------------------------------------------

In :doc:`optimal taxation in an LQ economy<lqramsey>` and :doc:`recursive optimal taxation <opt_tax_recur>`, we study **complete-markets**
models in which the government recognizes that it can manipulate  Arrow securities prices.


In :doc:`optimal taxation with incomplete markets <amss>`, we study an **incomplete-markets** model in which the government  manipulates asset prices.

